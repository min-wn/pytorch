#include "torch_npu/csrc/core/npu/NPUGraph.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/aten/NPUGeneratorImpl.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

#include <ATen/Functions.h>
#include <ATen/core/dispatch/Dispatcher.h>

namespace c10_npu {

static bool _npu_graphs_debug = false;
constexpr int kSynchronizeBusyWaitMillis = 10;

static void check_dual_stream_sync_streams(
    const NPUStream& main_stream,
    const NPUStream& primary_stream,
    const NPUStream& secondary_stream)
{
    TORCH_CHECK(
        main_stream.device_index() == primary_stream.device_index() &&
        main_stream.device_index() == secondary_stream.device_index(),
        "dual_stream_sync requires streams on the same NPU device.",
        PTA_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        main_stream != primary_stream && main_stream != secondary_stream &&
        primary_stream != secondary_stream,
        "dual_stream_sync requires distinct main, primary, and secondary streams.",
        PTA_ERROR(ErrCode::PARAM));
}

static std::vector<c10::SymInt> to_symint_vector(const std::vector<int64_t>& values)
{
    std::vector<c10::SymInt> result;
    result.reserve(values.size());
    for (const auto value : values) {
        result.emplace_back(value);
    }
    return result;
}

static void check_dual_fia_split(
    const at::Tensor& query,
    const at::Tensor& attention_out,
    int64_t split_start,
    int64_t split_graph_tokens,
    const char* split_name)
{
    TORCH_CHECK(
        query.dim() >= 1 && attention_out.dim() >= 1,
        "dual_fused_infer_attention_score requires query and attention_out with at least one dimension.",
        PTA_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        split_start >= 0 && split_graph_tokens > 0,
        "dual_fused_infer_attention_score requires non-negative split_start and positive split_graph_tokens for ",
        split_name,
        ".",
        PTA_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        split_start + split_graph_tokens <= query.size(0),
        "dual_fused_infer_attention_score ",
        split_name,
        " exceeds query dim0: start=",
        split_start,
        ", graph_tokens=",
        split_graph_tokens,
        ", query_tokens=",
        query.size(0),
        ".",
        PTA_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        split_start + split_graph_tokens <= attention_out.size(0),
        "dual_fused_infer_attention_score ",
        split_name,
        " exceeds attention_out dim0: start=",
        split_start,
        ", graph_tokens=",
        split_graph_tokens,
        ", output_tokens=",
        attention_out.size(0),
        ".",
        PTA_ERROR(ErrCode::PARAM));
}

static void call_fia_out(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const c10::optional<at::Tensor>& atten_mask,
    const c10::optional<at::Tensor>& block_table,
    const std::vector<int64_t>& actual_seq_lengths,
    const std::vector<int64_t>& actual_seq_lengths_kv,
    const c10::optional<at::Tensor>& workspace,
    at::Tensor& attention_out,
    at::Tensor& softmax_lse,
    int64_t num_heads,
    double scale,
    int64_t block_size,
    int64_t num_key_value_heads,
    int64_t sparse_mode,
    const std::string& input_layout,
    int64_t pre_tokens,
    int64_t next_tokens,
    bool softmax_lse_flag)
{
    using FiaOutFn = std::tuple<at::Tensor, at::Tensor> (
        const at::Tensor&,
        const at::Tensor&,
        const at::Tensor&,
        const c10::optional<at::Tensor>&,
        const c10::optional<at::Tensor>&,
        at::OptionalSymIntArrayRef,
        at::OptionalSymIntArrayRef,
        const c10::optional<at::Tensor>&,
        const c10::optional<at::Tensor>&,
        const c10::optional<at::Tensor>&,
        const c10::optional<at::Tensor>&,
        const c10::optional<at::Tensor>&,
        const c10::optional<at::Tensor>&,
        const c10::optional<at::Tensor>&,
        const c10::optional<at::Tensor>&,
        const c10::optional<at::Tensor>&,
        const c10::optional<at::Tensor>&,
        const c10::optional<at::Tensor>&,
        const c10::optional<at::Tensor>&,
        const c10::optional<at::Tensor>&,
        const c10::optional<at::Tensor>&,
        const c10::optional<at::Tensor>&,
        const c10::optional<at::Tensor>&,
        at::OptionalSymIntArrayRef,
        const c10::optional<at::Tensor>&,
        const c10::optional<at::Tensor>&,
        const c10::optional<at::Tensor>&,
        int64_t,
        double,
        int64_t,
        int64_t,
        c10::string_view,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        bool,
        const c10::optional<at::Tensor>&,
        at::TensorList);

    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("npu::npu_fused_infer_attention_score", "out")
                         .typed<FiaOutFn>();
    auto actual_seq_lengths_symint = to_symint_vector(actual_seq_lengths);
    auto actual_seq_lengths_kv_symint = to_symint_vector(actual_seq_lengths_kv);
    std::array<at::Tensor, 2> outputs = {attention_out, softmax_lse};

    op.call(
        query,
        key,
        value,
        c10::nullopt, // pse_shift
        atten_mask,
        at::OptionalSymIntArrayRef(actual_seq_lengths_symint),
        at::OptionalSymIntArrayRef(actual_seq_lengths_kv_symint),
        c10::nullopt, // dequant_scale1
        c10::nullopt, // quant_scale1
        c10::nullopt, // dequant_scale2
        c10::nullopt, // quant_scale2
        c10::nullopt, // quant_offset2
        c10::nullopt, // antiquant_scale
        c10::nullopt, // antiquant_offset
        c10::nullopt, // key_antiquant_scale
        c10::nullopt, // key_antiquant_offset
        c10::nullopt, // value_antiquant_scale
        c10::nullopt, // value_antiquant_offset
        block_table,
        c10::nullopt, // query_padding_size
        c10::nullopt, // kv_padding_size
        c10::nullopt, // key_shared_prefix
        c10::nullopt, // value_shared_prefix
        at::OptionalSymIntArrayRef(c10::nullopt), // actual_shared_prefix_len
        c10::nullopt, // query_rope
        c10::nullopt, // key_rope
        c10::nullopt, // key_rope_antiquant_scale
        num_heads,
        scale,
        pre_tokens,
        next_tokens,
        c10::string_view(input_layout),
        num_key_value_heads,
        sparse_mode,
        0, // inner_precise
        block_size,
        0, // antiquant_mode
        0, // key_antiquant_mode
        0, // value_antiquant_mode
        softmax_lse_flag,
        workspace,
        at::TensorList(outputs));
}

MempoolId_t graph_pool_handle()
{
    // Sets just the second value, to distinguish it from MempoolId_ts created from
    // aclmdlRICaptureGetInfo id_s in capture_begin.
    auto new_pool = c10_npu::MemPool();
    return new_pool.id();
}

DualTaskGroupHandle make_dual_task_group_handle(
    NPUTaskGroupHandle primary,
    NPUTaskGroupHandle secondary)
{
    DualTaskGroupHandle handle;
    handle.primary = primary;
    handle.secondary = secondary;
    return handle;
}

DualStreamSyncHandle::DualStreamSyncHandle()
    : fork_event(ACL_EVENT_EXTERNAL),
      primary_done_event(ACL_EVENT_EXTERNAL),
      secondary_done_event(ACL_EVENT_EXTERNAL)
{
}

std::shared_ptr<DualStreamSyncHandle> dual_stream_sync_begin(
    c10_npu::NPUStream main_stream,
    c10_npu::NPUStream primary_stream,
    c10_npu::NPUStream secondary_stream)
{
    check_dual_stream_sync_streams(main_stream, primary_stream, secondary_stream);
    auto handle = std::make_shared<DualStreamSyncHandle>();
    handle->fork_event.record(main_stream);
    handle->fork_event.block(primary_stream);
    handle->fork_event.block(secondary_stream);
    return handle;
}

void dual_stream_sync_end(
    c10_npu::NPUStream main_stream,
    c10_npu::NPUStream primary_stream,
    c10_npu::NPUStream secondary_stream,
    const std::shared_ptr<DualStreamSyncHandle>& handle)
{
    TORCH_CHECK(handle != nullptr, "dual_stream_sync_end requires a valid handle.", PTA_ERROR(ErrCode::PARAM));
    check_dual_stream_sync_streams(main_stream, primary_stream, secondary_stream);
    handle->primary_done_event.record(primary_stream);
    handle->secondary_done_event.record(secondary_stream);
    handle->primary_done_event.block(main_stream);
    handle->secondary_done_event.block(main_stream);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> dual_fused_infer_attention_score(
    c10_npu::NPUStream main_stream,
    c10_npu::NPUStream primary_stream,
    c10_npu::NPUStream secondary_stream,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const c10::optional<at::Tensor>& atten_mask,
    const c10::optional<at::Tensor>& block_table_0,
    const c10::optional<at::Tensor>& block_table_1,
    const std::vector<int64_t>& actual_seq_lengths_0,
    const std::vector<int64_t>& actual_seq_lengths_1,
    const std::vector<int64_t>& actual_seq_lengths_kv_0,
    const std::vector<int64_t>& actual_seq_lengths_kv_1,
    int64_t split_start_0,
    int64_t split_graph_tokens_0,
    int64_t split_start_1,
    int64_t split_graph_tokens_1,
    const c10::optional<at::Tensor>& workspace_0,
    const c10::optional<at::Tensor>& workspace_1,
    const at::Tensor& attention_out,
    const at::Tensor& softmax_lse_0,
    const at::Tensor& softmax_lse_1,
    int64_t num_heads,
    double scale,
    int64_t block_size,
    int64_t num_key_value_heads,
    int64_t sparse_mode,
    const std::string& input_layout,
    int64_t pre_tokens,
    int64_t next_tokens,
    bool softmax_lse_flag)
{
    TORCH_CHECK(
        input_layout == "TND",
        "dual_fused_infer_attention_score currently supports input_layout='TND' only.",
        PTA_ERROR(ErrCode::PARAM));
    check_dual_stream_sync_streams(main_stream, primary_stream, secondary_stream);
    check_dual_fia_split(query, attention_out, split_start_0, split_graph_tokens_0, "split0");
    check_dual_fia_split(query, attention_out, split_start_1, split_graph_tokens_1, "split1");

    auto query_0 = query.narrow(0, split_start_0, split_graph_tokens_0);
    auto query_1 = query.narrow(0, split_start_1, split_graph_tokens_1);
    auto attention_out_0 = attention_out.narrow(0, split_start_0, split_graph_tokens_0);
    auto attention_out_1 = attention_out.narrow(0, split_start_1, split_graph_tokens_1);
    auto softmax_lse_out_0 = softmax_lse_0;
    auto softmax_lse_out_1 = softmax_lse_1;

    auto handle = dual_stream_sync_begin(main_stream, primary_stream, secondary_stream);
    {
        c10_npu::NPUStreamGuard guard(primary_stream);
        call_fia_out(
            query_0,
            key,
            value,
            atten_mask,
            block_table_0,
            actual_seq_lengths_0,
            actual_seq_lengths_kv_0,
            workspace_0,
            attention_out_0,
            softmax_lse_out_0,
            num_heads,
            scale,
            block_size,
            num_key_value_heads,
            sparse_mode,
            input_layout,
            pre_tokens,
            next_tokens,
            softmax_lse_flag);
    }
    {
        c10_npu::NPUStreamGuard guard(secondary_stream);
        call_fia_out(
            query_1,
            key,
            value,
            atten_mask,
            block_table_1,
            actual_seq_lengths_1,
            actual_seq_lengths_kv_1,
            workspace_1,
            attention_out_1,
            softmax_lse_out_1,
            num_heads,
            scale,
            block_size,
            num_key_value_heads,
            sparse_mode,
            input_layout,
            pre_tokens,
            next_tokens,
            softmax_lse_flag);
    }
    dual_stream_sync_end(main_stream, primary_stream, secondary_stream, handle);
    return std::make_tuple(attention_out, softmax_lse_out_0, softmax_lse_out_1);
}

void graph_task_group_begin(c10_npu::NPUStream stream)
{
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureTaskGrpBegin(stream));
}

NPUTaskGroupHandle graph_task_group_end(c10_npu::NPUStream stream)
{
    aclrtTaskGrp group;
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureTaskGrpEnd(stream, &group));
    NPUTaskGroupHandle handle;
    handle.task_group = group;
    return handle;
}

void graph_task_update_begin(c10_npu::NPUStream stream, NPUTaskGroupHandle handle)
{
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureTaskUpdateBegin(stream, handle.task_group));
}

void graph_task_update_end(c10_npu::NPUStream stream)
{
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureTaskUpdateEnd(stream));
}

static void update_fia_task_group(
    c10_npu::NPUStream update_stream,
    NPUTaskGroupHandle handle,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const c10::optional<at::Tensor>& atten_mask,
    const c10::optional<at::Tensor>& block_table,
    const std::vector<int64_t>& actual_seq_lengths,
    const std::vector<int64_t>& actual_seq_lengths_kv,
    const c10::optional<at::Tensor>& workspace,
    at::Tensor& attention_out,
    at::Tensor& softmax_lse,
    int64_t num_heads,
    double scale,
    int64_t block_size,
    int64_t num_key_value_heads,
    int64_t sparse_mode,
    const std::string& input_layout,
    int64_t pre_tokens,
    int64_t next_tokens,
    bool softmax_lse_flag)
{
    graph_task_update_begin(update_stream, handle);
    try {
        call_fia_out(
            query,
            key,
            value,
            atten_mask,
            block_table,
            actual_seq_lengths,
            actual_seq_lengths_kv,
            workspace,
            attention_out,
            softmax_lse,
            num_heads,
            scale,
            block_size,
            num_key_value_heads,
            sparse_mode,
            input_layout,
            pre_tokens,
            next_tokens,
            softmax_lse_flag);
    } catch (...) {
        graph_task_update_end(update_stream);
        throw;
    }
    graph_task_update_end(update_stream);
}

void dual_fused_infer_attention_score_update(
    c10_npu::NPUStream update_stream,
    DualTaskGroupHandle handle,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const c10::optional<at::Tensor>& atten_mask,
    const c10::optional<at::Tensor>& block_table_0,
    const c10::optional<at::Tensor>& block_table_1,
    const std::vector<int64_t>& actual_seq_lengths_0,
    const std::vector<int64_t>& actual_seq_lengths_1,
    const std::vector<int64_t>& actual_seq_lengths_kv_0,
    const std::vector<int64_t>& actual_seq_lengths_kv_1,
    int64_t split_start_0,
    int64_t split_graph_tokens_0,
    int64_t split_start_1,
    int64_t split_graph_tokens_1,
    const c10::optional<at::Tensor>& workspace_0,
    const c10::optional<at::Tensor>& workspace_1,
    const at::Tensor& attention_out,
    const at::Tensor& softmax_lse_0,
    const at::Tensor& softmax_lse_1,
    int64_t num_heads,
    double scale,
    int64_t block_size,
    int64_t num_key_value_heads,
    int64_t sparse_mode,
    const std::string& input_layout,
    int64_t pre_tokens,
    int64_t next_tokens,
    bool softmax_lse_flag)
{
    TORCH_CHECK(
        input_layout == "TND",
        "dual_fused_infer_attention_score_update currently supports input_layout='TND' only.",
        PTA_ERROR(ErrCode::PARAM));
    check_dual_fia_split(query, attention_out, split_start_0, split_graph_tokens_0, "split0");
    check_dual_fia_split(query, attention_out, split_start_1, split_graph_tokens_1, "split1");

    auto query_0 = query.narrow(0, split_start_0, split_graph_tokens_0);
    auto query_1 = query.narrow(0, split_start_1, split_graph_tokens_1);
    auto attention_out_0 = attention_out.narrow(0, split_start_0, split_graph_tokens_0);
    auto attention_out_1 = attention_out.narrow(0, split_start_1, split_graph_tokens_1);
    auto softmax_lse_out_0 = softmax_lse_0;
    auto softmax_lse_out_1 = softmax_lse_1;

    c10_npu::NPUStreamGuard guard(update_stream);
    update_fia_task_group(
        update_stream,
        handle.primary,
        query_0,
        key,
        value,
        atten_mask,
        block_table_0,
        actual_seq_lengths_0,
        actual_seq_lengths_kv_0,
        workspace_0,
        attention_out_0,
        softmax_lse_out_0,
        num_heads,
        scale,
        block_size,
        num_key_value_heads,
        sparse_mode,
        input_layout,
        pre_tokens,
        next_tokens,
        softmax_lse_flag);
    update_fia_task_group(
        update_stream,
        handle.secondary,
        query_1,
        key,
        value,
        atten_mask,
        block_table_1,
        actual_seq_lengths_1,
        actual_seq_lengths_kv_1,
        workspace_1,
        attention_out_1,
        softmax_lse_out_1,
        num_heads,
        scale,
        block_size,
        num_key_value_heads,
        sparse_mode,
        input_layout,
        pre_tokens,
        next_tokens,
        softmax_lse_flag);
}

void launch_callback(c10_npu::NPUStream stream, NPUCallbackFunc func, void *fnData)
{
    aclrtCallbackBlockType type = aclrtCallbackBlockType::ACL_CALLBACK_BLOCK;
    NPU_CHECK_ERROR(c10_npu::acl::AclrtLaunchCallback(func, fnData, type, stream));
}

void subscribe_report(uint64_t threadId, c10_npu::NPUStream stream)
{
    NPU_CHECK_ERROR(c10_npu::acl::AclrtSubscribeReport(threadId, stream));
}

void unsubscribe_report(uint64_t threadId, c10_npu::NPUStream stream)
{
    NPU_CHECK_ERROR(c10_npu::acl::AclrtUnSubscribeReport(threadId, stream));
}

/**
 * Note [CUDA Graph Wrapper Class]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Q: Why do we need graph capture and launch bindings in Pytorch?
 *    Why can't they live in a user extension, for example?
 *
 * A1: Convenience.
 * A2: To ensure valid numerics on replay, some native CUDA ops (like RNG ops with
 *     CPU statefulness) need cooperation from the capture and replay bindings
 *     (see Note [CUDA Graph-safe RNG states] in CUDAGeneratorImpl.h).
 *
 *     We can't expect users to know about this cooperation.  If users write capture
 *     bindings naively in an extension, they likely won't interact with the native
 *     ops properly.  Their graphs would yield invalid numerics on replay.
 */

/**
 * Note [Interaction with CUDA graph capture] in CUDACachingAllocator.cpp
 * describes memory management for captures.
 */

std::atomic<int> NPUGraph::pending_event_queries = 0;

// Track any outstanding event queries that could happen e.g., in a NCCL watchdog so that they
// can be resolved before the capture begins. Note that event queries are not allowed during a
// graph capture in the default capture mode.
void NPUGraph::inc_pending_event_queries()
{
    pending_event_queries++;
}

void NPUGraph::dec_pending_event_queries()
{
    TORCH_INTERNAL_ASSERT(pending_event_queries > 0,
                          "Attempted to decrement the number of outstanding events to be queried, but it was <= 0.");
    pending_event_queries--;
}

int NPUGraph::num_pending_event_queries()
{
    return pending_event_queries;
}

NPUGraph::NPUGraph()
    // NPUStreams may not be default-constructed.
    : capture_stream_(c10_npu::getCurrentNPUStream()) {
}

void NPUGraph::register_generator_state(c10::intrusive_ptr<at_npu::NPUGeneratorState> state)
{
    captured_generator_states_[std::move(state)] = 0;
}

void NPUGraph::register_generator_state(const at::Generator& generator)
{
    c10::intrusive_ptr<at_npu::NPUGeneratorImpl> npu_gen =
        c10::dynamic_intrusive_pointer_cast<at_npu::NPUGeneratorImpl>(generator.getIntrusivePtr());
    npu_gen->register_graph(this);
}

void NPUGraph::capture_begin(MempoolId_t pool, aclmdlRICaptureMode capture_mode)
{
    static const auto _task_queue_enable = c10_npu::option::OptionsManager::GetTaskQueueEnable();
    TORCH_CHECK(_task_queue_enable != 2,
        "Do not support TASK_QUEUE_ENABLE = 2 during NPU graph capture, please "
        "export TASK_QUEUE_ENABLE=1/0.",
        PTA_ERROR(ErrCode::NOT_SUPPORT));

    TORCH_CHECK(!has_graph_exec_,
                "This NPUGraph instance already owns a captured graph. "
                "To capture a new graph, create a new instance.");

    auto stream = c10_npu::getCurrentNPUStream();

    TORCH_CHECK(stream != c10_npu::getDefaultNPUStream(),
                "NPU graphs must be captured on a non-default stream. "
                "(However, after capture, it's ok to replay them on the "
                "default stream.)");

    // default generator is always registered
    auto* gen = at::get_generator_or_default<at_npu::NPUGeneratorImpl>(c10::nullopt, at_npu::detail::getDefaultNPUGenerator());
    gen->register_graph(this);
    gen->set_secondary_stream_capture_state(false);

    for (auto& [generator_state, wholegraph_increments] : captured_generator_states_) {
        generator_state->capture_prologue();
    }

    capture_stream_ = stream;
    capture_dev_ = c10_npu::current_device();

    if (pool.first != 0 || pool.second != 0) {
        // Either value being nonzero means the user supplied a pool to share.
        // But only one should be nonzero.
        // If pool was created by another graph's capture_begin, first should be nonzero.
        // If pool was created by graph_pool_handle, second should be nonzero.
        TORCH_INTERNAL_ASSERT(!(pool.first && pool.second));
        mempool_id_ = pool;
    } else {
        // User did not ask us to share a mempool. Create graph pool handle using is_user_created=false.
        // Sets just the first value, to distinguish it from MempoolId_ts created by graph_pool_handle().
        auto mempool = c10_npu::MemPool({}, false);
        mempool_id_ = mempool.id();
        TORCH_INTERNAL_ASSERT(mempool_id_.first > 0);
    }

    // Addendum: beginAllocateStreamToPool is now called before cudaStreamBeginCapture to prevent an
    // autograd thread's free() call triggering an invalid cudaEventRecord in the caching allocator
    // due to the capture status being updated _after_ a capture had already started.
    c10_npu::NPUCachingAllocator::beginAllocateToPool(capture_dev_, mempool_id_, [this](aclrtStream stream) {
        aclmdlRICaptureStatus status;
        aclmdlRI model_ri;
        NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureGetInfo(stream, &status, &model_ri));
        return status == aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE && model_ri == model_ri_;
    });

    // At this point, any NCCL watchdogs should be aware that we are in capture mode
    // and therefore should not enqueue any additional work that could be event-queried.
    // We still must wait on any existing work that has not been cleaned up.
    while (num_pending_event_queries()) {
        TORCH_WARN_ONCE("Waiting for pending NCCL work to finish before starting graph capture.");
        std::this_thread::sleep_for(
            std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
    }

    // cudaStreamCaptureModeGlobal is the most conservative option to
    // prevent potentially unsafe CUDA API calls during capture.
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureBegin(capture_stream_, capture_mode));

    aclmdlRICaptureStatus status;
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureGetInfo(stream, &status, &model_ri_));
    TORCH_INTERNAL_ASSERT(status == aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE);
}

void NPUGraph::capture_end()
{
    auto stream = c10_npu::getCurrentNPUStream();

    TORCH_CHECK(stream == capture_stream_,
                "Capture must end on the same stream it began on.");

    aclmdlRI model_ri;
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureEnd(capture_stream_, &model_ri));

    c10_npu::NPUCachingAllocator::endAllocateToPool(capture_dev_, mempool_id_);

    TORCH_CHECK(model_ri == model_ri_, "Invalid end capture model id: ", model_ri);

    // In typical graph usage some tensors (e.g. the tensors used for graph IO) are not freed
    // between replays.
    // If Pytorch compiles and runs with a CUDA 11.4+ toolkit, there's a chance the allocator backend
    // is cudaMallocAsync.
    // cudaMallocAsync is generally graph-safe, but if some tensors are not freed between replays,
    // the graph's internal bookkeeping requires that we instantiate with
    // cudaGraphInstantiateFlagAutoFreeOnLaunch. See
    // cudaGraphLaunch
    // cudaGraphInstantiateWithFlags
    has_graph_exec_ = true;

    for (auto& [generator_state, wholegraph_increments] : captured_generator_states_) {
        wholegraph_increments = generator_state->capture_epilogue();
    }

    uint32_t num_graph_nodes = 0;
}

void NPUGraph::replay()
{
    TORCH_CHECK(has_graph_exec_,
                "Called NPUGraph::replay without a preceding successful capture.");

    c10::OptionalDeviceGuard device_guard{capture_stream_.device()};

    for (auto& [generator_state, wholegraph_increments] : captured_generator_states_) {
        generator_state->replay_prologue(wholegraph_increments);
    }

    // model_ri_ may be replayed in any stream.
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRIExecuteAsync(model_ri_, c10_npu::getCurrentNPUStream()));
}

void NPUGraph::enable_debug_mode()
{
    _npu_graphs_debug = true;
}

void NPUGraph::debug_dump()
{
    if (_npu_graphs_debug) {
        if (has_graph_exec_) {
            TORCH_WARN("DEBUG: calling NPUGraph::debug_dump() for model id ", model_ri_);
            NPU_CHECK_ERROR(c10_npu::acl::AclmdlRIDebugPrint(model_ri_));
        }
    } else {
        TORCH_WARN("NPU Graphs debug not enabled, set with NPUGraph::enable_debug_mode().");
    }
}

void NPUGraph::reset()
{
    // I'd prefer these checks throw exceptions, not print warnings,
    // but the destructor calls reset(), and at least one CI build
    // refuses to compile with a throwing destructor.
    //
    // Instead of calling reset() in the destructor to clean up, I could
    // call reset() in the __del__ method of a thin Python wrapper,
    // in which case reset would be allowed to throw exceptions.
    // But Stackoverflow does not like user-defined __del__.
    // __del__ prevents Graph instances from EVER being garbage collected
    // if they participate in a reference cycle.
    // And exceptions thrown in __del__ only print a warning anyway.
    //
    // Calling reset() in the C++ destructor, with warnings instead of exceptions
    // if calls fail, is the compromise we chose.
    //
    // If capture_begin, the capture, or capture_end failed at some point, this NPUGraph, the generator,
    // and the allocator could end up in all kinds of weird states depending where failure occurred.
    // If the user catches the failure exception in a script, or is running in REPL or (god forbid)
    // a Jupyter notebook, I don't see an easy way for reset() to gracefully fix all such possible error states.
    if (has_graph_exec_) {
        // notifyCaptureDestroy may throw. How should we handle this?
        c10_npu::NPUCachingAllocator::releasePool(capture_dev_, mempool_id_);
        NPU_CHECK_WARN(c10_npu::acl::AclmdlRIDestroy(model_ri_));
        has_graph_exec_ = false;
    }
}

// Returns an id another graph's capture_begin can use to share the same memory pool as this graph.
MempoolId_t NPUGraph::pool()
{
    TORCH_CHECK(has_graph_exec_,
                "Called NPUGraph::pool() without a preceding successful capture.");
    return mempool_id_;
}

NPUGraph::~NPUGraph()
{
    for (auto& [generator_state, wholegraph_increments] : captured_generator_states_) {
        generator_state->unregister_graph(this);
    }
    reset();
}

} // namespace c10_npu
