#include <thread>
#include <utility>
#include <vector>

#include "torch_npu/csrc/npu/Graph.h"

#include "op_plugin/OpApiInterface.h"
#include <pybind11/stl.h>

#include "torch_npu/csrc/core/npu/NPUGraph.h"
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"
#include "torch_npu/csrc/npu/Event.h"
#include "torch_npu/csrc/npu/Stream.h"

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;
static std::map<c10_npu::NPUStream, std::vector<PyFuncStruct *>> callbacks = {};
constexpr int processReportTimeout = 100;
static ThreadArgs* threadArgs = nullptr;
static uint64_t threadId = -1;

c10::optional<at::Tensor> optional_tensor_from_py_object(const py::object& obj)
{
    if (obj.is_none()) {
        return c10::nullopt;
    }
    return obj.cast<at::Tensor>();
}

struct DualSplitFiaUpdateRecord {
    c10_npu::NPUTaskGroupHandle handle;
    c10_npu::NPUEvent* event;
    at::Tensor query;
    at::Tensor key;
    at::Tensor value;
    c10::optional<at::Tensor> atten_mask;
    c10::optional<at::Tensor> block_table;
    std::vector<int64_t> actual_seq_lengths;
    std::vector<int64_t> actual_seq_lengths_kv;
    c10::optional<at::Tensor> workspace;
    at::Tensor attention_out;
    at::Tensor softmax_lse;
    int64_t num_heads;
    double scale;
    int64_t block_size;
    int64_t num_key_value_heads;
    int64_t sparse_mode;
    std::string input_layout;
    int64_t pre_tokens;
    int64_t next_tokens;
    bool softmax_lse_flag;
};

py::sequence require_py_sequence(const py::handle& obj, const char* name)
{
    TORCH_CHECK(PySequence_Check(obj.ptr()), name, " must be a sequence.",
        PTA_ERROR(ErrCode::PARAM));
    return py::reinterpret_borrow<py::sequence>(obj);
}

DualSplitFiaUpdateRecord parse_dual_split_fia_update_record(
    const py::handle& obj,
    const char* name)
{
    auto record = require_py_sequence(obj, name);
    TORCH_CHECK(record.size() == 21, name,
        " must contain 21 fields, but got ", record.size(),
        PTA_ERROR(ErrCode::PARAM));
    return DualSplitFiaUpdateRecord{
        record[0].cast<c10_npu::NPUTaskGroupHandle>(),
        THNPUtils_PyObject_to_NPUEvent(record[1].ptr()),
        record[2].cast<at::Tensor>(),
        record[3].cast<at::Tensor>(),
        record[4].cast<at::Tensor>(),
        optional_tensor_from_py_object(record[5]),
        optional_tensor_from_py_object(record[6]),
        record[7].cast<std::vector<int64_t>>(),
        record[8].cast<std::vector<int64_t>>(),
        optional_tensor_from_py_object(record[9]),
        record[10].cast<at::Tensor>(),
        record[11].cast<at::Tensor>(),
        record[12].cast<int64_t>(),
        record[13].cast<double>(),
        record[14].cast<int64_t>(),
        record[15].cast<int64_t>(),
        record[16].cast<int64_t>(),
        record[17].cast<std::string>(),
        record[18].cast<int64_t>(),
        record[19].cast<int64_t>(),
        record[20].cast<bool>()};
}

void run_dual_split_fia_update_pair(
    const c10_npu::NPUStream& update_stream,
    const DualSplitFiaUpdateRecord& first,
    const DualSplitFiaUpdateRecord& second)
{
    c10_npu::dual_split_fused_infer_attention_score_update(
        update_stream,
        first.handle,
        first.event,
        second.handle,
        second.event,
        first.query,
        first.key,
        first.value,
        first.atten_mask,
        first.block_table,
        first.actual_seq_lengths,
        first.actual_seq_lengths_kv,
        first.workspace,
        first.attention_out,
        first.softmax_lse,
        first.num_heads,
        first.scale,
        first.block_size,
        first.num_key_value_heads,
        first.sparse_mode,
        first.input_layout,
        first.pre_tokens,
        first.next_tokens,
        first.softmax_lse_flag,
        second.query,
        second.key,
        second.value,
        second.atten_mask,
        second.block_table,
        second.actual_seq_lengths,
        second.actual_seq_lengths_kv,
        second.workspace,
        second.attention_out,
        second.softmax_lse,
        second.num_heads,
        second.scale,
        second.block_size,
        second.num_key_value_heads,
        second.sparse_mode,
        second.input_layout,
        second.pre_tokens,
        second.next_tokens,
        second.softmax_lse_flag);
}

void *process_callback(void *arg)
{
    ThreadArgs* args = static_cast<ThreadArgs *>(arg);
    auto ret = aclrtSetCurrentContext(args->context);
    while (!args->exitFlag) {
        (void)aclrtProcessReport(processReportTimeout);
    }
    delete args;
    args = nullptr;
    return nullptr;
}

void LaunchCallFunc(void *userData)
{
    PyGILState_STATE state = PyGILState_Ensure();
    if (userData == nullptr) {
        return;
    }
    auto data = (PyFuncStruct *)(userData);
    PyObject *argslist = Py_BuildValue("(O)", data->pyFuncArgs);
    if (argslist == nullptr) {
        return;
    }
    PyObject *result = PyObject_CallObject(data->pyFunc, argslist);
    if (result == nullptr) {
        return;
    }
    if (argslist != nullptr) {
        Py_XDECREF(argslist);
    }
    if (result != nullptr) {
        Py_XDECREF(result);
    }
    PyGILState_Release(state);
}

class AclSkOptionHelper {
public:
    std::vector<aclskOption> optionsVec;
    std::vector<std::string> stringPool;
    std::vector<std::vector<char*>> ptrArrayPool;
    void processInitOption(const std::string& key, int value)
    {
        static const std::unordered_map<std::string, std::function<void(aclskOption&, int)>> optionHandlers = {
            {"preload_code", [](aclskOption& opt, int val) {
                opt.optionType = aclskOptionType::PRELOAD_CODE;
                opt.preload.preloadMode = static_cast<uint32_t>(val);
            }},
            {"split_mode", [](aclskOption& opt, int val) {
                opt.optionType = aclskOptionType::SPLIT_MODE;
                opt.splitMode.splitCnt = static_cast<uint32_t>(val);
            }},
            {"stream_fusion", [](aclskOption& opt, int val) {
                opt.optionType = aclskOptionType::STREAM_FUSION;
                opt.streamFusion.streamFusion = static_cast<uint32_t>(val);
            }},
            {"debug_sync_all", [](aclskOption& opt, int val) {
                opt.optionType = aclskOptionType::DEBUG_SYNC_ALL;
                opt.debugSync.debugSyncAll = static_cast<uint32_t>(val);
            }}
        };

        auto it = optionHandlers.find(key);
        if (it != optionHandlers.end()) {
            aclskOption opt = {};
            it->second(opt, value);
            optionsVec.push_back(opt);
        }
    }

    void processStringArrayOption(const std::string& key, const std::vector<std::string>& values)
    {
        if (key == "debug_dcci_disable_on_kernel") {
            processDcciDisableOnKernel(values);
        }
    }

    aclskOptions getStruct()
    {
        aclskOptions finalOpt = {};
        finalOpt.options = optionsVec.data();
        finalOpt.numOptions = optionsVec.size();
        return finalOpt;
    }

private:
    void processDcciDisableOnKernel(const std::vector<std::string>& values)
    {
        aclskOption opt = {};
        opt.optionType = aclskOptionType::DEBUG_DCCI_DISABLE_ON_KERNEL;
        opt.disableKernelDcci.kernelCnt = static_cast<int>(values.size());
        opt.disableKernelDcci.kernelNames = convertStringArray(values);
        optionsVec.push_back(opt);
    }

    char** convertStringArray(const std::vector<std::string>& values)
    {
        std::vector<char*> charPtrs;
        charPtrs.reserve(values.size());
        for (const auto& token : values) {
            stringPool.push_back(token);
            charPtrs.push_back(const_cast<char*>(stringPool.back().c_str()));
        }
        ptrArrayPool.push_back(std::move(charPtrs));
        return ptrArrayPool.back().data();
    }
};

void TORCH_NPU_API THNPGraph_init(PyObject* module) {
    // Pybind11 patch notes say "py::module_" is more up-to-date syntax,
    // but CI linter and some builds prefer "module".
    auto torch_N_m = py::handle(module).cast<py::module>();

    py::class_<c10_npu::NPUTaskGroupHandle>(torch_N_m, "_NPUTaskGroupHandle")
            .def_readonly("task_group", &c10_npu::NPUTaskGroupHandle::task_group);

    py::class_<c10_npu::DualTaskGroupHandle>(torch_N_m, "_DualTaskGroupHandle")
            .def_readonly("primary", &c10_npu::DualTaskGroupHandle::primary)
            .def_readonly("secondary", &c10_npu::DualTaskGroupHandle::secondary);

    shared_ptr_class_<c10_npu::DualStreamSyncHandle>(torch_N_m, "_DualStreamSyncHandle");

    torch_N_m.def("_graph_pool_handle", &c10_npu::graph_pool_handle)
        .def("_make_dual_task_group_handle", &c10_npu::make_dual_task_group_handle)
        .def("_dual_stream_sync_begin", [](py::object py_main_stream,
                                           py::object py_primary_stream,
                                           py::object py_secondary_stream) {
            auto main_stream = THNPUtils_PyObject_to_NPUStream((*py_main_stream).ptr());
            auto primary_stream = THNPUtils_PyObject_to_NPUStream((*py_primary_stream).ptr());
            auto secondary_stream = THNPUtils_PyObject_to_NPUStream((*py_secondary_stream).ptr());
            py::gil_scoped_release no_gil;
            return c10_npu::dual_stream_sync_begin(main_stream, primary_stream, secondary_stream);
        })
        .def("_dual_stream_sync_end", [](py::object py_main_stream,
                                         py::object py_primary_stream,
                                         py::object py_secondary_stream,
                                         const std::shared_ptr<c10_npu::DualStreamSyncHandle>& handle) {
            auto main_stream = THNPUtils_PyObject_to_NPUStream((*py_main_stream).ptr());
            auto primary_stream = THNPUtils_PyObject_to_NPUStream((*py_primary_stream).ptr());
            auto secondary_stream = THNPUtils_PyObject_to_NPUStream((*py_secondary_stream).ptr());
            py::gil_scoped_release no_gil;
            c10_npu::dual_stream_sync_end(main_stream, primary_stream, secondary_stream, handle);
        })
        .def("_dual_fused_infer_attention_score", [](py::object py_main_stream,
                                                     py::object py_primary_stream,
                                                     py::object py_secondary_stream,
                                                     const at::Tensor& query,
                                                     const at::Tensor& key,
                                                     const at::Tensor& value,
                                                     py::object py_atten_mask,
                                                     py::object py_block_table_0,
                                                     py::object py_block_table_1,
                                                     const std::vector<int64_t>& actual_seq_lengths_0,
                                                     const std::vector<int64_t>& actual_seq_lengths_1,
                                                     const std::vector<int64_t>& actual_seq_lengths_kv_0,
                                                     const std::vector<int64_t>& actual_seq_lengths_kv_1,
                                                     int64_t split_start_0,
                                                     int64_t split_graph_tokens_0,
                                                     int64_t split_start_1,
                                                     int64_t split_graph_tokens_1,
                                                     py::object py_workspace_0,
                                                     py::object py_workspace_1,
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
                                                     bool softmax_lse_flag) {
            auto main_stream = THNPUtils_PyObject_to_NPUStream((*py_main_stream).ptr());
            auto primary_stream = THNPUtils_PyObject_to_NPUStream((*py_primary_stream).ptr());
            auto secondary_stream = THNPUtils_PyObject_to_NPUStream((*py_secondary_stream).ptr());
            auto atten_mask = optional_tensor_from_py_object(py_atten_mask);
            auto block_table_0 = optional_tensor_from_py_object(py_block_table_0);
            auto block_table_1 = optional_tensor_from_py_object(py_block_table_1);
            auto workspace_0 = optional_tensor_from_py_object(py_workspace_0);
            auto workspace_1 = optional_tensor_from_py_object(py_workspace_1);
            py::gil_scoped_release no_gil;
            return c10_npu::dual_fused_infer_attention_score(
                main_stream,
                primary_stream,
                secondary_stream,
                query,
                key,
                value,
                atten_mask,
                block_table_0,
                block_table_1,
                actual_seq_lengths_0,
                actual_seq_lengths_1,
                actual_seq_lengths_kv_0,
                actual_seq_lengths_kv_1,
                split_start_0,
                split_graph_tokens_0,
                split_start_1,
                split_graph_tokens_1,
                workspace_0,
                workspace_1,
                attention_out,
                softmax_lse_0,
                softmax_lse_1,
                num_heads,
                scale,
                block_size,
                num_key_value_heads,
                sparse_mode,
                input_layout,
                pre_tokens,
                next_tokens,
                softmax_lse_flag);
        })
        .def("_dual_fused_infer_attention_score_update", [](py::object py_update_stream,
                                                            c10_npu::DualTaskGroupHandle handle,
                                                            const at::Tensor& query,
                                                            const at::Tensor& key,
                                                            const at::Tensor& value,
                                                            py::object py_atten_mask,
                                                            py::object py_block_table_0,
                                                            py::object py_block_table_1,
                                                            const std::vector<int64_t>& actual_seq_lengths_0,
                                                            const std::vector<int64_t>& actual_seq_lengths_1,
                                                            const std::vector<int64_t>& actual_seq_lengths_kv_0,
                                                            const std::vector<int64_t>& actual_seq_lengths_kv_1,
                                                            int64_t split_start_0,
                                                            int64_t split_graph_tokens_0,
                                                            int64_t split_start_1,
                                                            int64_t split_graph_tokens_1,
                                                            py::object py_workspace_0,
                                                            py::object py_workspace_1,
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
                                                            bool softmax_lse_flag) {
            auto update_stream = THNPUtils_PyObject_to_NPUStream((*py_update_stream).ptr());
            auto atten_mask = optional_tensor_from_py_object(py_atten_mask);
            auto block_table_0 = optional_tensor_from_py_object(py_block_table_0);
            auto block_table_1 = optional_tensor_from_py_object(py_block_table_1);
            auto workspace_0 = optional_tensor_from_py_object(py_workspace_0);
            auto workspace_1 = optional_tensor_from_py_object(py_workspace_1);
            py::gil_scoped_release no_gil;
            c10_npu::dual_fused_infer_attention_score_update(
                update_stream,
                handle,
                query,
                key,
                value,
                atten_mask,
                block_table_0,
                block_table_1,
                actual_seq_lengths_0,
                actual_seq_lengths_1,
                actual_seq_lengths_kv_0,
                actual_seq_lengths_kv_1,
                split_start_0,
                split_graph_tokens_0,
                split_start_1,
                split_graph_tokens_1,
                workspace_0,
                workspace_1,
                attention_out,
                softmax_lse_0,
                softmax_lse_1,
                num_heads,
                scale,
                block_size,
                num_key_value_heads,
                sparse_mode,
                input_layout,
                pre_tokens,
                next_tokens,
                softmax_lse_flag);
        })
        .def("_dual_split_fused_infer_attention_score_update", [](
                                                            py::object py_update_stream,
                                                            c10_npu::NPUTaskGroupHandle handle_0,
                                                            py::object py_event_0,
                                                            const at::Tensor& query_0,
                                                            const at::Tensor& key_0,
                                                            const at::Tensor& value_0,
                                                            py::object py_atten_mask_0,
                                                            py::object py_block_table_0,
                                                            const std::vector<int64_t>& actual_seq_lengths_0,
                                                            const std::vector<int64_t>& actual_seq_lengths_kv_0,
                                                            py::object py_workspace_0,
                                                            const at::Tensor& attention_out_0,
                                                            const at::Tensor& softmax_lse_0,
                                                            int64_t num_heads_0,
                                                            double scale_0,
                                                            int64_t block_size_0,
                                                            int64_t num_key_value_heads_0,
                                                            int64_t sparse_mode_0,
                                                            const std::string& input_layout_0,
                                                            int64_t pre_tokens_0,
                                                            int64_t next_tokens_0,
                                                            bool softmax_lse_flag_0,
                                                            c10_npu::NPUTaskGroupHandle handle_1,
                                                            py::object py_event_1,
                                                            const at::Tensor& query_1,
                                                            const at::Tensor& key_1,
                                                            const at::Tensor& value_1,
                                                            py::object py_atten_mask_1,
                                                            py::object py_block_table_1,
                                                            const std::vector<int64_t>& actual_seq_lengths_1,
                                                            const std::vector<int64_t>& actual_seq_lengths_kv_1,
                                                            py::object py_workspace_1,
                                                            const at::Tensor& attention_out_1,
                                                            const at::Tensor& softmax_lse_1,
                                                            int64_t num_heads_1,
                                                            double scale_1,
                                                            int64_t block_size_1,
                                                            int64_t num_key_value_heads_1,
                                                            int64_t sparse_mode_1,
                                                            const std::string& input_layout_1,
                                                            int64_t pre_tokens_1,
                                                            int64_t next_tokens_1,
                                                            bool softmax_lse_flag_1) {
            auto update_stream = THNPUtils_PyObject_to_NPUStream((*py_update_stream).ptr());
            auto event_0 = THNPUtils_PyObject_to_NPUEvent((*py_event_0).ptr());
            auto event_1 = THNPUtils_PyObject_to_NPUEvent((*py_event_1).ptr());
            auto atten_mask_0 = optional_tensor_from_py_object(py_atten_mask_0);
            auto block_table_0 = optional_tensor_from_py_object(py_block_table_0);
            auto workspace_0 = optional_tensor_from_py_object(py_workspace_0);
            auto atten_mask_1 = optional_tensor_from_py_object(py_atten_mask_1);
            auto block_table_1 = optional_tensor_from_py_object(py_block_table_1);
            auto workspace_1 = optional_tensor_from_py_object(py_workspace_1);
            py::gil_scoped_release no_gil;
            c10_npu::dual_split_fused_infer_attention_score_update(
                update_stream,
                handle_0,
                event_0,
                handle_1,
                event_1,
                query_0,
                key_0,
                value_0,
                atten_mask_0,
                block_table_0,
                actual_seq_lengths_0,
                actual_seq_lengths_kv_0,
                workspace_0,
                attention_out_0,
                softmax_lse_0,
                num_heads_0,
                scale_0,
                block_size_0,
                num_key_value_heads_0,
                sparse_mode_0,
                input_layout_0,
                pre_tokens_0,
                next_tokens_0,
                softmax_lse_flag_0,
                query_1,
                key_1,
                value_1,
                atten_mask_1,
                block_table_1,
                actual_seq_lengths_1,
                actual_seq_lengths_kv_1,
                workspace_1,
                attention_out_1,
                softmax_lse_1,
                num_heads_1,
                scale_1,
                block_size_1,
                num_key_value_heads_1,
                sparse_mode_1,
                input_layout_1,
                pre_tokens_1,
                next_tokens_1,
                softmax_lse_flag_1);
        })
        .def("_dual_split_fused_infer_attention_score_update_many", [](
                                                            py::object py_update_stream,
                                                            py::object py_records) {
            auto update_stream = THNPUtils_PyObject_to_NPUStream((*py_update_stream).ptr());
            auto records_seq = require_py_sequence(py_records, "records");
            std::vector<std::pair<DualSplitFiaUpdateRecord, DualSplitFiaUpdateRecord>> records;
            records.reserve(records_seq.size());
            for (py::ssize_t i = 0; i < records_seq.size(); ++i) {
                auto pair_seq = require_py_sequence(records_seq[i], "record pair");
                TORCH_CHECK(pair_seq.size() == 2,
                    "record pair must contain two split records, but got ",
                    pair_seq.size(), PTA_ERROR(ErrCode::PARAM));
                records.emplace_back(
                    parse_dual_split_fia_update_record(pair_seq[0], "split0 record"),
                    parse_dual_split_fia_update_record(pair_seq[1], "split1 record"));
            }
            py::gil_scoped_release no_gil;
            for (const auto& record_pair : records) {
                run_dual_split_fia_update_pair(
                    update_stream, record_pair.first, record_pair.second);
            }
        })
        .def("_graph_task_group_begin", [](py::object py_stream) {
            auto stream = (*py_stream).ptr();
            c10_npu::graph_task_group_begin(THNPUtils_PyObject_to_NPUStream(stream));
        })
        .def("_graph_task_group_end", [](py::object py_stream) {
            auto stream = (*py_stream).ptr();
            return c10_npu::graph_task_group_end(THNPUtils_PyObject_to_NPUStream(stream));
        })
        .def("_graph_task_update_begin", [](py::object py_stream, c10_npu::NPUTaskGroupHandle handle) {
            auto stream = (*py_stream).ptr();
            c10_npu::graph_task_update_begin(THNPUtils_PyObject_to_NPUStream(stream), handle);
        })
        .def("_graph_task_update_end", [](py::object py_stream) {
            auto stream = (*py_stream).ptr();
            c10_npu::graph_task_update_end(THNPUtils_PyObject_to_NPUStream(stream));
        })
        .def("_super_kernel_scope_begin", [](const char* scope_name) {
            c10_npu::super_kernel_scope_begin(scope_name);
        })
        .def("_super_kernel_scope_end", [](const char* scope_name) {
            c10_npu::super_kernel_scope_end(scope_name);
        })
        .def("_launch_host_func", [](py::object py_stream, py::object py_func, py::object py_data) {
            auto func = (*py_func).ptr();
            auto userDataList = (*py_data).ptr();
            auto stream = THNPUtils_PyObject_to_NPUStream((*py_stream).ptr());
            PyFuncStruct *data = new(std::nothrow) PyFuncStruct(func, userDataList);
            c10_npu::launch_callback(stream, LaunchCallFunc, data);
            callbacks[stream].emplace_back(data);
        })
        .def("_subscribe_report", [](py::object py_stream) {
            auto stream = (*py_stream).ptr();
            aclrtContext context = aclrtContext();
            NPU_CHECK_ERROR(aclrtGetCurrentContext(&context));
            if ((threadArgs == nullptr) || (threadId == -1)) {
                threadArgs = new ThreadArgs(context, false);
                pthread_create(&threadId, nullptr, process_callback, threadArgs);
            }
            c10_npu::subscribe_report(threadId, THNPUtils_PyObject_to_NPUStream(stream));
        })
        .def("_unsubscribe_report", [](py::object py_stream) {
            auto stream = THNPUtils_PyObject_to_NPUStream((*py_stream).ptr());
            c10_npu::unsubscribe_report(threadId, stream);
            auto it = callbacks.find(stream);
            if (it != callbacks.end()) {
                std::vector<PyFuncStruct *>& funcs = it->second;
                for (PyFuncStruct* func : funcs) {
                    delete func;
                    func = nullptr;
                }
                funcs.clear();
                callbacks.erase(it);
            }
            if (callbacks.empty()) {
                threadArgs->exitFlag = true;
                threadId = -1;
            }
        })
        .def("_npu_fused_infer_attention_score_out_graph", [](
            py::object py_stream,
            c10_npu::NPUTaskGroupHandle handle,
            py::object py_event,
            py::args args,
            py::kwargs kwargs
            ) -> std::tuple<at::Tensor &, at::Tensor &> {
                // 1. 定义parser
                static torch::PythonArgParser parser({
                    "npu_fused_infer_attention_score("
                    "Tensor query, Tensor key, Tensor value, *, "
                    "Tensor? pse_shift=None, "
                    "Tensor? atten_mask=None, "
                    "SymIntArrayRef actual_seq_lengths=None, "
                    "SymIntArrayRef actual_seq_lengths_kv=None, "
                    "Tensor? dequant_scale1=None, "
                    "Tensor? quant_scale1=None, "
                    "Tensor? dequant_scale2=None, "
                    "Tensor? quant_scale2=None, " // 10
                    "Tensor? quant_offset2=None, "
                    "Tensor? antiquant_scale=None, "
                    "Tensor? antiquant_offset=None, "
                    "Tensor? key_antiquant_scale=None, "
                    "Tensor? key_antiquant_offset=None, "
                    "Tensor? value_antiquant_scale=None, "
                    "Tensor? value_antiquant_offset=None, "
                    "Tensor? block_table=None, "
                    "Tensor? query_padding_size=None, "
                    "Tensor? kv_padding_size=None, " // 20
                    "Tensor? key_shared_prefix=None, "
                    "Tensor? value_shared_prefix=None, "
                    "SymIntArrayRef actual_shared_prefix_len=None, "
                    "Tensor? query_rope=None, "
                    "Tensor? key_rope=None, "
                    "Tensor? key_rope_antiquant_scale=None, "
                    "int64_t num_heads=1, "
                    "double scale=1.0, "
                    "int64_t pre_tokens=2147483647, "
                    "int64_t next_tokens=2147483647, " // 30
                    "std::string input_layout=\"BSH\", "
                    "int64_t num_key_value_heads=0, "
                    "int64_t sparse_mode=0, "
                    "int64_t inner_precise=0, "
                    "int64_t block_size=0, "
                    "int64_t antiquant_mode=0, "
                    "int64_t key_antiquant_mode=0, "
                    "int64_t value_antiquant_mode=0, "
                    "bool softmax_lse_flag=False, "
                    "Tensor? workspace=None, " // 40
                    "TensorList out)"
                });

                // 2. 转换为原生 PyObject*
                PyObject* args_ptr = args.ptr();
                PyObject* kwargs_ptr = kwargs.ptr();

                // 3. 解析参数
                torch::ParsedArgs<42> parsed;
                torch::PythonArgs py_args = parser.parse(args_ptr, kwargs_ptr, parsed);

                // 4. 必选参数
                at::Tensor query = py_args.tensor(0);
                at::Tensor key = py_args.tensor(1);
                at::Tensor value = py_args.tensor(2);

                // 5. 可选参数
                c10::optional<at::Tensor> pse_shift = py_args.optionalTensor(3);
                c10::optional<at::Tensor> atten_mask = py_args.optionalTensor(4);
                c10::OptionalArray<c10::SymInt> actual_seq_lengths = py_args.symintlistOptional(5);
                c10::OptionalArray<c10::SymInt> actual_seq_lengths_kv = py_args.symintlistOptional(6);
                c10::optional<at::Tensor> dequant_scale1 = py_args.optionalTensor(7);
                c10::optional<at::Tensor> quant_scale1 = py_args.optionalTensor(8);
                c10::optional<at::Tensor> dequant_scale2 = py_args.optionalTensor(9);
                c10::optional<at::Tensor> quant_scale2 = py_args.optionalTensor(10);
                c10::optional<at::Tensor> quant_offset2 = py_args.optionalTensor(11);
                c10::optional<at::Tensor> antiquant_scale = py_args.optionalTensor(12);
                c10::optional<at::Tensor> antiquant_offset = py_args.optionalTensor(13);
                c10::optional<at::Tensor> key_antiquant_scale = py_args.optionalTensor(14);
                c10::optional<at::Tensor> key_antiquant_offset = py_args.optionalTensor(15);
                c10::optional<at::Tensor> value_antiquant_scale = py_args.optionalTensor(16);
                c10::optional<at::Tensor> value_antiquant_offset = py_args.optionalTensor(17);
                c10::optional<at::Tensor> block_table = py_args.optionalTensor(18);
                c10::optional<at::Tensor> query_padding_size = py_args.optionalTensor(19);
                c10::optional<at::Tensor> kv_padding_size = py_args.optionalTensor(20);
                c10::optional<at::Tensor> key_shared_prefix = py_args.optionalTensor(21);
                c10::optional<at::Tensor> value_shared_prefix = py_args.optionalTensor(22);
                c10::OptionalArray<c10::SymInt> actual_shared_prefix_len = py_args.symintlistOptional(23);
                c10::optional<at::Tensor> query_rope = py_args.optionalTensor(24);
                c10::optional<at::Tensor> key_rope = py_args.optionalTensor(25);
                c10::optional<at::Tensor> key_rope_antiquant_scale = py_args.optionalTensor(26);
                int64_t num_heads = py_args.toInt64(27);
                double scale = py_args.toDouble(28);
                int64_t pre_tokens = py_args.toInt64(29);
                int64_t next_tokens = py_args.toInt64(30);
                std::string input_layout = py_args.string(31);
                int64_t num_key_value_heads = py_args.toInt64(32);
                int64_t sparse_mode = py_args.toInt64(33);
                int64_t inner_precise = py_args.toInt64(34);
                int64_t block_size = py_args.toInt64(35);
                int64_t antiquant_mode = py_args.toInt64(36);
                int64_t key_antiquant_mode = py_args.toInt64(37);
                int64_t value_antiquant_mode = py_args.toInt64(38);
                bool softmax_lse_flag = py_args.toBool(39);
                c10::optional<at::Tensor> workspace = py_args.optionalTensor(40);
                std::vector<at::Tensor> out = py_args.tensorlist(41);
                TORCH_CHECK(out.size() == 2,
                    "out must have 2 tensors (attention_out, softmax_lse), but got ",
                    out.size(), PTA_ERROR(ErrCode::PARAM));
                at::Tensor attention_out = out[0];
                at::Tensor softmax_lse = out[1];

                auto stream = THNPUtils_PyObject_to_NPUStream((*py_stream).ptr());
                auto event_ptr = THNPUtils_PyObject_to_NPUEvent((*py_event).ptr());
                pybind11::gil_scoped_release no_gil;

                c10_npu::graph_task_update_begin(stream, handle);

                auto fia_result = op_api::npu_fused_infer_attention_score_out_symint(
                    query, key, value,
                    pse_shift, atten_mask, actual_seq_lengths, actual_seq_lengths_kv,
                    dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2,
                    antiquant_scale, antiquant_offset, key_antiquant_scale, key_antiquant_offset,
                    value_antiquant_scale, value_antiquant_offset, block_table,
                    query_padding_size, kv_padding_size, key_shared_prefix, value_shared_prefix,
                    actual_shared_prefix_len, query_rope, key_rope, key_rope_antiquant_scale,
                    num_heads, scale, pre_tokens, next_tokens, input_layout,
                    num_key_value_heads, sparse_mode, inner_precise, block_size,
                    antiquant_mode, key_antiquant_mode, value_antiquant_mode,
                    softmax_lse_flag, workspace, attention_out, softmax_lse
                );

                c10_npu::graph_task_update_end(stream);
                event_ptr->record(stream);
                return fia_result;
        });

    shared_ptr_class_<c10_npu::NPUGraph>(torch_N_m, "_NPUGraph")
        .def(py::init<>())
        .def(
            "capture_begin",
            [](c10_npu::NPUGraph& self,
               std::optional<c10_npu::MempoolId_t> pool_opt,
               std::string capture_error_mode,
			   bool report_shape) {
                aclmdlRICaptureMode capture_mode;
                c10_npu::MempoolId_t pool = pool_opt.has_value()
                    ? pool_opt.value() : c10_npu::MempoolId_t{0, 0};
                if (capture_error_mode == "global") {
                    capture_mode = aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_GLOBAL;
                } else if (capture_error_mode == "thread_local") {
                    capture_mode = aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL;
                } else if (capture_error_mode == "relaxed") {
                    capture_mode = aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_RELAXED;
                } else {
                    TORCH_CHECK(
                        false,
                        "Unknown capture error mode. Expected `global`, `thread_local`, or `relaxed`, got ",
                        capture_error_mode);
                }
                return self.capture_begin(pool, capture_mode, report_shape);
            },
            py::arg("pool"),
            py::arg("capture_error_mode"),
            py::arg("report_shape"),
            py::call_guard<py::gil_scoped_release>())
        .def(
            "capture_end",
            torch::wrap_pybind_function_no_gil(&c10_npu::NPUGraph::capture_end))
        .def(
            "register_generator_state",
            [](c10_npu::NPUGraph& self, py::handle raw_generator) {
                auto generator = THPGenerator_Unwrap(raw_generator.ptr());
                // We've unwrapped Python object to C++ object,
                // so we could release GIL before calling into C++
                py::gil_scoped_release release;
                return self.register_generator_state(generator);
            },
            py::arg("generator"))
        .def(
            "replay",
            torch::wrap_pybind_function_no_gil(&c10_npu::NPUGraph::replay))
        .def(
            "reset",
            torch::wrap_pybind_function_no_gil(&c10_npu::NPUGraph::reset))
        .def(
            "pool",
            torch::wrap_pybind_function_no_gil(&c10_npu::NPUGraph::pool))
        .def(
            "debug_dump",
            torch::wrap_pybind_function_no_gil(&c10_npu::NPUGraph::debug_dump),
            py::arg("debug_path"))
        .def(
            "enable_debug_mode",
            torch::wrap_pybind_function_no_gil(&c10_npu::NPUGraph::enable_debug_mode))
        .def(
            "super_kernel_optimize",
            [](c10_npu::NPUGraph& self,
               py::object optimize_options,
               py::object debug_options) {
                AclSkOptionHelper helper;
                if (!optimize_options.is_none()) {
                    auto opts = optimize_options.cast<py::dict>();
                    for (auto item : opts) {
                        std::string key = py::str(item.first);
                        if (py::isinstance<py::int_>(item.second)) {
                            helper.processInitOption(key, item.second.cast<int>());
                        }
                    }
                }

                if (!debug_options.is_none()) {
                    auto opts = debug_options.cast<py::dict>();
                    for (auto item : opts) {
                        std::string key = py::str(item.first);
                        if (py::isinstance<py::int_>(item.second)) {
                            helper.processInitOption(key, item.second.cast<int>());
                        } else if (py::isinstance<py::list>(item.second)) {
                            helper.processStringArrayOption(key, item.second.cast<std::vector<std::string>>());
                        }
                    }
                }
                aclskOptions options = helper.getStruct();
                {
                    py::gil_scoped_release release;
                    return self.super_kernel_optimize(&options);
                }
            },
            py::arg("optimize_options"),
            py::arg("debug_options"));
}
