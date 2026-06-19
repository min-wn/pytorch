#include <torch/csrc/dynamo/compiled_autograd.h>

#include "torch_npu/csrc/framework/autograd/FunctionsManual.h"

#include "torch_npu/csrc/aten/CustomFunctions.h"

// @generated from ../../../../../../../vllm-workspace/pytorch/codegen/autograd/templates/Functions.cpp

// The manual function definitions that used to be here are now in torch/csrc/autograd/FunctionsManual.cpp
// This speeds up re-compilation and allow to share these implementations so that they can be
// used for forward mode AD formulas as well.

using namespace at_npu::autograd::generated::details;
using namespace at_npu::native::custom_ops;
using at::Tensor;
using at::Scalar;
using at::IntArrayRef;
using at::TensorList;

namespace at_npu { namespace autograd { namespace generated {

static at::IValue compute_output_metadata(const torch::autograd::edge_list& next_edges)
{
    auto output_metadata = torch::dynamo::autograd::IValuePacker<
        std::vector<std::optional<InputMetadata>>>::pack(
            torch::dynamo::autograd::get_input_metadata(next_edges));
    return output_metadata;
}

static C10_NOINLINE variable_list compiled_autograd_apply_functional(
    const PackedArgs& packed_args,
    const edge_list& next_edges,
    SwapSavedVariables& saved,
    const variable_list& grads,
    const std::string& name)
{
    auto output_metadata = compute_output_metadata(next_edges);
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    return pyinterface->call_function(
        saved.get_py_compiler(),
        "apply_functional",
        name,
        grads,
        packed_args.vec(),
        output_metadata);
}

static variable_list NpuGatherSparseIndexBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,1> needs_input_grad, Tensor& index, std::vector<c10::SymInt>& input_sym_sizes)
{
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (needs_input_grad[/*input*/0]) {
    auto grad_result = any_grad_defined ? (npu_gather_sparse_index_backward(grad, input_sym_sizes, index)) : Tensor();
    copy_range(grad_inputs, input_ix, grad_result);
  }
  return grad_inputs;
}
inline variable_list NpuGatherSparseIndexBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 1>>();
  auto index = packed_args.unpack<Tensor>();
  auto input_sym_sizes = packed_args.unpack<std::vector<c10::SymInt>>();
  return NpuGatherSparseIndexBackward0_apply_functional(variable_list(grads), needs_input_grad, index, input_sym_sizes);
#endif
}

variable_list NpuGatherSparseIndexBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto index = index_.unpack();
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ input_ix }),
  };
  return NpuGatherSparseIndexBackward0_apply_functional(std::move(grads), needs_input_grad, index, input_sym_sizes);
}

void NpuGatherSparseIndexBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(index_, false);
    args.collect(input_sym_sizes);
}
variable_list NpuGatherSparseIndexBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(index_);
  saved.before(input_sym_sizes);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 1>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<std::vector<c10::SymInt>>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuGatherSparseIndexBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto index = index_.unpack();
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ input_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(index);
  packed_args.pack(input_sym_sizes);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(index_);
  saved.after(input_sym_sizes);
  return output_result;
#endif
}

static variable_list GatherBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,1> needs_input_grad, int64_t& dim, Tensor& index, std::vector<c10::SymInt>& self_sym_sizes, bool& sparse_grad)
{
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (needs_input_grad[/*self*/0]) {
    auto grad_result = any_grad_defined ? (npu_gather_backward(grad, self_sym_sizes, dim, index, sparse_grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
inline variable_list GatherBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 1>>();
  auto dim = packed_args.unpack<int64_t>();
  auto index = packed_args.unpack<Tensor>();
  auto self_sym_sizes = packed_args.unpack<std::vector<c10::SymInt>>();
  auto sparse_grad = packed_args.unpack<bool>();
  return GatherBackward0_apply_functional(variable_list(grads), needs_input_grad, dim, index, self_sym_sizes, sparse_grad);
#endif
}

variable_list GatherBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto index = index_.unpack();
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  return GatherBackward0_apply_functional(std::move(grads), needs_input_grad, dim, index, self_sym_sizes, sparse_grad);
}

void GatherBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(dim);
    args.collect(index_, false);
    args.collect(self_sym_sizes);
    args.collect(sparse_grad);
}
variable_list GatherBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(dim);
  saved.before(index_);
  saved.before(self_sym_sizes);
  saved.before(sparse_grad);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 1>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<std::vector<c10::SymInt>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<bool>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), GatherBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto index = index_.unpack();
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(dim);
  packed_args.pack(index);
  packed_args.pack(self_sym_sizes);
  packed_args.pack(sparse_grad);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(dim);
  saved.after(index_);
  saved.after(self_sym_sizes);
  saved.after(sparse_grad);
  return output_result;
#endif
}

static variable_list DropoutWithByteMaskBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,1> needs_input_grad, double& p, Tensor& result1)
{
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (needs_input_grad[/*self*/0]) {
    auto grad_result = any_grad_defined ? (_dropout_with_byte_mask_backward(grad, result1, p)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
inline variable_list DropoutWithByteMaskBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 1>>();
  auto p = packed_args.unpack<double>();
  auto result1 = packed_args.unpack<Tensor>();
  return DropoutWithByteMaskBackward0_apply_functional(variable_list(grads), needs_input_grad, p, result1);
#endif
}

variable_list DropoutWithByteMaskBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto result1 = result1_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  return DropoutWithByteMaskBackward0_apply_functional(std::move(grads), needs_input_grad, p, result1);
}

void DropoutWithByteMaskBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(p);
    args.collect(result1_, true);
}
variable_list DropoutWithByteMaskBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(p);
  saved.before(result1_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 1>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<double>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), DropoutWithByteMaskBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto result1 = result1_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(p);
  packed_args.pack(result1);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(p);
  saved.after(result1_);
  return output_result;
#endif
}

static variable_list NpuCiouBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,2> needs_input_grad, Tensor& gtboxes, bool& is_cross, int64_t& mode, Tensor& self, bool& trans, Tensor& result1)
{
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto gtboxes_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  if (needs_input_grad[0] || needs_input_grad[1]) {

    auto grad_result = npu_ciou_backward(grad, self, gtboxes, result1, trans, is_cross, mode);
      if (needs_input_grad[/*self*/0]) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (needs_input_grad[/*gtboxes*/1]) {
        copy_range(grad_inputs, gtboxes_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
inline variable_list NpuCiouBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 2>>();
  auto gtboxes = packed_args.unpack<Tensor>();
  auto is_cross = packed_args.unpack<bool>();
  auto mode = packed_args.unpack<int64_t>();
  auto self = packed_args.unpack<Tensor>();
  auto trans = packed_args.unpack<bool>();
  auto result1 = packed_args.unpack<Tensor>();
  return NpuCiouBackward0_apply_functional(variable_list(grads), needs_input_grad, gtboxes, is_cross, mode, self, trans, result1);
#endif
}

variable_list NpuCiouBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto gtboxes = gtboxes_.unpack();
  auto self = self_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto gtboxes_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 2>{
    task_should_compute_output({ self_ix }),
    task_should_compute_output({ gtboxes_ix }),
  };
  return NpuCiouBackward0_apply_functional(std::move(grads), needs_input_grad, gtboxes, is_cross, mode, self, trans, result1);
}

void NpuCiouBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(gtboxes_, false);
    args.collect(is_cross);
    args.collect(mode);
    args.collect(self_, false);
    args.collect(trans);
    args.collect(result1_, true);
}
variable_list NpuCiouBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(gtboxes_);
  saved.before(is_cross);
  saved.before(mode);
  saved.before(self_);
  saved.before(trans);
  saved.before(result1_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 2>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<bool>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<bool>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuCiouBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto gtboxes = gtboxes_.unpack();
  auto self = self_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto gtboxes_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 2>{
    task_should_compute_output({ self_ix }),
    task_should_compute_output({ gtboxes_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(gtboxes);
  packed_args.pack(is_cross);
  packed_args.pack(mode);
  packed_args.pack(self);
  packed_args.pack(trans);
  packed_args.pack(result1);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(gtboxes_);
  saved.after(is_cross);
  saved.after(mode);
  saved.after(self_);
  saved.after(trans);
  saved.after(result1_);
  return output_result;
#endif
}

static variable_list NpuDropoutBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,1> needs_input_grad, double& p, Tensor& result1)
{
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (needs_input_grad[/*self*/0]) {
    auto grad_result = any_grad_defined ? (npu_dropout_backward(grad, result1, p)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
inline variable_list NpuDropoutBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 1>>();
  auto p = packed_args.unpack<double>();
  auto result1 = packed_args.unpack<Tensor>();
  return NpuDropoutBackward0_apply_functional(variable_list(grads), needs_input_grad, p, result1);
#endif
}

variable_list NpuDropoutBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto result1 = result1_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  return NpuDropoutBackward0_apply_functional(std::move(grads), needs_input_grad, p, result1);
}

void NpuDropoutBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(p);
    args.collect(result1_, true);
}
variable_list NpuDropoutBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(p);
  saved.before(result1_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 1>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<double>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuDropoutBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto result1 = result1_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(p);
  packed_args.pack(result1);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(p);
  saved.after(result1_);
  return output_result;
#endif
}

static variable_list NpuFormatCastBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,1> needs_input_grad)
{
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (needs_input_grad[/*self*/0]) {
    auto grad_result = any_grad_defined ? (grad) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
inline variable_list NpuFormatCastBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 1>>();

  return NpuFormatCastBackward0_apply_functional(variable_list(grads), needs_input_grad);
#endif
}

variable_list NpuFormatCastBackward0::apply(variable_list&& grads) {



  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  return NpuFormatCastBackward0_apply_functional(std::move(grads), needs_input_grad);
}

void NpuFormatCastBackward0::compiled_args(CompiledNodeArgs& args) const {

}
variable_list NpuFormatCastBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else


  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 1>>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuFormatCastBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  packed_args.pack(needs_input_grad);


  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());


  return output_result;
#endif
}

static variable_list NpuFormatCastBackward1_apply_functional(
  variable_list&& grads,
  std::array<bool,1> needs_input_grad)
{
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (needs_input_grad[/*self*/0]) {
    auto grad_result = any_grad_defined ? (grad) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
inline variable_list NpuFormatCastBackward1_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 1>>();

  return NpuFormatCastBackward1_apply_functional(variable_list(grads), needs_input_grad);
#endif
}

variable_list NpuFormatCastBackward1::apply(variable_list&& grads) {



  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  return NpuFormatCastBackward1_apply_functional(std::move(grads), needs_input_grad);
}

void NpuFormatCastBackward1::compiled_args(CompiledNodeArgs& args) const {

}
variable_list NpuFormatCastBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else


  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 1>>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuFormatCastBackward1_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  packed_args.pack(needs_input_grad);


  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());


  return output_result;
#endif
}

static variable_list BinaryCrossEntropyWithLogitsBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,1> needs_input_grad, Tensor& pos_weight, int64_t& reduction, Tensor& self, Tensor& target, Tensor& weight)
{
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (needs_input_grad[/*self*/0]) {
    auto grad_result = any_grad_defined ? (npu_binary_cross_entropy_with_logits_backward(grad, self, target, weight, pos_weight, reduction)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
inline variable_list BinaryCrossEntropyWithLogitsBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 1>>();
  auto pos_weight = packed_args.unpack<Tensor>();
  auto reduction = packed_args.unpack<int64_t>();
  auto self = packed_args.unpack<Tensor>();
  auto target = packed_args.unpack<Tensor>();
  auto weight = packed_args.unpack<Tensor>();
  return BinaryCrossEntropyWithLogitsBackward0_apply_functional(variable_list(grads), needs_input_grad, pos_weight, reduction, self, target, weight);
#endif
}

variable_list BinaryCrossEntropyWithLogitsBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto pos_weight = pos_weight_.unpack();
  auto self = self_.unpack();
  auto target = target_.unpack();
  auto weight = weight_.unpack();
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  return BinaryCrossEntropyWithLogitsBackward0_apply_functional(std::move(grads), needs_input_grad, pos_weight, reduction, self, target, weight);
}

void BinaryCrossEntropyWithLogitsBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(pos_weight_, false);
    args.collect(reduction);
    args.collect(self_, false);
    args.collect(target_, false);
    args.collect(weight_, false);
}
variable_list BinaryCrossEntropyWithLogitsBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(pos_weight_);
  saved.before(reduction);
  saved.before(self_);
  saved.before(target_);
  saved.before(weight_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 1>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), BinaryCrossEntropyWithLogitsBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto pos_weight = pos_weight_.unpack();
  auto self = self_.unpack();
  auto target = target_.unpack();
  auto weight = weight_.unpack();
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(pos_weight);
  packed_args.pack(reduction);
  packed_args.pack(self);
  packed_args.pack(target);
  packed_args.pack(weight);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(pos_weight_);
  saved.after(reduction);
  saved.after(self_);
  saved.after(target_);
  saved.after(weight_);
  return output_result;
#endif
}

static variable_list FastGeluBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,1> needs_input_grad, Tensor& self)
{
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (needs_input_grad[/*self*/0]) {
    auto grad_result = any_grad_defined ? (npu_fast_gelu_backward(grad, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
inline variable_list FastGeluBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 1>>();
  auto self = packed_args.unpack<Tensor>();
  return FastGeluBackward0_apply_functional(variable_list(grads), needs_input_grad, self);
#endif
}

variable_list FastGeluBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto self = self_.unpack();
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  return FastGeluBackward0_apply_functional(std::move(grads), needs_input_grad, self);
}

void FastGeluBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(self_, false);
}
variable_list FastGeluBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(self_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 1>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), FastGeluBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto self = self_.unpack();
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(self);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(self_);
  return output_result;
#endif
}

static variable_list KlDivBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,1> needs_input_grad, bool& log_target, int64_t& reduction, Tensor& self, Tensor& target)
{
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (needs_input_grad[/*self*/0]) {
    auto grad_result = any_grad_defined ? (kl_div_backward(grad, self, target, reduction, log_target)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
inline variable_list KlDivBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 1>>();
  auto log_target = packed_args.unpack<bool>();
  auto reduction = packed_args.unpack<int64_t>();
  auto self = packed_args.unpack<Tensor>();
  auto target = packed_args.unpack<Tensor>();
  return KlDivBackward0_apply_functional(variable_list(grads), needs_input_grad, log_target, reduction, self, target);
#endif
}

variable_list KlDivBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto self = self_.unpack();
  auto target = target_.unpack();
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  return KlDivBackward0_apply_functional(std::move(grads), needs_input_grad, log_target, reduction, self, target);
}

void KlDivBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(log_target);
    args.collect(reduction);
    args.collect(self_, false);
    args.collect(target_, false);
}
variable_list KlDivBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(log_target);
  saved.before(reduction);
  saved.before(self_);
  saved.before(target_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 1>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<bool>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), KlDivBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto self = self_.unpack();
  auto target = target_.unpack();
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(log_target);
  packed_args.pack(reduction);
  packed_args.pack(self);
  packed_args.pack(target);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(log_target);
  saved.after(reduction);
  saved.after(self_);
  saved.after(target_);
  return output_result;
#endif
}

static variable_list L1LossBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,2> needs_input_grad, int64_t& reduction, Tensor& self, Tensor& target)
{
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto target_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (needs_input_grad[/*self*/0]) {
    auto grad_result = any_grad_defined ? (l1_loss_backward(grad, self, target, reduction)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (needs_input_grad[/*target*/1]) {
    auto grad_result = any_grad_defined ? (l1_loss_backward(grad, self, target, reduction) * -1) : Tensor();
    copy_range(grad_inputs, target_ix, grad_result);
  }
  return grad_inputs;
}
inline variable_list L1LossBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 2>>();
  auto reduction = packed_args.unpack<int64_t>();
  auto self = packed_args.unpack<Tensor>();
  auto target = packed_args.unpack<Tensor>();
  return L1LossBackward0_apply_functional(variable_list(grads), needs_input_grad, reduction, self, target);
#endif
}

variable_list L1LossBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto self = self_.unpack();
  auto target = target_.unpack();
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto target_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 2>{
    task_should_compute_output({ self_ix }),
    task_should_compute_output({ target_ix }),
  };
  return L1LossBackward0_apply_functional(std::move(grads), needs_input_grad, reduction, self, target);
}

void L1LossBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(reduction);
    args.collect(self_, false);
    args.collect(target_, false);
}
variable_list L1LossBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(reduction);
  saved.before(self_);
  saved.before(target_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 2>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), L1LossBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto self = self_.unpack();
  auto target = target_.unpack();
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto target_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 2>{
    task_should_compute_output({ self_ix }),
    task_should_compute_output({ target_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(reduction);
  packed_args.pack(self);
  packed_args.pack(target);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(reduction);
  saved.after(self_);
  saved.after(target_);
  return output_result;
#endif
}

static variable_list MatmulBackwardBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,3> needs_input_grad, Tensor& grad_out, Tensor& other, Tensor& self)
{
  IndexRangeGenerator gen;
  auto grad_out_ix = gen.range(1);
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (needs_input_grad[0] || needs_input_grad[1] || needs_input_grad[2]) {
      auto grad_input_mask = std::array<bool, 3>{
        needs_input_grad[0],
        needs_input_grad[1],
        needs_input_grad[2],
      };
    auto grad_result = matmul_double_backward(grads[0], grads[1], grad_out, self, other, grad_input_mask);
      if (needs_input_grad[/*grad_out*/0]) {
        copy_range(grad_inputs, grad_out_ix, std::get<0>(grad_result));
      }
      if (needs_input_grad[/*self*/1]) {
        copy_range(grad_inputs, self_ix, std::get<1>(grad_result));
      }
      if (needs_input_grad[/*other*/2]) {
        copy_range(grad_inputs, other_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
inline variable_list MatmulBackwardBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 3>>();
  auto grad_out = packed_args.unpack<Tensor>();
  auto other = packed_args.unpack<Tensor>();
  auto self = packed_args.unpack<Tensor>();
  return MatmulBackwardBackward0_apply_functional(variable_list(grads), needs_input_grad, grad_out, other, self);
#endif
}

variable_list MatmulBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto grad_out = grad_out_.unpack();
  auto other = other_.unpack();
  auto self = self_.unpack();
  IndexRangeGenerator gen;
  auto grad_out_ix = gen.range(1);
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 3>{
    task_should_compute_output({ grad_out_ix }),
    task_should_compute_output({ self_ix }),
    task_should_compute_output({ other_ix }),
  };
  return MatmulBackwardBackward0_apply_functional(std::move(grads), needs_input_grad, grad_out, other, self);
}

void MatmulBackwardBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(grad_out_, false);
    args.collect(other_, false);
    args.collect(self_, false);
}
variable_list MatmulBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(grad_out_);
  saved.before(other_);
  saved.before(self_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 3>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), MatmulBackwardBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto grad_out = grad_out_.unpack();
  auto other = other_.unpack();
  auto self = self_.unpack();
  IndexRangeGenerator gen;
  auto grad_out_ix = gen.range(1);
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 3>{
    task_should_compute_output({ grad_out_ix }),
    task_should_compute_output({ self_ix }),
    task_should_compute_output({ other_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(grad_out);
  packed_args.pack(other);
  packed_args.pack(self);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(grad_out_);
  saved.after(other_);
  saved.after(self_);
  return output_result;
#endif
}

static variable_list NpuAddLayerNormBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,4> needs_input_grad, Tensor& gamma, Tensor& x1, Tensor& x2, Tensor& result1, Tensor& result2)
{
  IndexRangeGenerator gen;
  auto x1_ix = gen.range(1);
  auto x2_ix = gen.range(1);
  auto gamma_ix = gen.range(1);
  auto beta_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (needs_input_grad[0] || needs_input_grad[1] || needs_input_grad[2] || needs_input_grad[3]) {

    auto grad_result = npu_add_layer_norm_backward(grads[0], x1, x2, result2, result1, gamma, grads[1]);
      if (needs_input_grad[/*x1*/0]) {
        copy_range(grad_inputs, x1_ix, std::get<0>(grad_result));
      }
      if (needs_input_grad[/*x2*/1]) {
        copy_range(grad_inputs, x2_ix, std::get<1>(grad_result));
      }
      if (needs_input_grad[/*gamma*/2]) {
        copy_range(grad_inputs, gamma_ix, std::get<2>(grad_result));
      }
      if (needs_input_grad[/*beta*/3]) {
        copy_range(grad_inputs, beta_ix, std::get<3>(grad_result));
      }
  }
  return grad_inputs;
}
inline variable_list NpuAddLayerNormBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 4>>();
  auto gamma = packed_args.unpack<Tensor>();
  auto x1 = packed_args.unpack<Tensor>();
  auto x2 = packed_args.unpack<Tensor>();
  auto result1 = packed_args.unpack<Tensor>();
  auto result2 = packed_args.unpack<Tensor>();
  return NpuAddLayerNormBackward0_apply_functional(variable_list(grads), needs_input_grad, gamma, x1, x2, result1, result2);
#endif
}

variable_list NpuAddLayerNormBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto gamma = gamma_.unpack();
  auto x1 = x1_.unpack();
  auto x2 = x2_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto x1_ix = gen.range(1);
  auto x2_ix = gen.range(1);
  auto gamma_ix = gen.range(1);
  auto beta_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 4>{
    task_should_compute_output({ x1_ix }),
    task_should_compute_output({ x2_ix }),
    task_should_compute_output({ gamma_ix }),
    task_should_compute_output({ beta_ix }),
  };
  return NpuAddLayerNormBackward0_apply_functional(std::move(grads), needs_input_grad, gamma, x1, x2, result1, result2);
}

void NpuAddLayerNormBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(gamma_, false);
    args.collect(x1_, false);
    args.collect(x2_, false);
    args.collect(result1_, true);
    args.collect(result2_, true);
}
variable_list NpuAddLayerNormBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(gamma_);
  saved.before(x1_);
  saved.before(x2_);
  saved.before(result1_);
  saved.before(result2_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 4>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuAddLayerNormBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto gamma = gamma_.unpack();
  auto x1 = x1_.unpack();
  auto x2 = x2_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto x1_ix = gen.range(1);
  auto x2_ix = gen.range(1);
  auto gamma_ix = gen.range(1);
  auto beta_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 4>{
    task_should_compute_output({ x1_ix }),
    task_should_compute_output({ x2_ix }),
    task_should_compute_output({ gamma_ix }),
    task_should_compute_output({ beta_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(gamma);
  packed_args.pack(x1);
  packed_args.pack(x2);
  packed_args.pack(result1);
  packed_args.pack(result2);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(gamma_);
  saved.after(x1_);
  saved.after(x2_);
  saved.after(result1_);
  saved.after(result2_);
  return output_result;
#endif
}

static variable_list NpuGeluBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,1> needs_input_grad, std::string& approximate, Tensor& self)
{
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (needs_input_grad[/*self*/0]) {
    auto grad_result = any_grad_defined ? (npu_gelu_backward(grad, self, approximate)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
inline variable_list NpuGeluBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 1>>();
  auto approximate = packed_args.unpack<std::string>();
  auto self = packed_args.unpack<Tensor>();
  return NpuGeluBackward0_apply_functional(variable_list(grads), needs_input_grad, approximate, self);
#endif
}

variable_list NpuGeluBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto self = self_.unpack();
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  return NpuGeluBackward0_apply_functional(std::move(grads), needs_input_grad, approximate, self);
}

void NpuGeluBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(approximate);
    args.collect(self_, false);
}
variable_list NpuGeluBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(approximate);
  saved.before(self_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 1>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<std::string>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuGeluBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto self = self_.unpack();
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(approximate);
  packed_args.pack(self);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(approximate);
  saved.after(self_);
  return output_result;
#endif
}

static variable_list NpuBmmv2Backward0_apply_functional(
  variable_list&& grads,
  std::array<bool,2> needs_input_grad, Tensor& mat2, std::vector<c10::SymInt>& mat2_sym_sizes, Tensor& self, std::vector<c10::SymInt>& self_sym_sizes)
{
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto mat2_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (needs_input_grad[/*mat2*/1]) {
    auto grad_result = any_grad_defined ? (npu_bmm_v2_mat2_backward(grad, self, mat2, mat2_sym_sizes)) : Tensor();
    copy_range(grad_inputs, mat2_ix, grad_result);
  }
  if (needs_input_grad[/*self*/0]) {
    auto grad_result = any_grad_defined ? (npu_bmm_v2_mat1_backward(grad, self, mat2, self_sym_sizes)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
inline variable_list NpuBmmv2Backward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 2>>();
  auto mat2 = packed_args.unpack<Tensor>();
  auto mat2_sym_sizes = packed_args.unpack<std::vector<c10::SymInt>>();
  auto self = packed_args.unpack<Tensor>();
  auto self_sym_sizes = packed_args.unpack<std::vector<c10::SymInt>>();
  return NpuBmmv2Backward0_apply_functional(variable_list(grads), needs_input_grad, mat2, mat2_sym_sizes, self, self_sym_sizes);
#endif
}

variable_list NpuBmmv2Backward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto mat2 = mat2_.unpack();
  auto self = self_.unpack();
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto mat2_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 2>{
    task_should_compute_output({ self_ix }),
    task_should_compute_output({ mat2_ix }),
  };
  return NpuBmmv2Backward0_apply_functional(std::move(grads), needs_input_grad, mat2, mat2_sym_sizes, self, self_sym_sizes);
}

void NpuBmmv2Backward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(mat2_, false);
    args.collect(mat2_sym_sizes);
    args.collect(self_, false);
    args.collect(self_sym_sizes);
}
variable_list NpuBmmv2Backward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(mat2_);
  saved.before(mat2_sym_sizes);
  saved.before(self_);
  saved.before(self_sym_sizes);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 2>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<std::vector<c10::SymInt>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<std::vector<c10::SymInt>>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuBmmv2Backward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto mat2 = mat2_.unpack();
  auto self = self_.unpack();
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto mat2_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 2>{
    task_should_compute_output({ self_ix }),
    task_should_compute_output({ mat2_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(mat2);
  packed_args.pack(mat2_sym_sizes);
  packed_args.pack(self);
  packed_args.pack(self_sym_sizes);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(mat2_);
  saved.after(mat2_sym_sizes);
  saved.after(self_);
  saved.after(self_sym_sizes);
  return output_result;
#endif
}

static variable_list NpuConfusionTransposeBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,1> needs_input_grad, std::vector<int64_t>& perm, std::vector<c10::SymInt>& self_sym_sizes, bool& transpose_first)
{
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (needs_input_grad[/*self*/0]) {
    auto grad_result = any_grad_defined ? (npu_confusion_transpose_backward(grad, perm, self_sym_sizes, !transpose_first)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
inline variable_list NpuConfusionTransposeBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 1>>();
  auto perm = packed_args.unpack<std::vector<int64_t>>();
  auto self_sym_sizes = packed_args.unpack<std::vector<c10::SymInt>>();
  auto transpose_first = packed_args.unpack<bool>();
  return NpuConfusionTransposeBackward0_apply_functional(variable_list(grads), needs_input_grad, perm, self_sym_sizes, transpose_first);
#endif
}

variable_list NpuConfusionTransposeBackward0::apply(variable_list&& grads) {



  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  return NpuConfusionTransposeBackward0_apply_functional(std::move(grads), needs_input_grad, perm, self_sym_sizes, transpose_first);
}

void NpuConfusionTransposeBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(perm);
    args.collect(self_sym_sizes);
    args.collect(transpose_first);
}
variable_list NpuConfusionTransposeBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(perm);
  saved.before(self_sym_sizes);
  saved.before(transpose_first);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 1>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<std::vector<int64_t>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<std::vector<c10::SymInt>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<bool>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuConfusionTransposeBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(perm);
  packed_args.pack(self_sym_sizes);
  packed_args.pack(transpose_first);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(perm);
  saved.after(self_sym_sizes);
  saved.after(transpose_first);
  return output_result;
#endif
}

static variable_list NpuConvolutionBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,3> needs_input_grad, std::vector<int64_t>& dilation, int64_t& groups, Tensor& input, std::vector<int64_t>& padding, std::vector<int64_t>& stride, Tensor& weight)
{
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  if (needs_input_grad[0] || needs_input_grad[1] || needs_input_grad[2]) {
      auto grad_input_mask = std::array<bool, 3>{
        needs_input_grad[0],
        needs_input_grad[1],
        needs_input_grad[2],
      };
    auto grad_result = npu_convolution_backward(input, grad, weight, stride, padding, dilation, groups, grad_input_mask);
      if (needs_input_grad[/*input*/0]) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (needs_input_grad[/*weight*/1]) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (needs_input_grad[/*bias*/2]) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
inline variable_list NpuConvolutionBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 3>>();
  auto dilation = packed_args.unpack<std::vector<int64_t>>();
  auto groups = packed_args.unpack<int64_t>();
  auto input = packed_args.unpack<Tensor>();
  auto padding = packed_args.unpack<std::vector<int64_t>>();
  auto stride = packed_args.unpack<std::vector<int64_t>>();
  auto weight = packed_args.unpack<Tensor>();
  return NpuConvolutionBackward0_apply_functional(variable_list(grads), needs_input_grad, dilation, groups, input, padding, stride, weight);
#endif
}

variable_list NpuConvolutionBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto input = input_.unpack();
  auto weight = weight_.unpack();
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 3>{
    task_should_compute_output({ input_ix }),
    task_should_compute_output({ weight_ix }),
    task_should_compute_output({ bias_ix }),
  };
  return NpuConvolutionBackward0_apply_functional(std::move(grads), needs_input_grad, dilation, groups, input, padding, stride, weight);
}

void NpuConvolutionBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(dilation);
    args.collect(groups);
    args.collect(input_, false);
    args.collect(padding);
    args.collect(stride);
    args.collect(weight_, false);
}
variable_list NpuConvolutionBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(dilation);
  saved.before(groups);
  saved.before(input_);
  saved.before(padding);
  saved.before(stride);
  saved.before(weight_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 3>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<std::vector<int64_t>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<std::vector<int64_t>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<std::vector<int64_t>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuConvolutionBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto input = input_.unpack();
  auto weight = weight_.unpack();
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 3>{
    task_should_compute_output({ input_ix }),
    task_should_compute_output({ weight_ix }),
    task_should_compute_output({ bias_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(dilation);
  packed_args.pack(groups);
  packed_args.pack(input);
  packed_args.pack(padding);
  packed_args.pack(stride);
  packed_args.pack(weight);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(dilation);
  saved.after(groups);
  saved.after(input_);
  saved.after(padding);
  saved.after(stride);
  saved.after(weight_);
  return output_result;
#endif
}

static variable_list NpuConvolutionTransposeBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,3> needs_input_grad, std::vector<int64_t>& dilation, int64_t& groups, Tensor& input, std::vector<int64_t>& output_padding, std::vector<int64_t>& padding, std::vector<int64_t>& stride, Tensor& weight)
{
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  if (needs_input_grad[0] || needs_input_grad[1] || needs_input_grad[2]) {
      auto grad_input_mask = std::array<bool, 3>{
        needs_input_grad[0],
        needs_input_grad[1],
        needs_input_grad[2],
      };
    auto grad_result = npu_convolution_transpose_backward(input, grad, weight, padding, output_padding, stride, dilation, groups, grad_input_mask);
      if (needs_input_grad[/*input*/0]) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (needs_input_grad[/*weight*/1]) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (needs_input_grad[/*bias*/2]) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
inline variable_list NpuConvolutionTransposeBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 3>>();
  auto dilation = packed_args.unpack<std::vector<int64_t>>();
  auto groups = packed_args.unpack<int64_t>();
  auto input = packed_args.unpack<Tensor>();
  auto output_padding = packed_args.unpack<std::vector<int64_t>>();
  auto padding = packed_args.unpack<std::vector<int64_t>>();
  auto stride = packed_args.unpack<std::vector<int64_t>>();
  auto weight = packed_args.unpack<Tensor>();
  return NpuConvolutionTransposeBackward0_apply_functional(variable_list(grads), needs_input_grad, dilation, groups, input, output_padding, padding, stride, weight);
#endif
}

variable_list NpuConvolutionTransposeBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto input = input_.unpack();
  auto weight = weight_.unpack();
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 3>{
    task_should_compute_output({ input_ix }),
    task_should_compute_output({ weight_ix }),
    task_should_compute_output({ bias_ix }),
  };
  return NpuConvolutionTransposeBackward0_apply_functional(std::move(grads), needs_input_grad, dilation, groups, input, output_padding, padding, stride, weight);
}

void NpuConvolutionTransposeBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(dilation);
    args.collect(groups);
    args.collect(input_, false);
    args.collect(output_padding);
    args.collect(padding);
    args.collect(stride);
    args.collect(weight_, false);
}
variable_list NpuConvolutionTransposeBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(dilation);
  saved.before(groups);
  saved.before(input_);
  saved.before(output_padding);
  saved.before(padding);
  saved.before(stride);
  saved.before(weight_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 3>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<std::vector<int64_t>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<std::vector<int64_t>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<std::vector<int64_t>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<std::vector<int64_t>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuConvolutionTransposeBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto input = input_.unpack();
  auto weight = weight_.unpack();
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 3>{
    task_should_compute_output({ input_ix }),
    task_should_compute_output({ weight_ix }),
    task_should_compute_output({ bias_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(dilation);
  packed_args.pack(groups);
  packed_args.pack(input);
  packed_args.pack(output_padding);
  packed_args.pack(padding);
  packed_args.pack(stride);
  packed_args.pack(weight);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(dilation);
  saved.after(groups);
  saved.after(input_);
  saved.after(output_padding);
  saved.after(padding);
  saved.after(stride);
  saved.after(weight_);
  return output_result;
#endif
}

static variable_list NpuDeepNormBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,4> needs_input_grad, double& alpha, Tensor& gamma, Tensor& gx, Tensor& x, Tensor& result0, Tensor& result1)
{
  IndexRangeGenerator gen;
  auto x_ix = gen.range(1);
  auto gx_ix = gen.range(1);
  auto beta_ix = gen.range(1);
  auto gamma_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  if (needs_input_grad[0] || needs_input_grad[1] || needs_input_grad[2] || needs_input_grad[3]) {

    auto grad_result = npu_deep_norm_backward(grad, x, gx, gamma, result0, result1, alpha);
      if (needs_input_grad[/*x*/0]) {
        copy_range(grad_inputs, x_ix, std::get<0>(grad_result));
      }
      if (needs_input_grad[/*gx*/1]) {
        copy_range(grad_inputs, gx_ix, std::get<1>(grad_result));
      }
      if (needs_input_grad[/*beta*/2]) {
        copy_range(grad_inputs, beta_ix, std::get<2>(grad_result));
      }
      if (needs_input_grad[/*gamma*/3]) {
        copy_range(grad_inputs, gamma_ix, std::get<3>(grad_result));
      }
  }
  return grad_inputs;
}
inline variable_list NpuDeepNormBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 4>>();
  auto alpha = packed_args.unpack<double>();
  auto gamma = packed_args.unpack<Tensor>();
  auto gx = packed_args.unpack<Tensor>();
  auto x = packed_args.unpack<Tensor>();
  auto result0 = packed_args.unpack<Tensor>();
  auto result1 = packed_args.unpack<Tensor>();
  return NpuDeepNormBackward0_apply_functional(variable_list(grads), needs_input_grad, alpha, gamma, gx, x, result0, result1);
#endif
}

variable_list NpuDeepNormBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto gamma = gamma_.unpack();
  auto gx = gx_.unpack();
  auto x = x_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto x_ix = gen.range(1);
  auto gx_ix = gen.range(1);
  auto beta_ix = gen.range(1);
  auto gamma_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 4>{
    task_should_compute_output({ x_ix }),
    task_should_compute_output({ gx_ix }),
    task_should_compute_output({ beta_ix }),
    task_should_compute_output({ gamma_ix }),
  };
  return NpuDeepNormBackward0_apply_functional(std::move(grads), needs_input_grad, alpha, gamma, gx, x, result0, result1);
}

void NpuDeepNormBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(alpha);
    args.collect(gamma_, false);
    args.collect(gx_, false);
    args.collect(x_, false);
    args.collect(result0_, true);
    args.collect(result1_, true);
}
variable_list NpuDeepNormBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(alpha);
  saved.before(gamma_);
  saved.before(gx_);
  saved.before(x_);
  saved.before(result0_);
  saved.before(result1_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 4>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<double>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuDeepNormBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto gamma = gamma_.unpack();
  auto gx = gx_.unpack();
  auto x = x_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto x_ix = gen.range(1);
  auto gx_ix = gen.range(1);
  auto beta_ix = gen.range(1);
  auto gamma_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 4>{
    task_should_compute_output({ x_ix }),
    task_should_compute_output({ gx_ix }),
    task_should_compute_output({ beta_ix }),
    task_should_compute_output({ gamma_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(alpha);
  packed_args.pack(gamma);
  packed_args.pack(gx);
  packed_args.pack(x);
  packed_args.pack(result0);
  packed_args.pack(result1);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(alpha);
  saved.after(gamma_);
  saved.after(gx_);
  saved.after(x_);
  saved.after(result0_);
  saved.after(result1_);
  return output_result;
#endif
}

static variable_list NpuDeformableConv2DBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,4> needs_input_grad, int64_t& deformable_groups, std::vector<int64_t>& dilation, int64_t& groups, Tensor& input, std::vector<int64_t>& kernel_size, bool& modulated, Tensor& offset, std::vector<int64_t>& padding, std::vector<int64_t>& stride, Tensor& weight, Tensor& result1)
{
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto offset_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  if (needs_input_grad[0] || needs_input_grad[1] || needs_input_grad[2] || needs_input_grad[3]) {

    auto grad_result = npu_deformable_conv2dbk(input, grad, result1, weight, offset, kernel_size, stride, padding, dilation, groups, deformable_groups, modulated);
      if (needs_input_grad[/*input*/0]) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (needs_input_grad[/*weight*/1]) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (needs_input_grad[/*offset*/2]) {
        copy_range(grad_inputs, offset_ix, std::get<2>(grad_result));
      }
      if (needs_input_grad[/*bias*/3]) {
        copy_range(grad_inputs, bias_ix, std::get<3>(grad_result));
      }
  }
  return grad_inputs;
}
inline variable_list NpuDeformableConv2DBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 4>>();
  auto deformable_groups = packed_args.unpack<int64_t>();
  auto dilation = packed_args.unpack<std::vector<int64_t>>();
  auto groups = packed_args.unpack<int64_t>();
  auto input = packed_args.unpack<Tensor>();
  auto kernel_size = packed_args.unpack<std::vector<int64_t>>();
  auto modulated = packed_args.unpack<bool>();
  auto offset = packed_args.unpack<Tensor>();
  auto padding = packed_args.unpack<std::vector<int64_t>>();
  auto stride = packed_args.unpack<std::vector<int64_t>>();
  auto weight = packed_args.unpack<Tensor>();
  auto result1 = packed_args.unpack<Tensor>();
  return NpuDeformableConv2DBackward0_apply_functional(variable_list(grads), needs_input_grad, deformable_groups, dilation, groups, input, kernel_size, modulated, offset, padding, stride, weight, result1);
#endif
}

variable_list NpuDeformableConv2DBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto input = input_.unpack();
  auto offset = offset_.unpack();
  auto weight = weight_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto offset_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 4>{
    task_should_compute_output({ input_ix }),
    task_should_compute_output({ weight_ix }),
    task_should_compute_output({ offset_ix }),
    task_should_compute_output({ bias_ix }),
  };
  return NpuDeformableConv2DBackward0_apply_functional(std::move(grads), needs_input_grad, deformable_groups, dilation, groups, input, kernel_size, modulated, offset, padding, stride, weight, result1);
}

void NpuDeformableConv2DBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(deformable_groups);
    args.collect(dilation);
    args.collect(groups);
    args.collect(input_, false);
    args.collect(kernel_size);
    args.collect(modulated);
    args.collect(offset_, false);
    args.collect(padding);
    args.collect(stride);
    args.collect(weight_, false);
    args.collect(result1_, true);
}
variable_list NpuDeformableConv2DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(deformable_groups);
  saved.before(dilation);
  saved.before(groups);
  saved.before(input_);
  saved.before(kernel_size);
  saved.before(modulated);
  saved.before(offset_);
  saved.before(padding);
  saved.before(stride);
  saved.before(weight_);
  saved.before(result1_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 4>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<std::vector<int64_t>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<std::vector<int64_t>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<bool>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<std::vector<int64_t>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<std::vector<int64_t>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuDeformableConv2DBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto input = input_.unpack();
  auto offset = offset_.unpack();
  auto weight = weight_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto offset_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 4>{
    task_should_compute_output({ input_ix }),
    task_should_compute_output({ weight_ix }),
    task_should_compute_output({ offset_ix }),
    task_should_compute_output({ bias_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(deformable_groups);
  packed_args.pack(dilation);
  packed_args.pack(groups);
  packed_args.pack(input);
  packed_args.pack(kernel_size);
  packed_args.pack(modulated);
  packed_args.pack(offset);
  packed_args.pack(padding);
  packed_args.pack(stride);
  packed_args.pack(weight);
  packed_args.pack(result1);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(deformable_groups);
  saved.after(dilation);
  saved.after(groups);
  saved.after(input_);
  saved.after(kernel_size);
  saved.after(modulated);
  saved.after(offset_);
  saved.after(padding);
  saved.after(stride);
  saved.after(weight_);
  saved.after(result1_);
  return output_result;
#endif
}

static variable_list NpuDiouBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,2> needs_input_grad, Tensor& gtboxes, bool& is_cross, int64_t& mode, Tensor& self, bool& trans)
{
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto gtboxes_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  if (needs_input_grad[0] || needs_input_grad[1]) {

    auto grad_result = npu_diou_backward(grad, self, gtboxes, trans, is_cross, mode);
      if (needs_input_grad[/*self*/0]) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (needs_input_grad[/*gtboxes*/1]) {
        copy_range(grad_inputs, gtboxes_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
inline variable_list NpuDiouBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 2>>();
  auto gtboxes = packed_args.unpack<Tensor>();
  auto is_cross = packed_args.unpack<bool>();
  auto mode = packed_args.unpack<int64_t>();
  auto self = packed_args.unpack<Tensor>();
  auto trans = packed_args.unpack<bool>();
  return NpuDiouBackward0_apply_functional(variable_list(grads), needs_input_grad, gtboxes, is_cross, mode, self, trans);
#endif
}

variable_list NpuDiouBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto gtboxes = gtboxes_.unpack();
  auto self = self_.unpack();
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto gtboxes_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 2>{
    task_should_compute_output({ self_ix }),
    task_should_compute_output({ gtboxes_ix }),
  };
  return NpuDiouBackward0_apply_functional(std::move(grads), needs_input_grad, gtboxes, is_cross, mode, self, trans);
}

void NpuDiouBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(gtboxes_, false);
    args.collect(is_cross);
    args.collect(mode);
    args.collect(self_, false);
    args.collect(trans);
}
variable_list NpuDiouBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(gtboxes_);
  saved.before(is_cross);
  saved.before(mode);
  saved.before(self_);
  saved.before(trans);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 2>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<bool>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<bool>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuDiouBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto gtboxes = gtboxes_.unpack();
  auto self = self_.unpack();
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto gtboxes_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 2>{
    task_should_compute_output({ self_ix }),
    task_should_compute_output({ gtboxes_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(gtboxes);
  packed_args.pack(is_cross);
  packed_args.pack(mode);
  packed_args.pack(self);
  packed_args.pack(trans);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(gtboxes_);
  saved.after(is_cross);
  saved.after(mode);
  saved.after(self_);
  saved.after(trans);
  return output_result;
#endif
}

static variable_list NpuDropoutDoMaskBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,1> needs_input_grad, double& p, Tensor& result1)
{
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (needs_input_grad[/*self*/0]) {
    auto grad_result = any_grad_defined ? (npu_dropout_backward(grad, result1, p)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
inline variable_list NpuDropoutDoMaskBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 1>>();
  auto p = packed_args.unpack<double>();
  auto result1 = packed_args.unpack<Tensor>();
  return NpuDropoutDoMaskBackward0_apply_functional(variable_list(grads), needs_input_grad, p, result1);
#endif
}

variable_list NpuDropoutDoMaskBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto result1 = result1_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  return NpuDropoutDoMaskBackward0_apply_functional(std::move(grads), needs_input_grad, p, result1);
}

void NpuDropoutDoMaskBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(p);
    args.collect(result1_, true);
}
variable_list NpuDropoutDoMaskBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(p);
  saved.before(result1_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 1>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<double>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuDropoutDoMaskBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto result1 = result1_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(p);
  packed_args.pack(result1);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(p);
  saved.after(result1_);
  return output_result;
#endif
}

static variable_list NpuDropoutWithAddSoftmaxBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,2> needs_input_grad, at::Scalar& alpha, int64_t& dim, double& prob, Tensor& result0, Tensor& result1)
{
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto x1_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  if (needs_input_grad[0] || needs_input_grad[1]) {

    auto grad_result = npu_dropout_with_add_softmax_backward(grad, result0, result1, alpha, prob, dim);
      if (needs_input_grad[/*self*/0]) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (needs_input_grad[/*x1*/1]) {
        copy_range(grad_inputs, x1_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
inline variable_list NpuDropoutWithAddSoftmaxBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 2>>();
  auto alpha = packed_args.unpack<at::Scalar>();
  auto dim = packed_args.unpack<int64_t>();
  auto prob = packed_args.unpack<double>();
  auto result0 = packed_args.unpack<Tensor>();
  auto result1 = packed_args.unpack<Tensor>();
  return NpuDropoutWithAddSoftmaxBackward0_apply_functional(variable_list(grads), needs_input_grad, alpha, dim, prob, result0, result1);
#endif
}

variable_list NpuDropoutWithAddSoftmaxBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto x1_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 2>{
    task_should_compute_output({ self_ix }),
    task_should_compute_output({ x1_ix }),
  };
  return NpuDropoutWithAddSoftmaxBackward0_apply_functional(std::move(grads), needs_input_grad, alpha, dim, prob, result0, result1);
}

void NpuDropoutWithAddSoftmaxBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(alpha);
    args.collect(dim);
    args.collect(prob);
    args.collect(result0_, true);
    args.collect(result1_, true);
}
variable_list NpuDropoutWithAddSoftmaxBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(alpha);
  saved.before(dim);
  saved.before(prob);
  saved.before(result0_);
  saved.before(result1_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 2>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<at::Scalar>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<double>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuDropoutWithAddSoftmaxBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto x1_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 2>{
    task_should_compute_output({ self_ix }),
    task_should_compute_output({ x1_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(alpha);
  packed_args.pack(dim);
  packed_args.pack(prob);
  packed_args.pack(result0);
  packed_args.pack(result1);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(alpha);
  saved.after(dim);
  saved.after(prob);
  saved.after(result0_);
  saved.after(result1_);
  return output_result;
#endif
}

static variable_list NpuDtypeCastBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,1> needs_input_grad, at::ScalarType& self_scalar_type)
{
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (needs_input_grad[/*self*/0]) {
    auto grad_result = any_grad_defined ? (npu_dtype_cast_backward(grad, self_scalar_type)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
inline variable_list NpuDtypeCastBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 1>>();
  auto self_scalar_type = packed_args.unpack<at::ScalarType>();
  return NpuDtypeCastBackward0_apply_functional(variable_list(grads), needs_input_grad, self_scalar_type);
#endif
}

variable_list NpuDtypeCastBackward0::apply(variable_list&& grads) {



  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  return NpuDtypeCastBackward0_apply_functional(std::move(grads), needs_input_grad, self_scalar_type);
}

void NpuDtypeCastBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(self_scalar_type);
}
variable_list NpuDtypeCastBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(self_scalar_type);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 1>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<at::ScalarType>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuDtypeCastBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(self_scalar_type);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(self_scalar_type);
  return output_result;
#endif
}

static variable_list NpuDtypeCastBackward1_apply_functional(
  variable_list&& grads,
  std::array<bool,1> needs_input_grad, at::ScalarType& self_scalar_type)
{
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (needs_input_grad[/*self*/0]) {
    auto grad_result = any_grad_defined ? (_npu_dtype_cast_backward(grad, self_scalar_type)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
inline variable_list NpuDtypeCastBackward1_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 1>>();
  auto self_scalar_type = packed_args.unpack<at::ScalarType>();
  return NpuDtypeCastBackward1_apply_functional(variable_list(grads), needs_input_grad, self_scalar_type);
#endif
}

variable_list NpuDtypeCastBackward1::apply(variable_list&& grads) {



  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  return NpuDtypeCastBackward1_apply_functional(std::move(grads), needs_input_grad, self_scalar_type);
}

void NpuDtypeCastBackward1::compiled_args(CompiledNodeArgs& args) const {
    args.collect(self_scalar_type);
}
variable_list NpuDtypeCastBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(self_scalar_type);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 1>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<at::ScalarType>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuDtypeCastBackward1_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(self_scalar_type);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(self_scalar_type);
  return output_result;
#endif
}

static variable_list NpuFusedAttentionScoreFwdBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,3> needs_input_grad, bool& dx_transpose, double& keep_prob, Tensor& key_layer, bool& key_transpose, Tensor& query_layer, bool& query_transpose, at::Scalar& scale, Tensor& value_layer, bool& value_transpose, Tensor& result1, Tensor& result2)
{
  IndexRangeGenerator gen;
  auto query_layer_ix = gen.range(1);
  auto key_layer_ix = gen.range(1);
  auto value_layer_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  if (needs_input_grad[0] || needs_input_grad[1] || needs_input_grad[2]) {

    auto grad_result = npu_fused_attention_score_backward(grad, result1, query_layer, key_layer, value_layer, result2, scale, keep_prob, query_transpose, key_transpose, value_transpose, dx_transpose);
      if (needs_input_grad[/*query_layer*/0]) {
        copy_range(grad_inputs, query_layer_ix, std::get<0>(grad_result));
      }
      if (needs_input_grad[/*key_layer*/1]) {
        copy_range(grad_inputs, key_layer_ix, std::get<1>(grad_result));
      }
      if (needs_input_grad[/*value_layer*/2]) {
        copy_range(grad_inputs, value_layer_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
inline variable_list NpuFusedAttentionScoreFwdBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 3>>();
  auto dx_transpose = packed_args.unpack<bool>();
  auto keep_prob = packed_args.unpack<double>();
  auto key_layer = packed_args.unpack<Tensor>();
  auto key_transpose = packed_args.unpack<bool>();
  auto query_layer = packed_args.unpack<Tensor>();
  auto query_transpose = packed_args.unpack<bool>();
  auto scale = packed_args.unpack<at::Scalar>();
  auto value_layer = packed_args.unpack<Tensor>();
  auto value_transpose = packed_args.unpack<bool>();
  auto result1 = packed_args.unpack<Tensor>();
  auto result2 = packed_args.unpack<Tensor>();
  return NpuFusedAttentionScoreFwdBackward0_apply_functional(variable_list(grads), needs_input_grad, dx_transpose, keep_prob, key_layer, key_transpose, query_layer, query_transpose, scale, value_layer, value_transpose, result1, result2);
#endif
}

variable_list NpuFusedAttentionScoreFwdBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto key_layer = key_layer_.unpack();
  auto query_layer = query_layer_.unpack();
  auto value_layer = value_layer_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto query_layer_ix = gen.range(1);
  auto key_layer_ix = gen.range(1);
  auto value_layer_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 3>{
    task_should_compute_output({ query_layer_ix }),
    task_should_compute_output({ key_layer_ix }),
    task_should_compute_output({ value_layer_ix }),
  };
  return NpuFusedAttentionScoreFwdBackward0_apply_functional(std::move(grads), needs_input_grad, dx_transpose, keep_prob, key_layer, key_transpose, query_layer, query_transpose, scale, value_layer, value_transpose, result1, result2);
}

void NpuFusedAttentionScoreFwdBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(dx_transpose);
    args.collect(keep_prob);
    args.collect(key_layer_, false);
    args.collect(key_transpose);
    args.collect(query_layer_, false);
    args.collect(query_transpose);
    args.collect(scale);
    args.collect(value_layer_, false);
    args.collect(value_transpose);
    args.collect(result1_, true);
    args.collect(result2_, true);
}
variable_list NpuFusedAttentionScoreFwdBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(dx_transpose);
  saved.before(keep_prob);
  saved.before(key_layer_);
  saved.before(key_transpose);
  saved.before(query_layer_);
  saved.before(query_transpose);
  saved.before(scale);
  saved.before(value_layer_);
  saved.before(value_transpose);
  saved.before(result1_);
  saved.before(result2_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 3>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<bool>::packed_type(),
      torch::dynamo::autograd::IValuePacker<double>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<bool>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<bool>::packed_type(),
      torch::dynamo::autograd::IValuePacker<at::Scalar>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<bool>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuFusedAttentionScoreFwdBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto key_layer = key_layer_.unpack();
  auto query_layer = query_layer_.unpack();
  auto value_layer = value_layer_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto query_layer_ix = gen.range(1);
  auto key_layer_ix = gen.range(1);
  auto value_layer_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 3>{
    task_should_compute_output({ query_layer_ix }),
    task_should_compute_output({ key_layer_ix }),
    task_should_compute_output({ value_layer_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(dx_transpose);
  packed_args.pack(keep_prob);
  packed_args.pack(key_layer);
  packed_args.pack(key_transpose);
  packed_args.pack(query_layer);
  packed_args.pack(query_transpose);
  packed_args.pack(scale);
  packed_args.pack(value_layer);
  packed_args.pack(value_transpose);
  packed_args.pack(result1);
  packed_args.pack(result2);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(dx_transpose);
  saved.after(keep_prob);
  saved.after(key_layer_);
  saved.after(key_transpose);
  saved.after(query_layer_);
  saved.after(query_transpose);
  saved.after(scale);
  saved.after(value_layer_);
  saved.after(value_transpose);
  saved.after(result1_);
  saved.after(result2_);
  return output_result;
#endif
}

static variable_list NpuFusionAttentionBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,4> needs_input_grad, c10::OptionalArray<int64_t>& actual_seq_kvlen, c10::OptionalArray<int64_t>& actual_seq_qlen, Tensor& atten_mask, bool& gen_mask_parallel, int64_t& head_num, int64_t& inner_precise, std::string& input_layout, double& keep_prob, Tensor& key, int64_t& next_tockens, Tensor& padding_mask, int64_t& pre_tockens, c10::OptionalArray<int64_t>& prefix, Tensor& pse, Tensor& query, double& scale, std::string& softmax_layout, int64_t& sparse_mode, bool& sync, Tensor& value, Tensor& result0, Tensor& result1, Tensor& result2, Tensor& result3, int64_t& result4, int64_t& result5, int64_t& result6)
{
  IndexRangeGenerator gen;
  auto query_ix = gen.range(1);
  auto key_ix = gen.range(1);
  auto value_ix = gen.range(1);
  auto pse_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  if (needs_input_grad[0] || needs_input_grad[1] || needs_input_grad[2] || needs_input_grad[3]) {

    auto grad_result = npu_fusion_attention_grad(query, key, value, grad, head_num, input_layout, pse, padding_mask, atten_mask, result1, result2, result3, result0, scale, keep_prob, pre_tockens, next_tockens, inner_precise, result4, result5, result6, prefix, actual_seq_qlen, actual_seq_kvlen, sparse_mode, gen_mask_parallel, sync, softmax_layout);
      if (needs_input_grad[/*query*/0]) {
        copy_range(grad_inputs, query_ix, std::get<0>(grad_result));
      }
      if (needs_input_grad[/*key*/1]) {
        copy_range(grad_inputs, key_ix, std::get<1>(grad_result));
      }
      if (needs_input_grad[/*value*/2]) {
        copy_range(grad_inputs, value_ix, std::get<2>(grad_result));
      }
      if (needs_input_grad[/*pse*/3]) {
        copy_range(grad_inputs, pse_ix, std::get<3>(grad_result));
      }
  }
  return grad_inputs;
}
inline variable_list NpuFusionAttentionBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 4>>();
  auto actual_seq_kvlen = packed_args.unpack<c10::OptionalArray<int64_t>>();
  auto actual_seq_qlen = packed_args.unpack<c10::OptionalArray<int64_t>>();
  auto atten_mask = packed_args.unpack<Tensor>();
  auto gen_mask_parallel = packed_args.unpack<bool>();
  auto head_num = packed_args.unpack<int64_t>();
  auto inner_precise = packed_args.unpack<int64_t>();
  auto input_layout = packed_args.unpack<std::string>();
  auto keep_prob = packed_args.unpack<double>();
  auto key = packed_args.unpack<Tensor>();
  auto next_tockens = packed_args.unpack<int64_t>();
  auto padding_mask = packed_args.unpack<Tensor>();
  auto pre_tockens = packed_args.unpack<int64_t>();
  auto prefix = packed_args.unpack<c10::OptionalArray<int64_t>>();
  auto pse = packed_args.unpack<Tensor>();
  auto query = packed_args.unpack<Tensor>();
  auto scale = packed_args.unpack<double>();
  auto softmax_layout = packed_args.unpack<std::string>();
  auto sparse_mode = packed_args.unpack<int64_t>();
  auto sync = packed_args.unpack<bool>();
  auto value = packed_args.unpack<Tensor>();
  auto result0 = packed_args.unpack<Tensor>();
  auto result1 = packed_args.unpack<Tensor>();
  auto result2 = packed_args.unpack<Tensor>();
  auto result3 = packed_args.unpack<Tensor>();
  auto result4 = packed_args.unpack<int64_t>();
  auto result5 = packed_args.unpack<int64_t>();
  auto result6 = packed_args.unpack<int64_t>();
  return NpuFusionAttentionBackward0_apply_functional(variable_list(grads), needs_input_grad, actual_seq_kvlen, actual_seq_qlen, atten_mask, gen_mask_parallel, head_num, inner_precise, input_layout, keep_prob, key, next_tockens, padding_mask, pre_tockens, prefix, pse, query, scale, softmax_layout, sparse_mode, sync, value, result0, result1, result2, result3, result4, result5, result6);
#endif
}

variable_list NpuFusionAttentionBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto atten_mask = atten_mask_.unpack();
  auto key = key_.unpack();
  auto padding_mask = padding_mask_.unpack();
  auto pse = pse_.unpack();
  auto query = query_.unpack();
  auto value = value_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto query_ix = gen.range(1);
  auto key_ix = gen.range(1);
  auto value_ix = gen.range(1);
  auto pse_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 4>{
    task_should_compute_output({ query_ix }),
    task_should_compute_output({ key_ix }),
    task_should_compute_output({ value_ix }),
    task_should_compute_output({ pse_ix }),
  };
  return NpuFusionAttentionBackward0_apply_functional(std::move(grads), needs_input_grad, actual_seq_kvlen, actual_seq_qlen, atten_mask, gen_mask_parallel, head_num, inner_precise, input_layout, keep_prob, key, next_tockens, padding_mask, pre_tockens, prefix, pse, query, scale, softmax_layout, sparse_mode, sync, value, result0, result1, result2, result3, result4, result5, result6);
}

void NpuFusionAttentionBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(actual_seq_kvlen);
    args.collect(actual_seq_qlen);
    args.collect(atten_mask_, false);
    args.collect(gen_mask_parallel);
    args.collect(head_num);
    args.collect(inner_precise);
    args.collect(input_layout);
    args.collect(keep_prob);
    args.collect(key_, false);
    args.collect(next_tockens);
    args.collect(padding_mask_, false);
    args.collect(pre_tockens);
    args.collect(prefix);
    args.collect(pse_, false);
    args.collect(query_, false);
    args.collect(scale);
    args.collect(softmax_layout);
    args.collect(sparse_mode);
    args.collect(sync);
    args.collect(value_, false);
    args.collect(result0_, true);
    args.collect(result1_, true);
    args.collect(result2_, true);
    args.collect(result3_, true);
    args.collect(result4);
    args.collect(result5);
    args.collect(result6);
}
variable_list NpuFusionAttentionBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(actual_seq_kvlen);
  saved.before(actual_seq_qlen);
  saved.before(atten_mask_);
  saved.before(gen_mask_parallel);
  saved.before(head_num);
  saved.before(inner_precise);
  saved.before(input_layout);
  saved.before(keep_prob);
  saved.before(key_);
  saved.before(next_tockens);
  saved.before(padding_mask_);
  saved.before(pre_tockens);
  saved.before(prefix);
  saved.before(pse_);
  saved.before(query_);
  saved.before(scale);
  saved.before(softmax_layout);
  saved.before(sparse_mode);
  saved.before(sync);
  saved.before(value_);
  saved.before(result0_);
  saved.before(result1_);
  saved.before(result2_);
  saved.before(result3_);
  saved.before(result4);
  saved.before(result5);
  saved.before(result6);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 4>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<c10::OptionalArray<int64_t>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<c10::OptionalArray<int64_t>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<bool>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<std::string>::packed_type(),
      torch::dynamo::autograd::IValuePacker<double>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<c10::OptionalArray<int64_t>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<double>::packed_type(),
      torch::dynamo::autograd::IValuePacker<std::string>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<bool>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuFusionAttentionBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto atten_mask = atten_mask_.unpack();
  auto key = key_.unpack();
  auto padding_mask = padding_mask_.unpack();
  auto pse = pse_.unpack();
  auto query = query_.unpack();
  auto value = value_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto query_ix = gen.range(1);
  auto key_ix = gen.range(1);
  auto value_ix = gen.range(1);
  auto pse_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 4>{
    task_should_compute_output({ query_ix }),
    task_should_compute_output({ key_ix }),
    task_should_compute_output({ value_ix }),
    task_should_compute_output({ pse_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(actual_seq_kvlen);
  packed_args.pack(actual_seq_qlen);
  packed_args.pack(atten_mask);
  packed_args.pack(gen_mask_parallel);
  packed_args.pack(head_num);
  packed_args.pack(inner_precise);
  packed_args.pack(input_layout);
  packed_args.pack(keep_prob);
  packed_args.pack(key);
  packed_args.pack(next_tockens);
  packed_args.pack(padding_mask);
  packed_args.pack(pre_tockens);
  packed_args.pack(prefix);
  packed_args.pack(pse);
  packed_args.pack(query);
  packed_args.pack(scale);
  packed_args.pack(softmax_layout);
  packed_args.pack(sparse_mode);
  packed_args.pack(sync);
  packed_args.pack(value);
  packed_args.pack(result0);
  packed_args.pack(result1);
  packed_args.pack(result2);
  packed_args.pack(result3);
  packed_args.pack(result4);
  packed_args.pack(result5);
  packed_args.pack(result6);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(actual_seq_kvlen);
  saved.after(actual_seq_qlen);
  saved.after(atten_mask_);
  saved.after(gen_mask_parallel);
  saved.after(head_num);
  saved.after(inner_precise);
  saved.after(input_layout);
  saved.after(keep_prob);
  saved.after(key_);
  saved.after(next_tockens);
  saved.after(padding_mask_);
  saved.after(pre_tockens);
  saved.after(prefix);
  saved.after(pse_);
  saved.after(query_);
  saved.after(scale);
  saved.after(softmax_layout);
  saved.after(sparse_mode);
  saved.after(sync);
  saved.after(value_);
  saved.after(result0_);
  saved.after(result1_);
  saved.after(result2_);
  saved.after(result3_);
  saved.after(result4);
  saved.after(result5);
  saved.after(result6);
  return output_result;
#endif
}

static variable_list NpuFusionAttentionV2Backward0_apply_functional(
  variable_list&& grads,
  std::array<bool,6> needs_input_grad, c10::OptionalArray<int64_t>& actual_seq_kvlen, c10::OptionalArray<int64_t>& actual_seq_qlen, Tensor& atten_mask, bool& gen_mask_parallel, int64_t& head_num, int64_t& inner_precise, std::string& input_layout, double& keep_prob, Tensor& key, Tensor& key_rope, c10::OptionalArray<int64_t>& kv_start_idx, int64_t& next_tokens, Tensor& padding_mask, int64_t& pre_tokens, c10::OptionalArray<int64_t>& prefix, Tensor& pse, int64_t& pse_type, c10::OptionalArray<int64_t>& q_start_idx, Tensor& query, Tensor& query_rope, double& scale, int64_t& sparse_mode, bool& sync, Tensor& value, Tensor& result0, Tensor& result1, Tensor& result2, Tensor& result3, int64_t& result4, int64_t& result5, int64_t& result6)
{
  IndexRangeGenerator gen;
  auto query_ix = gen.range(1);
  auto key_ix = gen.range(1);
  auto value_ix = gen.range(1);
  auto pse_ix = gen.range(1);
  auto query_rope_ix = gen.range(1);
  auto key_rope_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  if (needs_input_grad[0] || needs_input_grad[1] || needs_input_grad[2] || needs_input_grad[3] || needs_input_grad[4] || needs_input_grad[5]) {

    auto grad_result = npu_fusion_attention_grad_v2(query, key, value, grad, head_num, input_layout, pse, padding_mask, atten_mask, result1, result2, result3, result0, query_rope, key_rope, scale, keep_prob, pre_tokens, next_tokens, inner_precise, result4, result5, result6, prefix, actual_seq_qlen, actual_seq_kvlen, sparse_mode, gen_mask_parallel, sync, pse_type, q_start_idx, kv_start_idx);
      if (needs_input_grad[/*query*/0]) {
        copy_range(grad_inputs, query_ix, std::get<0>(grad_result));
      }
      if (needs_input_grad[/*key*/1]) {
        copy_range(grad_inputs, key_ix, std::get<1>(grad_result));
      }
      if (needs_input_grad[/*value*/2]) {
        copy_range(grad_inputs, value_ix, std::get<2>(grad_result));
      }
      if (needs_input_grad[/*pse*/3]) {
        copy_range(grad_inputs, pse_ix, std::get<3>(grad_result));
      }
      if (needs_input_grad[/*query_rope*/4]) {
        copy_range(grad_inputs, query_rope_ix, std::get<4>(grad_result));
      }
      if (needs_input_grad[/*key_rope*/5]) {
        copy_range(grad_inputs, key_rope_ix, std::get<5>(grad_result));
      }
  }
  return grad_inputs;
}
inline variable_list NpuFusionAttentionV2Backward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 6>>();
  auto actual_seq_kvlen = packed_args.unpack<c10::OptionalArray<int64_t>>();
  auto actual_seq_qlen = packed_args.unpack<c10::OptionalArray<int64_t>>();
  auto atten_mask = packed_args.unpack<Tensor>();
  auto gen_mask_parallel = packed_args.unpack<bool>();
  auto head_num = packed_args.unpack<int64_t>();
  auto inner_precise = packed_args.unpack<int64_t>();
  auto input_layout = packed_args.unpack<std::string>();
  auto keep_prob = packed_args.unpack<double>();
  auto key = packed_args.unpack<Tensor>();
  auto key_rope = packed_args.unpack<Tensor>();
  auto kv_start_idx = packed_args.unpack<c10::OptionalArray<int64_t>>();
  auto next_tokens = packed_args.unpack<int64_t>();
  auto padding_mask = packed_args.unpack<Tensor>();
  auto pre_tokens = packed_args.unpack<int64_t>();
  auto prefix = packed_args.unpack<c10::OptionalArray<int64_t>>();
  auto pse = packed_args.unpack<Tensor>();
  auto pse_type = packed_args.unpack<int64_t>();
  auto q_start_idx = packed_args.unpack<c10::OptionalArray<int64_t>>();
  auto query = packed_args.unpack<Tensor>();
  auto query_rope = packed_args.unpack<Tensor>();
  auto scale = packed_args.unpack<double>();
  auto sparse_mode = packed_args.unpack<int64_t>();
  auto sync = packed_args.unpack<bool>();
  auto value = packed_args.unpack<Tensor>();
  auto result0 = packed_args.unpack<Tensor>();
  auto result1 = packed_args.unpack<Tensor>();
  auto result2 = packed_args.unpack<Tensor>();
  auto result3 = packed_args.unpack<Tensor>();
  auto result4 = packed_args.unpack<int64_t>();
  auto result5 = packed_args.unpack<int64_t>();
  auto result6 = packed_args.unpack<int64_t>();
  return NpuFusionAttentionV2Backward0_apply_functional(variable_list(grads), needs_input_grad, actual_seq_kvlen, actual_seq_qlen, atten_mask, gen_mask_parallel, head_num, inner_precise, input_layout, keep_prob, key, key_rope, kv_start_idx, next_tokens, padding_mask, pre_tokens, prefix, pse, pse_type, q_start_idx, query, query_rope, scale, sparse_mode, sync, value, result0, result1, result2, result3, result4, result5, result6);
#endif
}

variable_list NpuFusionAttentionV2Backward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto atten_mask = atten_mask_.unpack();
  auto key = key_.unpack();
  auto key_rope = key_rope_.unpack();
  auto padding_mask = padding_mask_.unpack();
  auto pse = pse_.unpack();
  auto query = query_.unpack();
  auto query_rope = query_rope_.unpack();
  auto value = value_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto query_ix = gen.range(1);
  auto key_ix = gen.range(1);
  auto value_ix = gen.range(1);
  auto pse_ix = gen.range(1);
  auto query_rope_ix = gen.range(1);
  auto key_rope_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 6>{
    task_should_compute_output({ query_ix }),
    task_should_compute_output({ key_ix }),
    task_should_compute_output({ value_ix }),
    task_should_compute_output({ pse_ix }),
    task_should_compute_output({ query_rope_ix }),
    task_should_compute_output({ key_rope_ix }),
  };
  return NpuFusionAttentionV2Backward0_apply_functional(std::move(grads), needs_input_grad, actual_seq_kvlen, actual_seq_qlen, atten_mask, gen_mask_parallel, head_num, inner_precise, input_layout, keep_prob, key, key_rope, kv_start_idx, next_tokens, padding_mask, pre_tokens, prefix, pse, pse_type, q_start_idx, query, query_rope, scale, sparse_mode, sync, value, result0, result1, result2, result3, result4, result5, result6);
}

void NpuFusionAttentionV2Backward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(actual_seq_kvlen);
    args.collect(actual_seq_qlen);
    args.collect(atten_mask_, false);
    args.collect(gen_mask_parallel);
    args.collect(head_num);
    args.collect(inner_precise);
    args.collect(input_layout);
    args.collect(keep_prob);
    args.collect(key_, false);
    args.collect(key_rope_, false);
    args.collect(kv_start_idx);
    args.collect(next_tokens);
    args.collect(padding_mask_, false);
    args.collect(pre_tokens);
    args.collect(prefix);
    args.collect(pse_, false);
    args.collect(pse_type);
    args.collect(q_start_idx);
    args.collect(query_, false);
    args.collect(query_rope_, false);
    args.collect(scale);
    args.collect(sparse_mode);
    args.collect(sync);
    args.collect(value_, false);
    args.collect(result0_, true);
    args.collect(result1_, true);
    args.collect(result2_, true);
    args.collect(result3_, true);
    args.collect(result4);
    args.collect(result5);
    args.collect(result6);
}
variable_list NpuFusionAttentionV2Backward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(actual_seq_kvlen);
  saved.before(actual_seq_qlen);
  saved.before(atten_mask_);
  saved.before(gen_mask_parallel);
  saved.before(head_num);
  saved.before(inner_precise);
  saved.before(input_layout);
  saved.before(keep_prob);
  saved.before(key_);
  saved.before(key_rope_);
  saved.before(kv_start_idx);
  saved.before(next_tokens);
  saved.before(padding_mask_);
  saved.before(pre_tokens);
  saved.before(prefix);
  saved.before(pse_);
  saved.before(pse_type);
  saved.before(q_start_idx);
  saved.before(query_);
  saved.before(query_rope_);
  saved.before(scale);
  saved.before(sparse_mode);
  saved.before(sync);
  saved.before(value_);
  saved.before(result0_);
  saved.before(result1_);
  saved.before(result2_);
  saved.before(result3_);
  saved.before(result4);
  saved.before(result5);
  saved.before(result6);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 6>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<c10::OptionalArray<int64_t>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<c10::OptionalArray<int64_t>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<bool>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<std::string>::packed_type(),
      torch::dynamo::autograd::IValuePacker<double>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<c10::OptionalArray<int64_t>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<c10::OptionalArray<int64_t>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<c10::OptionalArray<int64_t>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<double>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<bool>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuFusionAttentionV2Backward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto atten_mask = atten_mask_.unpack();
  auto key = key_.unpack();
  auto key_rope = key_rope_.unpack();
  auto padding_mask = padding_mask_.unpack();
  auto pse = pse_.unpack();
  auto query = query_.unpack();
  auto query_rope = query_rope_.unpack();
  auto value = value_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto query_ix = gen.range(1);
  auto key_ix = gen.range(1);
  auto value_ix = gen.range(1);
  auto pse_ix = gen.range(1);
  auto query_rope_ix = gen.range(1);
  auto key_rope_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 6>{
    task_should_compute_output({ query_ix }),
    task_should_compute_output({ key_ix }),
    task_should_compute_output({ value_ix }),
    task_should_compute_output({ pse_ix }),
    task_should_compute_output({ query_rope_ix }),
    task_should_compute_output({ key_rope_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(actual_seq_kvlen);
  packed_args.pack(actual_seq_qlen);
  packed_args.pack(atten_mask);
  packed_args.pack(gen_mask_parallel);
  packed_args.pack(head_num);
  packed_args.pack(inner_precise);
  packed_args.pack(input_layout);
  packed_args.pack(keep_prob);
  packed_args.pack(key);
  packed_args.pack(key_rope);
  packed_args.pack(kv_start_idx);
  packed_args.pack(next_tokens);
  packed_args.pack(padding_mask);
  packed_args.pack(pre_tokens);
  packed_args.pack(prefix);
  packed_args.pack(pse);
  packed_args.pack(pse_type);
  packed_args.pack(q_start_idx);
  packed_args.pack(query);
  packed_args.pack(query_rope);
  packed_args.pack(scale);
  packed_args.pack(sparse_mode);
  packed_args.pack(sync);
  packed_args.pack(value);
  packed_args.pack(result0);
  packed_args.pack(result1);
  packed_args.pack(result2);
  packed_args.pack(result3);
  packed_args.pack(result4);
  packed_args.pack(result5);
  packed_args.pack(result6);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(actual_seq_kvlen);
  saved.after(actual_seq_qlen);
  saved.after(atten_mask_);
  saved.after(gen_mask_parallel);
  saved.after(head_num);
  saved.after(inner_precise);
  saved.after(input_layout);
  saved.after(keep_prob);
  saved.after(key_);
  saved.after(key_rope_);
  saved.after(kv_start_idx);
  saved.after(next_tokens);
  saved.after(padding_mask_);
  saved.after(pre_tokens);
  saved.after(prefix);
  saved.after(pse_);
  saved.after(pse_type);
  saved.after(q_start_idx);
  saved.after(query_);
  saved.after(query_rope_);
  saved.after(scale);
  saved.after(sparse_mode);
  saved.after(sync);
  saved.after(value_);
  saved.after(result0_);
  saved.after(result1_);
  saved.after(result2_);
  saved.after(result3_);
  saved.after(result4);
  saved.after(result5);
  saved.after(result6);
  return output_result;
#endif
}

static variable_list NpuGegluBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,1> needs_input_grad, bool& activate_left, int64_t& approximate, int64_t& dim, Tensor& self, Tensor& result1)
{
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (needs_input_grad[/*self*/0]) {
    auto grad_result = any_grad_defined ? (npu_geglu_grad(grad, self, result1, dim, approximate, activate_left)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
inline variable_list NpuGegluBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 1>>();
  auto activate_left = packed_args.unpack<bool>();
  auto approximate = packed_args.unpack<int64_t>();
  auto dim = packed_args.unpack<int64_t>();
  auto self = packed_args.unpack<Tensor>();
  auto result1 = packed_args.unpack<Tensor>();
  return NpuGegluBackward0_apply_functional(variable_list(grads), needs_input_grad, activate_left, approximate, dim, self, result1);
#endif
}

variable_list NpuGegluBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto self = self_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  return NpuGegluBackward0_apply_functional(std::move(grads), needs_input_grad, activate_left, approximate, dim, self, result1);
}

void NpuGegluBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(activate_left);
    args.collect(approximate);
    args.collect(dim);
    args.collect(self_, false);
    args.collect(result1_, true);
}
variable_list NpuGegluBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(activate_left);
  saved.before(approximate);
  saved.before(dim);
  saved.before(self_);
  saved.before(result1_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 1>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<bool>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuGegluBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto self = self_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(activate_left);
  packed_args.pack(approximate);
  packed_args.pack(dim);
  packed_args.pack(self);
  packed_args.pack(result1);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(activate_left);
  saved.after(approximate);
  saved.after(dim);
  saved.after(self_);
  saved.after(result1_);
  return output_result;
#endif
}

static variable_list NpuGiouBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,2> needs_input_grad, Tensor& gtboxes, bool& is_cross, int64_t& mode, Tensor& self, bool& trans)
{
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto gtboxes_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  if (needs_input_grad[0] || needs_input_grad[1]) {

    auto grad_result = npu_giou_backward(grad, self, gtboxes, trans, is_cross, mode);
      if (needs_input_grad[/*self*/0]) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (needs_input_grad[/*gtboxes*/1]) {
        copy_range(grad_inputs, gtboxes_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
inline variable_list NpuGiouBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 2>>();
  auto gtboxes = packed_args.unpack<Tensor>();
  auto is_cross = packed_args.unpack<bool>();
  auto mode = packed_args.unpack<int64_t>();
  auto self = packed_args.unpack<Tensor>();
  auto trans = packed_args.unpack<bool>();
  return NpuGiouBackward0_apply_functional(variable_list(grads), needs_input_grad, gtboxes, is_cross, mode, self, trans);
#endif
}

variable_list NpuGiouBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto gtboxes = gtboxes_.unpack();
  auto self = self_.unpack();
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto gtboxes_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 2>{
    task_should_compute_output({ self_ix }),
    task_should_compute_output({ gtboxes_ix }),
  };
  return NpuGiouBackward0_apply_functional(std::move(grads), needs_input_grad, gtboxes, is_cross, mode, self, trans);
}

void NpuGiouBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(gtboxes_, false);
    args.collect(is_cross);
    args.collect(mode);
    args.collect(self_, false);
    args.collect(trans);
}
variable_list NpuGiouBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(gtboxes_);
  saved.before(is_cross);
  saved.before(mode);
  saved.before(self_);
  saved.before(trans);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 2>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<bool>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<bool>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuGiouBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto gtboxes = gtboxes_.unpack();
  auto self = self_.unpack();
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto gtboxes_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 2>{
    task_should_compute_output({ self_ix }),
    task_should_compute_output({ gtboxes_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(gtboxes);
  packed_args.pack(is_cross);
  packed_args.pack(mode);
  packed_args.pack(self);
  packed_args.pack(trans);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(gtboxes_);
  saved.after(is_cross);
  saved.after(mode);
  saved.after(self_);
  saved.after(trans);
  return output_result;
#endif
}

static variable_list NpuGruBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,6> needs_input_grad, Tensor& bias_hidden, Tensor& bias_input, Tensor& hx, Tensor& input, Tensor& seq_length, Tensor& weight_hidden, Tensor& weight_input, Tensor& result0, Tensor& result1, Tensor& result2, Tensor& result3, Tensor& result4, Tensor& result5)
{
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto hx_ix = gen.range(1);
  auto weight_input_ix = gen.range(1);
  auto weight_hidden_ix = gen.range(1);
  auto bias_input_ix = gen.range(1);
  auto bias_hidden_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (needs_input_grad[2] || needs_input_grad[3] || needs_input_grad[0] || needs_input_grad[4] || needs_input_grad[5] || needs_input_grad[1]) {

    auto grad_result = npu_gru_backward(grads[0], grads[1], input, weight_input, weight_hidden, bias_input, bias_hidden, seq_length, hx, result0, result1, result2, result3, result4, result5);
      if (needs_input_grad[/*weight_input*/2]) {
        copy_range(grad_inputs, weight_input_ix, std::get<0>(grad_result));
      }
      if (needs_input_grad[/*weight_hidden*/3]) {
        copy_range(grad_inputs, weight_hidden_ix, std::get<1>(grad_result));
      }
      if (needs_input_grad[/*input*/0]) {
        copy_range(grad_inputs, input_ix, std::get<2>(grad_result));
      }
      if (needs_input_grad[/*bias_input*/4]) {
        copy_range(grad_inputs, bias_input_ix, std::get<3>(grad_result));
      }
      if (needs_input_grad[/*bias_hidden*/5]) {
        copy_range(grad_inputs, bias_hidden_ix, std::get<4>(grad_result));
      }
      if (needs_input_grad[/*hx*/1]) {
        copy_range(grad_inputs, hx_ix, std::get<5>(grad_result));
      }
  }
  return grad_inputs;
}
inline variable_list NpuGruBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 6>>();
  auto bias_hidden = packed_args.unpack<Tensor>();
  auto bias_input = packed_args.unpack<Tensor>();
  auto hx = packed_args.unpack<Tensor>();
  auto input = packed_args.unpack<Tensor>();
  auto seq_length = packed_args.unpack<Tensor>();
  auto weight_hidden = packed_args.unpack<Tensor>();
  auto weight_input = packed_args.unpack<Tensor>();
  auto result0 = packed_args.unpack<Tensor>();
  auto result1 = packed_args.unpack<Tensor>();
  auto result2 = packed_args.unpack<Tensor>();
  auto result3 = packed_args.unpack<Tensor>();
  auto result4 = packed_args.unpack<Tensor>();
  auto result5 = packed_args.unpack<Tensor>();
  return NpuGruBackward0_apply_functional(variable_list(grads), needs_input_grad, bias_hidden, bias_input, hx, input, seq_length, weight_hidden, weight_input, result0, result1, result2, result3, result4, result5);
#endif
}

variable_list NpuGruBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto bias_hidden = bias_hidden_.unpack();
  auto bias_input = bias_input_.unpack();
  auto hx = hx_.unpack();
  auto input = input_.unpack();
  auto seq_length = seq_length_.unpack();
  auto weight_hidden = weight_hidden_.unpack();
  auto weight_input = weight_input_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  auto result4 = result4_.unpack(shared_from_this());
  auto result5 = result5_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto hx_ix = gen.range(1);
  auto weight_input_ix = gen.range(1);
  auto weight_hidden_ix = gen.range(1);
  auto bias_input_ix = gen.range(1);
  auto bias_hidden_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 6>{
    task_should_compute_output({ input_ix }),
    task_should_compute_output({ hx_ix }),
    task_should_compute_output({ weight_input_ix }),
    task_should_compute_output({ weight_hidden_ix }),
    task_should_compute_output({ bias_input_ix }),
    task_should_compute_output({ bias_hidden_ix }),
  };
  return NpuGruBackward0_apply_functional(std::move(grads), needs_input_grad, bias_hidden, bias_input, hx, input, seq_length, weight_hidden, weight_input, result0, result1, result2, result3, result4, result5);
}

void NpuGruBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(bias_hidden_, false);
    args.collect(bias_input_, false);
    args.collect(hx_, false);
    args.collect(input_, false);
    args.collect(seq_length_, false);
    args.collect(weight_hidden_, false);
    args.collect(weight_input_, false);
    args.collect(result0_, true);
    args.collect(result1_, true);
    args.collect(result2_, true);
    args.collect(result3_, true);
    args.collect(result4_, true);
    args.collect(result5_, true);
}
variable_list NpuGruBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(bias_hidden_);
  saved.before(bias_input_);
  saved.before(hx_);
  saved.before(input_);
  saved.before(seq_length_);
  saved.before(weight_hidden_);
  saved.before(weight_input_);
  saved.before(result0_);
  saved.before(result1_);
  saved.before(result2_);
  saved.before(result3_);
  saved.before(result4_);
  saved.before(result5_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 6>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuGruBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto bias_hidden = bias_hidden_.unpack();
  auto bias_input = bias_input_.unpack();
  auto hx = hx_.unpack();
  auto input = input_.unpack();
  auto seq_length = seq_length_.unpack();
  auto weight_hidden = weight_hidden_.unpack();
  auto weight_input = weight_input_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  auto result4 = result4_.unpack(shared_from_this());
  auto result5 = result5_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto hx_ix = gen.range(1);
  auto weight_input_ix = gen.range(1);
  auto weight_hidden_ix = gen.range(1);
  auto bias_input_ix = gen.range(1);
  auto bias_hidden_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 6>{
    task_should_compute_output({ input_ix }),
    task_should_compute_output({ hx_ix }),
    task_should_compute_output({ weight_input_ix }),
    task_should_compute_output({ weight_hidden_ix }),
    task_should_compute_output({ bias_input_ix }),
    task_should_compute_output({ bias_hidden_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(bias_hidden);
  packed_args.pack(bias_input);
  packed_args.pack(hx);
  packed_args.pack(input);
  packed_args.pack(seq_length);
  packed_args.pack(weight_hidden);
  packed_args.pack(weight_input);
  packed_args.pack(result0);
  packed_args.pack(result1);
  packed_args.pack(result2);
  packed_args.pack(result3);
  packed_args.pack(result4);
  packed_args.pack(result5);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(bias_hidden_);
  saved.after(bias_input_);
  saved.after(hx_);
  saved.after(input_);
  saved.after(seq_length_);
  saved.after(weight_hidden_);
  saved.after(weight_input_);
  saved.after(result0_);
  saved.after(result1_);
  saved.after(result2_);
  saved.after(result3_);
  saved.after(result4_);
  saved.after(result5_);
  return output_result;
#endif
}

static variable_list NpuLinearBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,3> needs_input_grad, Tensor& input, Tensor& weight)
{
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (needs_input_grad[/*bias*/2]) {
    auto grad_result = any_grad_defined ? (maybe_multiply(grad, 1)) : Tensor();
    copy_range(grad_inputs, bias_ix, grad_result);
  }
  if (needs_input_grad[0] || needs_input_grad[1]) {

    auto grad_result = npu_linear_backward(grad, input, weight);
      if (needs_input_grad[/*input*/0]) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (needs_input_grad[/*weight*/1]) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
inline variable_list NpuLinearBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 3>>();
  auto input = packed_args.unpack<Tensor>();
  auto weight = packed_args.unpack<Tensor>();
  return NpuLinearBackward0_apply_functional(variable_list(grads), needs_input_grad, input, weight);
#endif
}

variable_list NpuLinearBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto input = input_.unpack();
  auto weight = weight_.unpack();
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 3>{
    task_should_compute_output({ input_ix }),
    task_should_compute_output({ weight_ix }),
    task_should_compute_output({ bias_ix }),
  };
  return NpuLinearBackward0_apply_functional(std::move(grads), needs_input_grad, input, weight);
}

void NpuLinearBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(input_, false);
    args.collect(weight_, false);
}
variable_list NpuLinearBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(input_);
  saved.before(weight_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 3>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuLinearBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto input = input_.unpack();
  auto weight = weight_.unpack();
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 3>{
    task_should_compute_output({ input_ix }),
    task_should_compute_output({ weight_ix }),
    task_should_compute_output({ bias_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(input);
  packed_args.pack(weight);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(input_);
  saved.after(weight_);
  return output_result;
#endif
}

static variable_list NpuLstmBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,5> needs_input_grad, Tensor& bias, Tensor& c, Tensor& h, Tensor& input, Tensor& weight, Tensor& result0, Tensor& result1, Tensor& result2, Tensor& result3, Tensor& result4, Tensor& result5, Tensor& result6, Tensor& result7)
{
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  auto h_ix = gen.range(1);
  auto c_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (needs_input_grad[0] || needs_input_grad[1] || needs_input_grad[2] || needs_input_grad[3] || needs_input_grad[4]) {

    auto grad_result = npu_lstm_backward(grads[0], grads[1], grads[2], input, weight, bias, h, c, result0, result1, result2, result3, result4, result5, result6, result7);
      if (needs_input_grad[/*input*/0]) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (needs_input_grad[/*weight*/1]) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (needs_input_grad[/*bias*/2]) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
      if (needs_input_grad[/*h*/3]) {
        copy_range(grad_inputs, h_ix, std::get<3>(grad_result));
      }
      if (needs_input_grad[/*c*/4]) {
        copy_range(grad_inputs, c_ix, std::get<4>(grad_result));
      }
  }
  return grad_inputs;
}
inline variable_list NpuLstmBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 5>>();
  auto bias = packed_args.unpack<Tensor>();
  auto c = packed_args.unpack<Tensor>();
  auto h = packed_args.unpack<Tensor>();
  auto input = packed_args.unpack<Tensor>();
  auto weight = packed_args.unpack<Tensor>();
  auto result0 = packed_args.unpack<Tensor>();
  auto result1 = packed_args.unpack<Tensor>();
  auto result2 = packed_args.unpack<Tensor>();
  auto result3 = packed_args.unpack<Tensor>();
  auto result4 = packed_args.unpack<Tensor>();
  auto result5 = packed_args.unpack<Tensor>();
  auto result6 = packed_args.unpack<Tensor>();
  auto result7 = packed_args.unpack<Tensor>();
  return NpuLstmBackward0_apply_functional(variable_list(grads), needs_input_grad, bias, c, h, input, weight, result0, result1, result2, result3, result4, result5, result6, result7);
#endif
}

variable_list NpuLstmBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto bias = bias_.unpack();
  auto c = c_.unpack();
  auto h = h_.unpack();
  auto input = input_.unpack();
  auto weight = weight_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  auto result4 = result4_.unpack(shared_from_this());
  auto result5 = result5_.unpack(shared_from_this());
  auto result6 = result6_.unpack(shared_from_this());
  auto result7 = result7_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  auto h_ix = gen.range(1);
  auto c_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 5>{
    task_should_compute_output({ input_ix }),
    task_should_compute_output({ weight_ix }),
    task_should_compute_output({ bias_ix }),
    task_should_compute_output({ h_ix }),
    task_should_compute_output({ c_ix }),
  };
  return NpuLstmBackward0_apply_functional(std::move(grads), needs_input_grad, bias, c, h, input, weight, result0, result1, result2, result3, result4, result5, result6, result7);
}

void NpuLstmBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(bias_, false);
    args.collect(c_, false);
    args.collect(h_, false);
    args.collect(input_, false);
    args.collect(weight_, false);
    args.collect(result0_, true);
    args.collect(result1_, true);
    args.collect(result2_, true);
    args.collect(result3_, true);
    args.collect(result4_, true);
    args.collect(result5_, true);
    args.collect(result6_, true);
    args.collect(result7_, true);
}
variable_list NpuLstmBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(bias_);
  saved.before(c_);
  saved.before(h_);
  saved.before(input_);
  saved.before(weight_);
  saved.before(result0_);
  saved.before(result1_);
  saved.before(result2_);
  saved.before(result3_);
  saved.before(result4_);
  saved.before(result5_);
  saved.before(result6_);
  saved.before(result7_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 5>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuLstmBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto bias = bias_.unpack();
  auto c = c_.unpack();
  auto h = h_.unpack();
  auto input = input_.unpack();
  auto weight = weight_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  auto result4 = result4_.unpack(shared_from_this());
  auto result5 = result5_.unpack(shared_from_this());
  auto result6 = result6_.unpack(shared_from_this());
  auto result7 = result7_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  auto h_ix = gen.range(1);
  auto c_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 5>{
    task_should_compute_output({ input_ix }),
    task_should_compute_output({ weight_ix }),
    task_should_compute_output({ bias_ix }),
    task_should_compute_output({ h_ix }),
    task_should_compute_output({ c_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(bias);
  packed_args.pack(c);
  packed_args.pack(h);
  packed_args.pack(input);
  packed_args.pack(weight);
  packed_args.pack(result0);
  packed_args.pack(result1);
  packed_args.pack(result2);
  packed_args.pack(result3);
  packed_args.pack(result4);
  packed_args.pack(result5);
  packed_args.pack(result6);
  packed_args.pack(result7);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(bias_);
  saved.after(c_);
  saved.after(h_);
  saved.after(input_);
  saved.after(weight_);
  saved.after(result0_);
  saved.after(result1_);
  saved.after(result2_);
  saved.after(result3_);
  saved.after(result4_);
  saved.after(result5_);
  saved.after(result6_);
  saved.after(result7_);
  return output_result;
#endif
}

static variable_list NpuLstmCellBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,7> needs_input_grad, Tensor& c, Tensor& h, Tensor& input, Tensor& w_hh, Tensor& w_ih, Tensor& result0, Tensor& result1, Tensor& result2, Tensor& result3, Tensor& result4, Tensor& result5, Tensor& result6, Tensor& result7)
{
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto w_ih_ix = gen.range(1);
  auto w_hh_ix = gen.range(1);
  auto h_ix = gen.range(1);
  auto c_ix = gen.range(1);
  auto b_ih_ix = gen.range(1);
  auto b_hh_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (needs_input_grad[0] || needs_input_grad[1] || needs_input_grad[2] || needs_input_grad[5] || needs_input_grad[6] || needs_input_grad[3] || needs_input_grad[4]) {

    auto grad_result = npu_lstm_cell_backward(grads[0], grads[1], grads[2], input, w_ih, w_hh, h, c, result0, result1, result2, result3, result4, result5, result6, result7);
      if (needs_input_grad[/*input*/0]) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (needs_input_grad[/*w_ih*/1]) {
        copy_range(grad_inputs, w_ih_ix, std::get<1>(grad_result));
      }
      if (needs_input_grad[/*w_hh*/2]) {
        copy_range(grad_inputs, w_hh_ix, std::get<2>(grad_result));
      }
      if (needs_input_grad[/*b_ih*/5]) {
        copy_range(grad_inputs, b_ih_ix, std::get<3>(grad_result));
      }
      if (needs_input_grad[/*b_hh*/6]) {
        copy_range(grad_inputs, b_hh_ix, std::get<4>(grad_result));
      }
      if (needs_input_grad[/*h*/3]) {
        copy_range(grad_inputs, h_ix, std::get<5>(grad_result));
      }
      if (needs_input_grad[/*c*/4]) {
        copy_range(grad_inputs, c_ix, std::get<6>(grad_result));
      }
  }
  return grad_inputs;
}
inline variable_list NpuLstmCellBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 7>>();
  auto c = packed_args.unpack<Tensor>();
  auto h = packed_args.unpack<Tensor>();
  auto input = packed_args.unpack<Tensor>();
  auto w_hh = packed_args.unpack<Tensor>();
  auto w_ih = packed_args.unpack<Tensor>();
  auto result0 = packed_args.unpack<Tensor>();
  auto result1 = packed_args.unpack<Tensor>();
  auto result2 = packed_args.unpack<Tensor>();
  auto result3 = packed_args.unpack<Tensor>();
  auto result4 = packed_args.unpack<Tensor>();
  auto result5 = packed_args.unpack<Tensor>();
  auto result6 = packed_args.unpack<Tensor>();
  auto result7 = packed_args.unpack<Tensor>();
  return NpuLstmCellBackward0_apply_functional(variable_list(grads), needs_input_grad, c, h, input, w_hh, w_ih, result0, result1, result2, result3, result4, result5, result6, result7);
#endif
}

variable_list NpuLstmCellBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto c = c_.unpack();
  auto h = h_.unpack();
  auto input = input_.unpack();
  auto w_hh = w_hh_.unpack();
  auto w_ih = w_ih_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  auto result4 = result4_.unpack(shared_from_this());
  auto result5 = result5_.unpack(shared_from_this());
  auto result6 = result6_.unpack(shared_from_this());
  auto result7 = result7_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto w_ih_ix = gen.range(1);
  auto w_hh_ix = gen.range(1);
  auto h_ix = gen.range(1);
  auto c_ix = gen.range(1);
  auto b_ih_ix = gen.range(1);
  auto b_hh_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 7>{
    task_should_compute_output({ input_ix }),
    task_should_compute_output({ w_ih_ix }),
    task_should_compute_output({ w_hh_ix }),
    task_should_compute_output({ h_ix }),
    task_should_compute_output({ c_ix }),
    task_should_compute_output({ b_ih_ix }),
    task_should_compute_output({ b_hh_ix }),
  };
  return NpuLstmCellBackward0_apply_functional(std::move(grads), needs_input_grad, c, h, input, w_hh, w_ih, result0, result1, result2, result3, result4, result5, result6, result7);
}

void NpuLstmCellBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(c_, false);
    args.collect(h_, false);
    args.collect(input_, false);
    args.collect(w_hh_, false);
    args.collect(w_ih_, false);
    args.collect(result0_, true);
    args.collect(result1_, true);
    args.collect(result2_, true);
    args.collect(result3_, true);
    args.collect(result4_, true);
    args.collect(result5_, true);
    args.collect(result6_, true);
    args.collect(result7_, true);
}
variable_list NpuLstmCellBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(c_);
  saved.before(h_);
  saved.before(input_);
  saved.before(w_hh_);
  saved.before(w_ih_);
  saved.before(result0_);
  saved.before(result1_);
  saved.before(result2_);
  saved.before(result3_);
  saved.before(result4_);
  saved.before(result5_);
  saved.before(result6_);
  saved.before(result7_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 7>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuLstmCellBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto c = c_.unpack();
  auto h = h_.unpack();
  auto input = input_.unpack();
  auto w_hh = w_hh_.unpack();
  auto w_ih = w_ih_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  auto result4 = result4_.unpack(shared_from_this());
  auto result5 = result5_.unpack(shared_from_this());
  auto result6 = result6_.unpack(shared_from_this());
  auto result7 = result7_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto w_ih_ix = gen.range(1);
  auto w_hh_ix = gen.range(1);
  auto h_ix = gen.range(1);
  auto c_ix = gen.range(1);
  auto b_ih_ix = gen.range(1);
  auto b_hh_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 7>{
    task_should_compute_output({ input_ix }),
    task_should_compute_output({ w_ih_ix }),
    task_should_compute_output({ w_hh_ix }),
    task_should_compute_output({ h_ix }),
    task_should_compute_output({ c_ix }),
    task_should_compute_output({ b_ih_ix }),
    task_should_compute_output({ b_hh_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(c);
  packed_args.pack(h);
  packed_args.pack(input);
  packed_args.pack(w_hh);
  packed_args.pack(w_ih);
  packed_args.pack(result0);
  packed_args.pack(result1);
  packed_args.pack(result2);
  packed_args.pack(result3);
  packed_args.pack(result4);
  packed_args.pack(result5);
  packed_args.pack(result6);
  packed_args.pack(result7);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(c_);
  saved.after(h_);
  saved.after(input_);
  saved.after(w_hh_);
  saved.after(w_ih_);
  saved.after(result0_);
  saved.after(result1_);
  saved.after(result2_);
  saved.after(result3_);
  saved.after(result4_);
  saved.after(result5_);
  saved.after(result6_);
  saved.after(result7_);
  return output_result;
#endif
}

static variable_list NpuLstmDataBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,5> needs_input_grad, Tensor& batch_sizes, Tensor& bias, Tensor& c, bool& direction, Tensor& h, Tensor& input, Tensor& weight, Tensor& result0, Tensor& result1, Tensor& result2, Tensor& result3, Tensor& result4, Tensor& result5, Tensor& result6, Tensor& result7)
{
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  auto h_ix = gen.range(1);
  auto c_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (needs_input_grad[0] || needs_input_grad[1] || needs_input_grad[2] || needs_input_grad[3] || needs_input_grad[4]) {

    auto grad_result = npu_lstm_data_backward(grads[0], grads[1], grads[2], input, batch_sizes, weight, bias, h, c, result0, result1, result2, result3, result4, result5, result6, result7, direction);
      if (needs_input_grad[/*input*/0]) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (needs_input_grad[/*weight*/1]) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (needs_input_grad[/*bias*/2]) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
      if (needs_input_grad[/*h*/3]) {
        copy_range(grad_inputs, h_ix, std::get<3>(grad_result));
      }
      if (needs_input_grad[/*c*/4]) {
        copy_range(grad_inputs, c_ix, std::get<4>(grad_result));
      }
  }
  return grad_inputs;
}
inline variable_list NpuLstmDataBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 5>>();
  auto batch_sizes = packed_args.unpack<Tensor>();
  auto bias = packed_args.unpack<Tensor>();
  auto c = packed_args.unpack<Tensor>();
  auto direction = packed_args.unpack<bool>();
  auto h = packed_args.unpack<Tensor>();
  auto input = packed_args.unpack<Tensor>();
  auto weight = packed_args.unpack<Tensor>();
  auto result0 = packed_args.unpack<Tensor>();
  auto result1 = packed_args.unpack<Tensor>();
  auto result2 = packed_args.unpack<Tensor>();
  auto result3 = packed_args.unpack<Tensor>();
  auto result4 = packed_args.unpack<Tensor>();
  auto result5 = packed_args.unpack<Tensor>();
  auto result6 = packed_args.unpack<Tensor>();
  auto result7 = packed_args.unpack<Tensor>();
  return NpuLstmDataBackward0_apply_functional(variable_list(grads), needs_input_grad, batch_sizes, bias, c, direction, h, input, weight, result0, result1, result2, result3, result4, result5, result6, result7);
#endif
}

variable_list NpuLstmDataBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto batch_sizes = batch_sizes_.unpack();
  auto bias = bias_.unpack();
  auto c = c_.unpack();
  auto h = h_.unpack();
  auto input = input_.unpack();
  auto weight = weight_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  auto result4 = result4_.unpack(shared_from_this());
  auto result5 = result5_.unpack(shared_from_this());
  auto result6 = result6_.unpack(shared_from_this());
  auto result7 = result7_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  auto h_ix = gen.range(1);
  auto c_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 5>{
    task_should_compute_output({ input_ix }),
    task_should_compute_output({ weight_ix }),
    task_should_compute_output({ bias_ix }),
    task_should_compute_output({ h_ix }),
    task_should_compute_output({ c_ix }),
  };
  return NpuLstmDataBackward0_apply_functional(std::move(grads), needs_input_grad, batch_sizes, bias, c, direction, h, input, weight, result0, result1, result2, result3, result4, result5, result6, result7);
}

void NpuLstmDataBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(batch_sizes_, false);
    args.collect(bias_, false);
    args.collect(c_, false);
    args.collect(direction);
    args.collect(h_, false);
    args.collect(input_, false);
    args.collect(weight_, false);
    args.collect(result0_, true);
    args.collect(result1_, true);
    args.collect(result2_, true);
    args.collect(result3_, true);
    args.collect(result4_, true);
    args.collect(result5_, true);
    args.collect(result6_, true);
    args.collect(result7_, true);
}
variable_list NpuLstmDataBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(batch_sizes_);
  saved.before(bias_);
  saved.before(c_);
  saved.before(direction);
  saved.before(h_);
  saved.before(input_);
  saved.before(weight_);
  saved.before(result0_);
  saved.before(result1_);
  saved.before(result2_);
  saved.before(result3_);
  saved.before(result4_);
  saved.before(result5_);
  saved.before(result6_);
  saved.before(result7_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 5>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<bool>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuLstmDataBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto batch_sizes = batch_sizes_.unpack();
  auto bias = bias_.unpack();
  auto c = c_.unpack();
  auto h = h_.unpack();
  auto input = input_.unpack();
  auto weight = weight_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  auto result4 = result4_.unpack(shared_from_this());
  auto result5 = result5_.unpack(shared_from_this());
  auto result6 = result6_.unpack(shared_from_this());
  auto result7 = result7_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  auto h_ix = gen.range(1);
  auto c_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 5>{
    task_should_compute_output({ input_ix }),
    task_should_compute_output({ weight_ix }),
    task_should_compute_output({ bias_ix }),
    task_should_compute_output({ h_ix }),
    task_should_compute_output({ c_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(batch_sizes);
  packed_args.pack(bias);
  packed_args.pack(c);
  packed_args.pack(direction);
  packed_args.pack(h);
  packed_args.pack(input);
  packed_args.pack(weight);
  packed_args.pack(result0);
  packed_args.pack(result1);
  packed_args.pack(result2);
  packed_args.pack(result3);
  packed_args.pack(result4);
  packed_args.pack(result5);
  packed_args.pack(result6);
  packed_args.pack(result7);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(batch_sizes_);
  saved.after(bias_);
  saved.after(c_);
  saved.after(direction);
  saved.after(h_);
  saved.after(input_);
  saved.after(weight_);
  saved.after(result0_);
  saved.after(result1_);
  saved.after(result2_);
  saved.after(result3_);
  saved.after(result4_);
  saved.after(result5_);
  saved.after(result6_);
  saved.after(result7_);
  return output_result;
#endif
}

static variable_list NpuMaxBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,1> needs_input_grad, int64_t& dim, bool& keepdim, std::vector<c10::SymInt>& self_sym_sizes, Tensor& indices)
{
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (needs_input_grad[/*self*/0]) {
    auto grad_result = any_grad_defined ? (npu_max_backward(grad, dim, indices, self_sym_sizes, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
inline variable_list NpuMaxBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 1>>();
  auto dim = packed_args.unpack<int64_t>();
  auto keepdim = packed_args.unpack<bool>();
  auto self_sym_sizes = packed_args.unpack<std::vector<c10::SymInt>>();
  auto indices = packed_args.unpack<Tensor>();
  return NpuMaxBackward0_apply_functional(variable_list(grads), needs_input_grad, dim, keepdim, self_sym_sizes, indices);
#endif
}

variable_list NpuMaxBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto indices = indices_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  return NpuMaxBackward0_apply_functional(std::move(grads), needs_input_grad, dim, keepdim, self_sym_sizes, indices);
}

void NpuMaxBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(dim);
    args.collect(keepdim);
    args.collect(self_sym_sizes);
    args.collect(indices_, true);
}
variable_list NpuMaxBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(dim);
  saved.before(keepdim);
  saved.before(self_sym_sizes);
  saved.before(indices_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 1>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<bool>::packed_type(),
      torch::dynamo::autograd::IValuePacker<std::vector<c10::SymInt>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuMaxBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto indices = indices_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(dim);
  packed_args.pack(keepdim);
  packed_args.pack(self_sym_sizes);
  packed_args.pack(indices);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(dim);
  saved.after(keepdim);
  saved.after(self_sym_sizes);
  saved.after(indices_);
  return output_result;
#endif
}

static variable_list NpuMinBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,1> needs_input_grad, int64_t& dim, bool& keepdim, std::vector<c10::SymInt>& self_sym_sizes, Tensor& indices)
{
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (needs_input_grad[/*self*/0]) {
    auto grad_result = any_grad_defined ? (npu_min_backward(grad, dim, indices, self_sym_sizes, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
inline variable_list NpuMinBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 1>>();
  auto dim = packed_args.unpack<int64_t>();
  auto keepdim = packed_args.unpack<bool>();
  auto self_sym_sizes = packed_args.unpack<std::vector<c10::SymInt>>();
  auto indices = packed_args.unpack<Tensor>();
  return NpuMinBackward0_apply_functional(variable_list(grads), needs_input_grad, dim, keepdim, self_sym_sizes, indices);
#endif
}

variable_list NpuMinBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto indices = indices_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  return NpuMinBackward0_apply_functional(std::move(grads), needs_input_grad, dim, keepdim, self_sym_sizes, indices);
}

void NpuMinBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(dim);
    args.collect(keepdim);
    args.collect(self_sym_sizes);
    args.collect(indices_, true);
}
variable_list NpuMinBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(dim);
  saved.before(keepdim);
  saved.before(self_sym_sizes);
  saved.before(indices_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 1>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<bool>::packed_type(),
      torch::dynamo::autograd::IValuePacker<std::vector<c10::SymInt>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuMinBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto indices = indices_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(dim);
  packed_args.pack(keepdim);
  packed_args.pack(self_sym_sizes);
  packed_args.pack(indices);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(dim);
  saved.after(keepdim);
  saved.after(self_sym_sizes);
  saved.after(indices_);
  return output_result;
#endif
}

static variable_list NpuMishBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,1> needs_input_grad, Tensor& self)
{
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (needs_input_grad[/*self*/0]) {
    auto grad_result = any_grad_defined ? (npu_mish_backward(grad, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
inline variable_list NpuMishBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 1>>();
  auto self = packed_args.unpack<Tensor>();
  return NpuMishBackward0_apply_functional(variable_list(grads), needs_input_grad, self);
#endif
}

variable_list NpuMishBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto self = self_.unpack();
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  return NpuMishBackward0_apply_functional(std::move(grads), needs_input_grad, self);
}

void NpuMishBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(self_, false);
}
variable_list NpuMishBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(self_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 1>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuMishBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto self = self_.unpack();
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(self);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(self_);
  return output_result;
#endif
}

static variable_list NpuMultiHeadAttentionBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,11> needs_input_grad, int64_t& attn_dim_per_head, int64_t& attn_head_num, double& dropout_prob, Tensor& key, Tensor& key_bias, Tensor& key_weight, Tensor& out_proj_bias, Tensor& out_proj_weight, Tensor& query, Tensor& query_bias, Tensor& query_weight, bool& softmax_use_float, int64_t& src_len, int64_t& tgt_len, Tensor& value, Tensor& value_bias, Tensor& value_weight, Tensor& result1, Tensor& result2, Tensor& result3, Tensor& result4, Tensor& result5, Tensor& result6, Tensor& result7)
{
  IndexRangeGenerator gen;
  auto query_ix = gen.range(1);
  auto key_ix = gen.range(1);
  auto value_ix = gen.range(1);
  auto query_weight_ix = gen.range(1);
  auto key_weight_ix = gen.range(1);
  auto value_weight_ix = gen.range(1);
  auto out_proj_weight_ix = gen.range(1);
  auto query_bias_ix = gen.range(1);
  auto key_bias_ix = gen.range(1);
  auto value_bias_ix = gen.range(1);
  auto out_proj_bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  if (needs_input_grad[3] || needs_input_grad[4] || needs_input_grad[5] || needs_input_grad[6] || needs_input_grad[0] || needs_input_grad[1] || needs_input_grad[2] || needs_input_grad[7] || needs_input_grad[8] || needs_input_grad[9] || needs_input_grad[10]) {

    auto grad_result = npu_multi_head_attention_backward(query, key, value, query_weight, key_weight, value_weight, out_proj_weight, query_bias, key_bias, value_bias, out_proj_bias, result2, result3, result4, result5, result6, result7, grad, result1, attn_head_num, attn_dim_per_head, src_len, tgt_len, dropout_prob, softmax_use_float);
      if (needs_input_grad[/*query_weight*/3]) {
        copy_range(grad_inputs, query_weight_ix, std::get<0>(grad_result));
      }
      if (needs_input_grad[/*key_weight*/4]) {
        copy_range(grad_inputs, key_weight_ix, std::get<1>(grad_result));
      }
      if (needs_input_grad[/*value_weight*/5]) {
        copy_range(grad_inputs, value_weight_ix, std::get<2>(grad_result));
      }
      if (needs_input_grad[/*out_proj_weight*/6]) {
        copy_range(grad_inputs, out_proj_weight_ix, std::get<3>(grad_result));
      }
      if (needs_input_grad[/*query*/0]) {
        copy_range(grad_inputs, query_ix, std::get<4>(grad_result));
      }
      if (needs_input_grad[/*key*/1]) {
        copy_range(grad_inputs, key_ix, std::get<5>(grad_result));
      }
      if (needs_input_grad[/*value*/2]) {
        copy_range(grad_inputs, value_ix, std::get<6>(grad_result));
      }
      if (needs_input_grad[/*query_bias*/7]) {
        copy_range(grad_inputs, query_bias_ix, std::get<7>(grad_result));
      }
      if (needs_input_grad[/*key_bias*/8]) {
        copy_range(grad_inputs, key_bias_ix, std::get<8>(grad_result));
      }
      if (needs_input_grad[/*value_bias*/9]) {
        copy_range(grad_inputs, value_bias_ix, std::get<9>(grad_result));
      }
      if (needs_input_grad[/*out_proj_bias*/10]) {
        copy_range(grad_inputs, out_proj_bias_ix, std::get<10>(grad_result));
      }
  }
  return grad_inputs;
}
inline variable_list NpuMultiHeadAttentionBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 11>>();
  auto attn_dim_per_head = packed_args.unpack<int64_t>();
  auto attn_head_num = packed_args.unpack<int64_t>();
  auto dropout_prob = packed_args.unpack<double>();
  auto key = packed_args.unpack<Tensor>();
  auto key_bias = packed_args.unpack<Tensor>();
  auto key_weight = packed_args.unpack<Tensor>();
  auto out_proj_bias = packed_args.unpack<Tensor>();
  auto out_proj_weight = packed_args.unpack<Tensor>();
  auto query = packed_args.unpack<Tensor>();
  auto query_bias = packed_args.unpack<Tensor>();
  auto query_weight = packed_args.unpack<Tensor>();
  auto softmax_use_float = packed_args.unpack<bool>();
  auto src_len = packed_args.unpack<int64_t>();
  auto tgt_len = packed_args.unpack<int64_t>();
  auto value = packed_args.unpack<Tensor>();
  auto value_bias = packed_args.unpack<Tensor>();
  auto value_weight = packed_args.unpack<Tensor>();
  auto result1 = packed_args.unpack<Tensor>();
  auto result2 = packed_args.unpack<Tensor>();
  auto result3 = packed_args.unpack<Tensor>();
  auto result4 = packed_args.unpack<Tensor>();
  auto result5 = packed_args.unpack<Tensor>();
  auto result6 = packed_args.unpack<Tensor>();
  auto result7 = packed_args.unpack<Tensor>();
  return NpuMultiHeadAttentionBackward0_apply_functional(variable_list(grads), needs_input_grad, attn_dim_per_head, attn_head_num, dropout_prob, key, key_bias, key_weight, out_proj_bias, out_proj_weight, query, query_bias, query_weight, softmax_use_float, src_len, tgt_len, value, value_bias, value_weight, result1, result2, result3, result4, result5, result6, result7);
#endif
}

variable_list NpuMultiHeadAttentionBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto key = key_.unpack();
  auto key_bias = key_bias_.unpack();
  auto key_weight = key_weight_.unpack();
  auto out_proj_bias = out_proj_bias_.unpack();
  auto out_proj_weight = out_proj_weight_.unpack();
  auto query = query_.unpack();
  auto query_bias = query_bias_.unpack();
  auto query_weight = query_weight_.unpack();
  auto value = value_.unpack();
  auto value_bias = value_bias_.unpack();
  auto value_weight = value_weight_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  auto result4 = result4_.unpack(shared_from_this());
  auto result5 = result5_.unpack(shared_from_this());
  auto result6 = result6_.unpack(shared_from_this());
  auto result7 = result7_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto query_ix = gen.range(1);
  auto key_ix = gen.range(1);
  auto value_ix = gen.range(1);
  auto query_weight_ix = gen.range(1);
  auto key_weight_ix = gen.range(1);
  auto value_weight_ix = gen.range(1);
  auto out_proj_weight_ix = gen.range(1);
  auto query_bias_ix = gen.range(1);
  auto key_bias_ix = gen.range(1);
  auto value_bias_ix = gen.range(1);
  auto out_proj_bias_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 11>{
    task_should_compute_output({ query_ix }),
    task_should_compute_output({ key_ix }),
    task_should_compute_output({ value_ix }),
    task_should_compute_output({ query_weight_ix }),
    task_should_compute_output({ key_weight_ix }),
    task_should_compute_output({ value_weight_ix }),
    task_should_compute_output({ out_proj_weight_ix }),
    task_should_compute_output({ query_bias_ix }),
    task_should_compute_output({ key_bias_ix }),
    task_should_compute_output({ value_bias_ix }),
    task_should_compute_output({ out_proj_bias_ix }),
  };
  return NpuMultiHeadAttentionBackward0_apply_functional(std::move(grads), needs_input_grad, attn_dim_per_head, attn_head_num, dropout_prob, key, key_bias, key_weight, out_proj_bias, out_proj_weight, query, query_bias, query_weight, softmax_use_float, src_len, tgt_len, value, value_bias, value_weight, result1, result2, result3, result4, result5, result6, result7);
}

void NpuMultiHeadAttentionBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(attn_dim_per_head);
    args.collect(attn_head_num);
    args.collect(dropout_prob);
    args.collect(key_, false);
    args.collect(key_bias_, false);
    args.collect(key_weight_, false);
    args.collect(out_proj_bias_, false);
    args.collect(out_proj_weight_, false);
    args.collect(query_, false);
    args.collect(query_bias_, false);
    args.collect(query_weight_, false);
    args.collect(softmax_use_float);
    args.collect(src_len);
    args.collect(tgt_len);
    args.collect(value_, false);
    args.collect(value_bias_, false);
    args.collect(value_weight_, false);
    args.collect(result1_, true);
    args.collect(result2_, true);
    args.collect(result3_, true);
    args.collect(result4_, true);
    args.collect(result5_, true);
    args.collect(result6_, true);
    args.collect(result7_, true);
}
variable_list NpuMultiHeadAttentionBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(attn_dim_per_head);
  saved.before(attn_head_num);
  saved.before(dropout_prob);
  saved.before(key_);
  saved.before(key_bias_);
  saved.before(key_weight_);
  saved.before(out_proj_bias_);
  saved.before(out_proj_weight_);
  saved.before(query_);
  saved.before(query_bias_);
  saved.before(query_weight_);
  saved.before(softmax_use_float);
  saved.before(src_len);
  saved.before(tgt_len);
  saved.before(value_);
  saved.before(value_bias_);
  saved.before(value_weight_);
  saved.before(result1_);
  saved.before(result2_);
  saved.before(result3_);
  saved.before(result4_);
  saved.before(result5_);
  saved.before(result6_);
  saved.before(result7_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 11>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<double>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<bool>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuMultiHeadAttentionBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto key = key_.unpack();
  auto key_bias = key_bias_.unpack();
  auto key_weight = key_weight_.unpack();
  auto out_proj_bias = out_proj_bias_.unpack();
  auto out_proj_weight = out_proj_weight_.unpack();
  auto query = query_.unpack();
  auto query_bias = query_bias_.unpack();
  auto query_weight = query_weight_.unpack();
  auto value = value_.unpack();
  auto value_bias = value_bias_.unpack();
  auto value_weight = value_weight_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  auto result4 = result4_.unpack(shared_from_this());
  auto result5 = result5_.unpack(shared_from_this());
  auto result6 = result6_.unpack(shared_from_this());
  auto result7 = result7_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto query_ix = gen.range(1);
  auto key_ix = gen.range(1);
  auto value_ix = gen.range(1);
  auto query_weight_ix = gen.range(1);
  auto key_weight_ix = gen.range(1);
  auto value_weight_ix = gen.range(1);
  auto out_proj_weight_ix = gen.range(1);
  auto query_bias_ix = gen.range(1);
  auto key_bias_ix = gen.range(1);
  auto value_bias_ix = gen.range(1);
  auto out_proj_bias_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 11>{
    task_should_compute_output({ query_ix }),
    task_should_compute_output({ key_ix }),
    task_should_compute_output({ value_ix }),
    task_should_compute_output({ query_weight_ix }),
    task_should_compute_output({ key_weight_ix }),
    task_should_compute_output({ value_weight_ix }),
    task_should_compute_output({ out_proj_weight_ix }),
    task_should_compute_output({ query_bias_ix }),
    task_should_compute_output({ key_bias_ix }),
    task_should_compute_output({ value_bias_ix }),
    task_should_compute_output({ out_proj_bias_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(attn_dim_per_head);
  packed_args.pack(attn_head_num);
  packed_args.pack(dropout_prob);
  packed_args.pack(key);
  packed_args.pack(key_bias);
  packed_args.pack(key_weight);
  packed_args.pack(out_proj_bias);
  packed_args.pack(out_proj_weight);
  packed_args.pack(query);
  packed_args.pack(query_bias);
  packed_args.pack(query_weight);
  packed_args.pack(softmax_use_float);
  packed_args.pack(src_len);
  packed_args.pack(tgt_len);
  packed_args.pack(value);
  packed_args.pack(value_bias);
  packed_args.pack(value_weight);
  packed_args.pack(result1);
  packed_args.pack(result2);
  packed_args.pack(result3);
  packed_args.pack(result4);
  packed_args.pack(result5);
  packed_args.pack(result6);
  packed_args.pack(result7);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(attn_dim_per_head);
  saved.after(attn_head_num);
  saved.after(dropout_prob);
  saved.after(key_);
  saved.after(key_bias_);
  saved.after(key_weight_);
  saved.after(out_proj_bias_);
  saved.after(out_proj_weight_);
  saved.after(query_);
  saved.after(query_bias_);
  saved.after(query_weight_);
  saved.after(softmax_use_float);
  saved.after(src_len);
  saved.after(tgt_len);
  saved.after(value_);
  saved.after(value_bias_);
  saved.after(value_weight_);
  saved.after(result1_);
  saved.after(result2_);
  saved.after(result3_);
  saved.after(result4_);
  saved.after(result5_);
  saved.after(result6_);
  saved.after(result7_);
  return output_result;
#endif
}

static variable_list NpuPsRoiPoolingBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,1> needs_input_grad, int64_t& group_size, int64_t& output_dim, Tensor& rois, c10::SymInt& self_sym_argsize_2, c10::SymInt& self_sym_argsize_3, double& spatial_scale)
{
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (needs_input_grad[/*self*/0]) {
    auto grad_result = any_grad_defined ? (npu_ps_roi_pooling_backward(grad, rois, spatial_scale, group_size, output_dim, {self_sym_argsize_2, self_sym_argsize_3})) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
inline variable_list NpuPsRoiPoolingBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 1>>();
  auto group_size = packed_args.unpack<int64_t>();
  auto output_dim = packed_args.unpack<int64_t>();
  auto rois = packed_args.unpack<Tensor>();
  auto self_sym_argsize_2 = packed_args.unpack<c10::SymInt>();
  auto self_sym_argsize_3 = packed_args.unpack<c10::SymInt>();
  auto spatial_scale = packed_args.unpack<double>();
  return NpuPsRoiPoolingBackward0_apply_functional(variable_list(grads), needs_input_grad, group_size, output_dim, rois, self_sym_argsize_2, self_sym_argsize_3, spatial_scale);
#endif
}

variable_list NpuPsRoiPoolingBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto rois = rois_.unpack();
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  return NpuPsRoiPoolingBackward0_apply_functional(std::move(grads), needs_input_grad, group_size, output_dim, rois, self_sym_argsize_2, self_sym_argsize_3, spatial_scale);
}

void NpuPsRoiPoolingBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(group_size);
    args.collect(output_dim);
    args.collect(rois_, false);
    args.collect(self_sym_argsize_2);
    args.collect(self_sym_argsize_3);
    args.collect(spatial_scale);
}
variable_list NpuPsRoiPoolingBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(group_size);
  saved.before(output_dim);
  saved.before(rois_);
  saved.before(self_sym_argsize_2);
  saved.before(self_sym_argsize_3);
  saved.before(spatial_scale);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 1>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<c10::SymInt>::packed_type(),
      torch::dynamo::autograd::IValuePacker<c10::SymInt>::packed_type(),
      torch::dynamo::autograd::IValuePacker<double>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuPsRoiPoolingBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto rois = rois_.unpack();
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(group_size);
  packed_args.pack(output_dim);
  packed_args.pack(rois);
  packed_args.pack(self_sym_argsize_2);
  packed_args.pack(self_sym_argsize_3);
  packed_args.pack(spatial_scale);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(group_size);
  saved.after(output_dim);
  saved.after(rois_);
  saved.after(self_sym_argsize_2);
  saved.after(self_sym_argsize_3);
  saved.after(spatial_scale);
  return output_result;
#endif
}

static variable_list NpuRmsNormBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,2> needs_input_grad, Tensor& gamma, Tensor& self, Tensor& result1)
{
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto gamma_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  if (needs_input_grad[0] || needs_input_grad[1]) {

    auto grad_result = npu_rms_norm_backward(grad, self, gamma, result1);
      if (needs_input_grad[/*self*/0]) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (needs_input_grad[/*gamma*/1]) {
        copy_range(grad_inputs, gamma_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
inline variable_list NpuRmsNormBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 2>>();
  auto gamma = packed_args.unpack<Tensor>();
  auto self = packed_args.unpack<Tensor>();
  auto result1 = packed_args.unpack<Tensor>();
  return NpuRmsNormBackward0_apply_functional(variable_list(grads), needs_input_grad, gamma, self, result1);
#endif
}

variable_list NpuRmsNormBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto gamma = gamma_.unpack();
  auto self = self_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto gamma_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 2>{
    task_should_compute_output({ self_ix }),
    task_should_compute_output({ gamma_ix }),
  };
  return NpuRmsNormBackward0_apply_functional(std::move(grads), needs_input_grad, gamma, self, result1);
}

void NpuRmsNormBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(gamma_, false);
    args.collect(self_, false);
    args.collect(result1_, true);
}
variable_list NpuRmsNormBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(gamma_);
  saved.before(self_);
  saved.before(result1_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 2>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuRmsNormBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto gamma = gamma_.unpack();
  auto self = self_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto gamma_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 2>{
    task_should_compute_output({ self_ix }),
    task_should_compute_output({ gamma_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(gamma);
  packed_args.pack(self);
  packed_args.pack(result1);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(gamma_);
  saved.after(self_);
  saved.after(result1_);
  return output_result;
#endif
}

static variable_list NpuRotaryMulBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,3> needs_input_grad, Tensor& r1, Tensor& r2, std::string& rotary_mode, Tensor& self)
{
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto r1_ix = gen.range(1);
  auto r2_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  if (needs_input_grad[0] || needs_input_grad[1] || needs_input_grad[2]) {

    auto grad_result = npu_rotary_mul_backward(grad, self, r1, r2, rotary_mode);
      if (needs_input_grad[/*self*/0]) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (needs_input_grad[/*r1*/1]) {
        copy_range(grad_inputs, r1_ix, std::get<1>(grad_result));
      }
      if (needs_input_grad[/*r2*/2]) {
        copy_range(grad_inputs, r2_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
inline variable_list NpuRotaryMulBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 3>>();
  auto r1 = packed_args.unpack<Tensor>();
  auto r2 = packed_args.unpack<Tensor>();
  auto rotary_mode = packed_args.unpack<std::string>();
  auto self = packed_args.unpack<Tensor>();
  return NpuRotaryMulBackward0_apply_functional(variable_list(grads), needs_input_grad, r1, r2, rotary_mode, self);
#endif
}

variable_list NpuRotaryMulBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto r1 = r1_.unpack();
  auto r2 = r2_.unpack();
  auto self = self_.unpack();
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto r1_ix = gen.range(1);
  auto r2_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 3>{
    task_should_compute_output({ self_ix }),
    task_should_compute_output({ r1_ix }),
    task_should_compute_output({ r2_ix }),
  };
  return NpuRotaryMulBackward0_apply_functional(std::move(grads), needs_input_grad, r1, r2, rotary_mode, self);
}

void NpuRotaryMulBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(r1_, false);
    args.collect(r2_, false);
    args.collect(rotary_mode);
    args.collect(self_, false);
}
variable_list NpuRotaryMulBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(r1_);
  saved.before(r2_);
  saved.before(rotary_mode);
  saved.before(self_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 3>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<std::string>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuRotaryMulBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto r1 = r1_.unpack();
  auto r2 = r2_.unpack();
  auto self = self_.unpack();
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto r1_ix = gen.range(1);
  auto r2_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 3>{
    task_should_compute_output({ self_ix }),
    task_should_compute_output({ r1_ix }),
    task_should_compute_output({ r2_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(r1);
  packed_args.pack(r2);
  packed_args.pack(rotary_mode);
  packed_args.pack(self);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(r1_);
  saved.after(r2_);
  saved.after(rotary_mode);
  saved.after(self_);
  return output_result;
#endif
}

static variable_list NpuScaledMaskedSoftmaxBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,1> needs_input_grad, bool& fixed_triu_mask, Tensor& mask, at::Scalar& scale, Tensor& result)
{
  IndexRangeGenerator gen;
  auto x_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (needs_input_grad[/*x*/0]) {
    auto grad_result = any_grad_defined ? (npu_scaled_masked_softmax_backward(grad, result, mask, scale, fixed_triu_mask)) : Tensor();
    copy_range(grad_inputs, x_ix, grad_result);
  }
  return grad_inputs;
}
inline variable_list NpuScaledMaskedSoftmaxBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 1>>();
  auto fixed_triu_mask = packed_args.unpack<bool>();
  auto mask = packed_args.unpack<Tensor>();
  auto scale = packed_args.unpack<at::Scalar>();
  auto result = packed_args.unpack<Tensor>();
  return NpuScaledMaskedSoftmaxBackward0_apply_functional(variable_list(grads), needs_input_grad, fixed_triu_mask, mask, scale, result);
#endif
}

variable_list NpuScaledMaskedSoftmaxBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto mask = mask_.unpack();
  auto result = result_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto x_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ x_ix }),
  };
  return NpuScaledMaskedSoftmaxBackward0_apply_functional(std::move(grads), needs_input_grad, fixed_triu_mask, mask, scale, result);
}

void NpuScaledMaskedSoftmaxBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(fixed_triu_mask);
    args.collect(mask_, false);
    args.collect(scale);
    args.collect(result_, true);
}
variable_list NpuScaledMaskedSoftmaxBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(fixed_triu_mask);
  saved.before(mask_);
  saved.before(scale);
  saved.before(result_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 1>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<bool>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<at::Scalar>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuScaledMaskedSoftmaxBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto mask = mask_.unpack();
  auto result = result_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto x_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ x_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(fixed_triu_mask);
  packed_args.pack(mask);
  packed_args.pack(scale);
  packed_args.pack(result);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(fixed_triu_mask);
  saved.after(mask_);
  saved.after(scale);
  saved.after(result_);
  return output_result;
#endif
}

static variable_list NpuSiluBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,1> needs_input_grad, Tensor& self, Tensor& result)
{
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (needs_input_grad[/*self*/0]) {
    auto grad_result = any_grad_defined ? (npu_silu_backward(grad, self, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
inline variable_list NpuSiluBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 1>>();
  auto self = packed_args.unpack<Tensor>();
  auto result = packed_args.unpack<Tensor>();
  return NpuSiluBackward0_apply_functional(variable_list(grads), needs_input_grad, self, result);
#endif
}

variable_list NpuSiluBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  return NpuSiluBackward0_apply_functional(std::move(grads), needs_input_grad, self, result);
}

void NpuSiluBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(self_, false);
    args.collect(result_, true);
}
variable_list NpuSiluBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(self_);
  saved.before(result_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 1>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuSiluBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(self);
  packed_args.pack(result);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(self_);
  saved.after(result_);
  return output_result;
#endif
}

static variable_list NpuSoftmaxCrossEntropyWithLogitsBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,1> needs_input_grad, Tensor& labels, Tensor& self)
{
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (needs_input_grad[/*self*/0]) {
    auto grad_result = any_grad_defined ? (npu_softmax_cross_entropy_with_logits_backward(grad, self, labels)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
inline variable_list NpuSoftmaxCrossEntropyWithLogitsBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 1>>();
  auto labels = packed_args.unpack<Tensor>();
  auto self = packed_args.unpack<Tensor>();
  return NpuSoftmaxCrossEntropyWithLogitsBackward0_apply_functional(variable_list(grads), needs_input_grad, labels, self);
#endif
}

variable_list NpuSoftmaxCrossEntropyWithLogitsBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto labels = labels_.unpack();
  auto self = self_.unpack();
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  return NpuSoftmaxCrossEntropyWithLogitsBackward0_apply_functional(std::move(grads), needs_input_grad, labels, self);
}

void NpuSoftmaxCrossEntropyWithLogitsBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(labels_, false);
    args.collect(self_, false);
}
variable_list NpuSoftmaxCrossEntropyWithLogitsBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(labels_);
  saved.before(self_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 1>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuSoftmaxCrossEntropyWithLogitsBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto labels = labels_.unpack();
  auto self = self_.unpack();
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(labels);
  packed_args.pack(self);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(labels_);
  saved.after(self_);
  return output_result;
#endif
}

static variable_list NpuSwigluBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,1> needs_input_grad, int64_t& dim, Tensor& self)
{
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (needs_input_grad[/*self*/0]) {
    auto grad_result = any_grad_defined ? (npu_swiglu_backward(grad, self, dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
inline variable_list NpuSwigluBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 1>>();
  auto dim = packed_args.unpack<int64_t>();
  auto self = packed_args.unpack<Tensor>();
  return NpuSwigluBackward0_apply_functional(variable_list(grads), needs_input_grad, dim, self);
#endif
}

variable_list NpuSwigluBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto self = self_.unpack();
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  return NpuSwigluBackward0_apply_functional(std::move(grads), needs_input_grad, dim, self);
}

void NpuSwigluBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(dim);
    args.collect(self_, false);
}
variable_list NpuSwigluBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(dim);
  saved.before(self_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 1>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuSwigluBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto self = self_.unpack();
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(dim);
  packed_args.pack(self);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(dim);
  saved.after(self_);
  return output_result;
#endif
}

static variable_list RepeatInterleaveBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,1> needs_input_grad, ::std::optional<int64_t>& dim, Tensor& repeats, Tensor& self)
{
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (needs_input_grad[/*self*/0]) {
    auto grad_result = any_grad_defined ? (repeat_interleave_backward_tensor(grad, self, repeats, dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
inline variable_list RepeatInterleaveBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 1>>();
  auto dim = packed_args.unpack<::std::optional<int64_t>>();
  auto repeats = packed_args.unpack<Tensor>();
  auto self = packed_args.unpack<Tensor>();
  return RepeatInterleaveBackward0_apply_functional(variable_list(grads), needs_input_grad, dim, repeats, self);
#endif
}

variable_list RepeatInterleaveBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto repeats = repeats_.unpack();
  auto self = self_.unpack();
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  return RepeatInterleaveBackward0_apply_functional(std::move(grads), needs_input_grad, dim, repeats, self);
}

void RepeatInterleaveBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(dim);
    args.collect(repeats_, false);
    args.collect(self_, false);
}
variable_list RepeatInterleaveBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(dim);
  saved.before(repeats_);
  saved.before(self_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 1>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<::std::optional<int64_t>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), RepeatInterleaveBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto repeats = repeats_.unpack();
  auto self = self_.unpack();
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(dim);
  packed_args.pack(repeats);
  packed_args.pack(self);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(dim);
  saved.after(repeats_);
  saved.after(self_);
  return output_result;
#endif
}

static variable_list RepeatInterleaveBackward1_apply_functional(
  variable_list&& grads,
  std::array<bool,1> needs_input_grad, ::std::optional<int64_t>& dim, c10::SymInt& repeats, Tensor& self)
{
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (needs_input_grad[/*self*/0]) {
    auto grad_result = any_grad_defined ? (repeat_interleave_backward_int(grad, self, repeats, dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
inline variable_list RepeatInterleaveBackward1_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 1>>();
  auto dim = packed_args.unpack<::std::optional<int64_t>>();
  auto repeats = packed_args.unpack<c10::SymInt>();
  auto self = packed_args.unpack<Tensor>();
  return RepeatInterleaveBackward1_apply_functional(variable_list(grads), needs_input_grad, dim, repeats, self);
#endif
}

variable_list RepeatInterleaveBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto self = self_.unpack();
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  return RepeatInterleaveBackward1_apply_functional(std::move(grads), needs_input_grad, dim, repeats, self);
}

void RepeatInterleaveBackward1::compiled_args(CompiledNodeArgs& args) const {
    args.collect(dim);
    args.collect(repeats);
    args.collect(self_, false);
}
variable_list RepeatInterleaveBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(dim);
  saved.before(repeats);
  saved.before(self_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 1>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<::std::optional<int64_t>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<c10::SymInt>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), RepeatInterleaveBackward1_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto self = self_.unpack();
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(dim);
  packed_args.pack(repeats);
  packed_args.pack(self);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(dim);
  saved.after(repeats);
  saved.after(self_);
  return output_result;
#endif
}

static variable_list StftBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,1> needs_input_grad, ::std::optional<int64_t>& hop_length, int64_t& n_fft, bool& normalized, ::std::optional<bool>& onesided, ::std::optional<bool>& return_complex, Tensor& self, ::std::optional<int64_t>& win_length, Tensor& window)
{
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (needs_input_grad[/*self*/0]) {
    auto grad_result = any_grad_defined ? (stft_backward(grad, self, n_fft, hop_length, win_length, window, normalized, onesided, return_complex)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
inline variable_list StftBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 1>>();
  auto hop_length = packed_args.unpack<::std::optional<int64_t>>();
  auto n_fft = packed_args.unpack<int64_t>();
  auto normalized = packed_args.unpack<bool>();
  auto onesided = packed_args.unpack<::std::optional<bool>>();
  auto return_complex = packed_args.unpack<::std::optional<bool>>();
  auto self = packed_args.unpack<Tensor>();
  auto win_length = packed_args.unpack<::std::optional<int64_t>>();
  auto window = packed_args.unpack<Tensor>();
  return StftBackward0_apply_functional(variable_list(grads), needs_input_grad, hop_length, n_fft, normalized, onesided, return_complex, self, win_length, window);
#endif
}

variable_list StftBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto self = self_.unpack();
  auto window = window_.unpack();
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  return StftBackward0_apply_functional(std::move(grads), needs_input_grad, hop_length, n_fft, normalized, onesided, return_complex, self, win_length, window);
}

void StftBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(hop_length);
    args.collect(n_fft);
    args.collect(normalized);
    args.collect(onesided);
    args.collect(return_complex);
    args.collect(self_, false);
    args.collect(win_length);
    args.collect(window_, false);
}
variable_list StftBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(hop_length);
  saved.before(n_fft);
  saved.before(normalized);
  saved.before(onesided);
  saved.before(return_complex);
  saved.before(self_);
  saved.before(win_length);
  saved.before(window_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 1>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<::std::optional<int64_t>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<bool>::packed_type(),
      torch::dynamo::autograd::IValuePacker<::std::optional<bool>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<::std::optional<bool>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<::std::optional<int64_t>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), StftBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto self = self_.unpack();
  auto window = window_.unpack();
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(hop_length);
  packed_args.pack(n_fft);
  packed_args.pack(normalized);
  packed_args.pack(onesided);
  packed_args.pack(return_complex);
  packed_args.pack(self);
  packed_args.pack(win_length);
  packed_args.pack(window);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(hop_length);
  saved.after(n_fft);
  saved.after(normalized);
  saved.after(onesided);
  saved.after(return_complex);
  saved.after(self_);
  saved.after(win_length);
  saved.after(window_);
  return output_result;
#endif
}

static variable_list FftR2CBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,1> needs_input_grad, std::vector<int64_t>& dim, int64_t& normalization, bool& onesided, Tensor& self)
{
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (needs_input_grad[/*self*/0]) {
    auto grad_result = any_grad_defined ? (fft_r2c_backward(grad, dim, normalization, onesided, self.size(dim.back()))) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
inline variable_list FftR2CBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 1>>();
  auto dim = packed_args.unpack<std::vector<int64_t>>();
  auto normalization = packed_args.unpack<int64_t>();
  auto onesided = packed_args.unpack<bool>();
  auto self = packed_args.unpack<Tensor>();
  return FftR2CBackward0_apply_functional(variable_list(grads), needs_input_grad, dim, normalization, onesided, self);
#endif
}

variable_list FftR2CBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto self = self_.unpack();
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  return FftR2CBackward0_apply_functional(std::move(grads), needs_input_grad, dim, normalization, onesided, self);
}

void FftR2CBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(dim);
    args.collect(normalization);
    args.collect(onesided);
    args.collect(self_, false);
}
variable_list FftR2CBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(dim);
  saved.before(normalization);
  saved.before(onesided);
  saved.before(self_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 1>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<std::vector<int64_t>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<bool>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), FftR2CBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto self = self_.unpack();
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(dim);
  packed_args.pack(normalization);
  packed_args.pack(onesided);
  packed_args.pack(self);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(dim);
  saved.after(normalization);
  saved.after(onesided);
  saved.after(self_);
  return output_result;
#endif
}

static variable_list FftC2RBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,1> needs_input_grad, std::vector<int64_t>& dim, int64_t& normalization)
{
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (needs_input_grad[/*self*/0]) {
    auto grad_result = any_grad_defined ? (fft_c2r_backward(grad, dim, normalization)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
inline variable_list FftC2RBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 1>>();
  auto dim = packed_args.unpack<std::vector<int64_t>>();
  auto normalization = packed_args.unpack<int64_t>();
  return FftC2RBackward0_apply_functional(variable_list(grads), needs_input_grad, dim, normalization);
#endif
}

variable_list FftC2RBackward0::apply(variable_list&& grads) {



  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  return FftC2RBackward0_apply_functional(std::move(grads), needs_input_grad, dim, normalization);
}

void FftC2RBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(dim);
    args.collect(normalization);
}
variable_list FftC2RBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(dim);
  saved.before(normalization);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 1>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<std::vector<int64_t>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), FftC2RBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(dim);
  packed_args.pack(normalization);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(dim);
  saved.after(normalization);
  return output_result;
#endif
}

static variable_list NpuMoeTokenPermuteWithRoutingMapBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,2> needs_input_grad, bool& drop_and_pad, Tensor& routing_map, c10::SymInt& routing_map_sym_argsize_1, c10::SymInt& tokens_sym_argsize_0, Tensor& result2)
{
  IndexRangeGenerator gen;
  auto tokens_ix = gen.range(1);
  auto probs_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (needs_input_grad[0] || needs_input_grad[1]) {

    auto grad_result = npu_moe_token_permute_with_routing_map_grad(grads[0], grads[1], result2, routing_map, routing_map_sym_argsize_1, tokens_sym_argsize_0, drop_and_pad);
      if (needs_input_grad[/*tokens*/0]) {
        copy_range(grad_inputs, tokens_ix, std::get<0>(grad_result));
      }
      if (needs_input_grad[/*probs*/1]) {
        copy_range(grad_inputs, probs_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
inline variable_list NpuMoeTokenPermuteWithRoutingMapBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 2>>();
  auto drop_and_pad = packed_args.unpack<bool>();
  auto routing_map = packed_args.unpack<Tensor>();
  auto routing_map_sym_argsize_1 = packed_args.unpack<c10::SymInt>();
  auto tokens_sym_argsize_0 = packed_args.unpack<c10::SymInt>();
  auto result2 = packed_args.unpack<Tensor>();
  return NpuMoeTokenPermuteWithRoutingMapBackward0_apply_functional(variable_list(grads), needs_input_grad, drop_and_pad, routing_map, routing_map_sym_argsize_1, tokens_sym_argsize_0, result2);
#endif
}

variable_list NpuMoeTokenPermuteWithRoutingMapBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto routing_map = routing_map_.unpack();
  auto result2 = result2_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto tokens_ix = gen.range(1);
  auto probs_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 2>{
    task_should_compute_output({ tokens_ix }),
    task_should_compute_output({ probs_ix }),
  };
  return NpuMoeTokenPermuteWithRoutingMapBackward0_apply_functional(std::move(grads), needs_input_grad, drop_and_pad, routing_map, routing_map_sym_argsize_1, tokens_sym_argsize_0, result2);
}

void NpuMoeTokenPermuteWithRoutingMapBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(drop_and_pad);
    args.collect(routing_map_, false);
    args.collect(routing_map_sym_argsize_1);
    args.collect(tokens_sym_argsize_0);
    args.collect(result2_, true);
}
variable_list NpuMoeTokenPermuteWithRoutingMapBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(drop_and_pad);
  saved.before(routing_map_);
  saved.before(routing_map_sym_argsize_1);
  saved.before(tokens_sym_argsize_0);
  saved.before(result2_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 2>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<bool>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<c10::SymInt>::packed_type(),
      torch::dynamo::autograd::IValuePacker<c10::SymInt>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuMoeTokenPermuteWithRoutingMapBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto routing_map = routing_map_.unpack();
  auto result2 = result2_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto tokens_ix = gen.range(1);
  auto probs_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 2>{
    task_should_compute_output({ tokens_ix }),
    task_should_compute_output({ probs_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(drop_and_pad);
  packed_args.pack(routing_map);
  packed_args.pack(routing_map_sym_argsize_1);
  packed_args.pack(tokens_sym_argsize_0);
  packed_args.pack(result2);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(drop_and_pad);
  saved.after(routing_map_);
  saved.after(routing_map_sym_argsize_1);
  saved.after(tokens_sym_argsize_0);
  saved.after(result2_);
  return output_result;
#endif
}

static variable_list NpuGroupNormSwishBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,3> needs_input_grad, Tensor& bias, Tensor& input, int64_t& num_groups, ::std::optional<double>& swish_scale, Tensor& weight, Tensor& result1, Tensor& result2)
{
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  if (needs_input_grad[0] || needs_input_grad[1] || needs_input_grad[2]) {
      auto grad_input_mask = std::array<bool, 3>{
        needs_input_grad[0],
        needs_input_grad[1],
        needs_input_grad[2],
      };
    auto grad_result = npu_group_norm_swish_grad(grad, input, num_groups, weight, bias, result1, result2, grad_input_mask, swish_scale);
      if (needs_input_grad[/*input*/0]) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (needs_input_grad[/*weight*/1]) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (needs_input_grad[/*bias*/2]) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
inline variable_list NpuGroupNormSwishBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 3>>();
  auto bias = packed_args.unpack<Tensor>();
  auto input = packed_args.unpack<Tensor>();
  auto num_groups = packed_args.unpack<int64_t>();
  auto swish_scale = packed_args.unpack<::std::optional<double>>();
  auto weight = packed_args.unpack<Tensor>();
  auto result1 = packed_args.unpack<Tensor>();
  auto result2 = packed_args.unpack<Tensor>();
  return NpuGroupNormSwishBackward0_apply_functional(variable_list(grads), needs_input_grad, bias, input, num_groups, swish_scale, weight, result1, result2);
#endif
}

variable_list NpuGroupNormSwishBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto bias = bias_.unpack();
  auto input = input_.unpack();
  auto weight = weight_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 3>{
    task_should_compute_output({ input_ix }),
    task_should_compute_output({ weight_ix }),
    task_should_compute_output({ bias_ix }),
  };
  return NpuGroupNormSwishBackward0_apply_functional(std::move(grads), needs_input_grad, bias, input, num_groups, swish_scale, weight, result1, result2);
}

void NpuGroupNormSwishBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(bias_, false);
    args.collect(input_, false);
    args.collect(num_groups);
    args.collect(swish_scale);
    args.collect(weight_, false);
    args.collect(result1_, true);
    args.collect(result2_, true);
}
variable_list NpuGroupNormSwishBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(bias_);
  saved.before(input_);
  saved.before(num_groups);
  saved.before(swish_scale);
  saved.before(weight_);
  saved.before(result1_);
  saved.before(result2_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 3>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<::std::optional<double>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuGroupNormSwishBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto bias = bias_.unpack();
  auto input = input_.unpack();
  auto weight = weight_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 3>{
    task_should_compute_output({ input_ix }),
    task_should_compute_output({ weight_ix }),
    task_should_compute_output({ bias_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(bias);
  packed_args.pack(input);
  packed_args.pack(num_groups);
  packed_args.pack(swish_scale);
  packed_args.pack(weight);
  packed_args.pack(result1);
  packed_args.pack(result2);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(bias_);
  saved.after(input_);
  saved.after(num_groups);
  saved.after(swish_scale);
  saved.after(weight_);
  saved.after(result1_);
  saved.after(result2_);
  return output_result;
#endif
}

static variable_list NpuCrossEntropyLossBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,1> needs_input_grad, int64_t& ignore_index, double& label_smoothing, double& lse_square_scale_for_zloss, std::string& reduction, Tensor& target, Tensor& weight, Tensor& result1, Tensor& result3)
{
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (needs_input_grad[/*input*/0]) {
    auto grad_result = any_grad_defined ? (npu_cross_entropy_loss_backward(grads[0], result1, target, weight, grads[1], result3, reduction, ignore_index, label_smoothing, lse_square_scale_for_zloss)) : Tensor();
    copy_range(grad_inputs, input_ix, grad_result);
  }
  return grad_inputs;
}
inline variable_list NpuCrossEntropyLossBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 1>>();
  auto ignore_index = packed_args.unpack<int64_t>();
  auto label_smoothing = packed_args.unpack<double>();
  auto lse_square_scale_for_zloss = packed_args.unpack<double>();
  auto reduction = packed_args.unpack<std::string>();
  auto target = packed_args.unpack<Tensor>();
  auto weight = packed_args.unpack<Tensor>();
  auto result1 = packed_args.unpack<Tensor>();
  auto result3 = packed_args.unpack<Tensor>();
  return NpuCrossEntropyLossBackward0_apply_functional(variable_list(grads), needs_input_grad, ignore_index, label_smoothing, lse_square_scale_for_zloss, reduction, target, weight, result1, result3);
#endif
}

variable_list NpuCrossEntropyLossBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto target = target_.unpack();
  auto weight = weight_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ input_ix }),
  };
  return NpuCrossEntropyLossBackward0_apply_functional(std::move(grads), needs_input_grad, ignore_index, label_smoothing, lse_square_scale_for_zloss, reduction, target, weight, result1, result3);
}

void NpuCrossEntropyLossBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(ignore_index);
    args.collect(label_smoothing);
    args.collect(lse_square_scale_for_zloss);
    args.collect(reduction);
    args.collect(target_, false);
    args.collect(weight_, false);
    args.collect(result1_, true);
    args.collect(result3_, true);
}
variable_list NpuCrossEntropyLossBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(ignore_index);
  saved.before(label_smoothing);
  saved.before(lse_square_scale_for_zloss);
  saved.before(reduction);
  saved.before(target_);
  saved.before(weight_);
  saved.before(result1_);
  saved.before(result3_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 1>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<double>::packed_type(),
      torch::dynamo::autograd::IValuePacker<double>::packed_type(),
      torch::dynamo::autograd::IValuePacker<std::string>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuCrossEntropyLossBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto target = target_.unpack();
  auto weight = weight_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ input_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(ignore_index);
  packed_args.pack(label_smoothing);
  packed_args.pack(lse_square_scale_for_zloss);
  packed_args.pack(reduction);
  packed_args.pack(target);
  packed_args.pack(weight);
  packed_args.pack(result1);
  packed_args.pack(result3);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(ignore_index);
  saved.after(label_smoothing);
  saved.after(lse_square_scale_for_zloss);
  saved.after(reduction);
  saved.after(target_);
  saved.after(weight_);
  saved.after(result1_);
  saved.after(result3_);
  return output_result;
#endif
}

static variable_list NpuNsaCompressBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,2> needs_input_grad, c10::OptionalArray<int64_t>& actual_seq_len, int64_t& compress_block_size, int64_t& compress_stride, Tensor& input, Tensor& weight)
{
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  if (needs_input_grad[0] || needs_input_grad[1]) {

    auto grad_result = npu_nsa_compress_grad(grad, input, weight, compress_block_size, compress_stride, actual_seq_len);
      if (needs_input_grad[/*input*/0]) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (needs_input_grad[/*weight*/1]) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
inline variable_list NpuNsaCompressBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 2>>();
  auto actual_seq_len = packed_args.unpack<c10::OptionalArray<int64_t>>();
  auto compress_block_size = packed_args.unpack<int64_t>();
  auto compress_stride = packed_args.unpack<int64_t>();
  auto input = packed_args.unpack<Tensor>();
  auto weight = packed_args.unpack<Tensor>();
  return NpuNsaCompressBackward0_apply_functional(variable_list(grads), needs_input_grad, actual_seq_len, compress_block_size, compress_stride, input, weight);
#endif
}

variable_list NpuNsaCompressBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto input = input_.unpack();
  auto weight = weight_.unpack();
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 2>{
    task_should_compute_output({ input_ix }),
    task_should_compute_output({ weight_ix }),
  };
  return NpuNsaCompressBackward0_apply_functional(std::move(grads), needs_input_grad, actual_seq_len, compress_block_size, compress_stride, input, weight);
}

void NpuNsaCompressBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(actual_seq_len);
    args.collect(compress_block_size);
    args.collect(compress_stride);
    args.collect(input_, false);
    args.collect(weight_, false);
}
variable_list NpuNsaCompressBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(actual_seq_len);
  saved.before(compress_block_size);
  saved.before(compress_stride);
  saved.before(input_);
  saved.before(weight_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 2>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<c10::OptionalArray<int64_t>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuNsaCompressBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto input = input_.unpack();
  auto weight = weight_.unpack();
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 2>{
    task_should_compute_output({ input_ix }),
    task_should_compute_output({ weight_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(actual_seq_len);
  packed_args.pack(compress_block_size);
  packed_args.pack(compress_stride);
  packed_args.pack(input);
  packed_args.pack(weight);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(actual_seq_len);
  saved.after(compress_block_size);
  saved.after(compress_stride);
  saved.after(input_);
  saved.after(weight_);
  return output_result;
#endif
}

static variable_list NpuNsaSelectAttentionBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,3> needs_input_grad, c10::OptionalArray<int64_t>& actual_seq_kvlen, c10::OptionalArray<int64_t>& actual_seq_qlen, Tensor& atten_mask, int64_t& head_num, Tensor& key, Tensor& query, double& scale_value, int64_t& select_block_count, int64_t& select_block_size, Tensor& topk_indices, Tensor& value, Tensor& result0, Tensor& result1, Tensor& result2)
{
  IndexRangeGenerator gen;
  auto query_ix = gen.range(1);
  auto key_ix = gen.range(1);
  auto value_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  if (needs_input_grad[0] || needs_input_grad[1] || needs_input_grad[2]) {

    auto grad_result = npu_nsa_select_attention_grad(grad, query, key, value, result0, result1, result2, topk_indices, scale_value, head_num, select_block_size, select_block_count, atten_mask, actual_seq_qlen, actual_seq_kvlen);
      if (needs_input_grad[/*query*/0]) {
        copy_range(grad_inputs, query_ix, std::get<0>(grad_result));
      }
      if (needs_input_grad[/*key*/1]) {
        copy_range(grad_inputs, key_ix, std::get<1>(grad_result));
      }
      if (needs_input_grad[/*value*/2]) {
        copy_range(grad_inputs, value_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
inline variable_list NpuNsaSelectAttentionBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 3>>();
  auto actual_seq_kvlen = packed_args.unpack<c10::OptionalArray<int64_t>>();
  auto actual_seq_qlen = packed_args.unpack<c10::OptionalArray<int64_t>>();
  auto atten_mask = packed_args.unpack<Tensor>();
  auto head_num = packed_args.unpack<int64_t>();
  auto key = packed_args.unpack<Tensor>();
  auto query = packed_args.unpack<Tensor>();
  auto scale_value = packed_args.unpack<double>();
  auto select_block_count = packed_args.unpack<int64_t>();
  auto select_block_size = packed_args.unpack<int64_t>();
  auto topk_indices = packed_args.unpack<Tensor>();
  auto value = packed_args.unpack<Tensor>();
  auto result0 = packed_args.unpack<Tensor>();
  auto result1 = packed_args.unpack<Tensor>();
  auto result2 = packed_args.unpack<Tensor>();
  return NpuNsaSelectAttentionBackward0_apply_functional(variable_list(grads), needs_input_grad, actual_seq_kvlen, actual_seq_qlen, atten_mask, head_num, key, query, scale_value, select_block_count, select_block_size, topk_indices, value, result0, result1, result2);
#endif
}

variable_list NpuNsaSelectAttentionBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto atten_mask = atten_mask_.unpack();
  auto key = key_.unpack();
  auto query = query_.unpack();
  auto topk_indices = topk_indices_.unpack();
  auto value = value_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto query_ix = gen.range(1);
  auto key_ix = gen.range(1);
  auto value_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 3>{
    task_should_compute_output({ query_ix }),
    task_should_compute_output({ key_ix }),
    task_should_compute_output({ value_ix }),
  };
  return NpuNsaSelectAttentionBackward0_apply_functional(std::move(grads), needs_input_grad, actual_seq_kvlen, actual_seq_qlen, atten_mask, head_num, key, query, scale_value, select_block_count, select_block_size, topk_indices, value, result0, result1, result2);
}

void NpuNsaSelectAttentionBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(actual_seq_kvlen);
    args.collect(actual_seq_qlen);
    args.collect(atten_mask_, false);
    args.collect(head_num);
    args.collect(key_, false);
    args.collect(query_, false);
    args.collect(scale_value);
    args.collect(select_block_count);
    args.collect(select_block_size);
    args.collect(topk_indices_, false);
    args.collect(value_, false);
    args.collect(result0_, true);
    args.collect(result1_, true);
    args.collect(result2_, true);
}
variable_list NpuNsaSelectAttentionBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(actual_seq_kvlen);
  saved.before(actual_seq_qlen);
  saved.before(atten_mask_);
  saved.before(head_num);
  saved.before(key_);
  saved.before(query_);
  saved.before(scale_value);
  saved.before(select_block_count);
  saved.before(select_block_size);
  saved.before(topk_indices_);
  saved.before(value_);
  saved.before(result0_);
  saved.before(result1_);
  saved.before(result2_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 3>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<c10::OptionalArray<int64_t>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<c10::OptionalArray<int64_t>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<double>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuNsaSelectAttentionBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto atten_mask = atten_mask_.unpack();
  auto key = key_.unpack();
  auto query = query_.unpack();
  auto topk_indices = topk_indices_.unpack();
  auto value = value_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto query_ix = gen.range(1);
  auto key_ix = gen.range(1);
  auto value_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 3>{
    task_should_compute_output({ query_ix }),
    task_should_compute_output({ key_ix }),
    task_should_compute_output({ value_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(actual_seq_kvlen);
  packed_args.pack(actual_seq_qlen);
  packed_args.pack(atten_mask);
  packed_args.pack(head_num);
  packed_args.pack(key);
  packed_args.pack(query);
  packed_args.pack(scale_value);
  packed_args.pack(select_block_count);
  packed_args.pack(select_block_size);
  packed_args.pack(topk_indices);
  packed_args.pack(value);
  packed_args.pack(result0);
  packed_args.pack(result1);
  packed_args.pack(result2);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(actual_seq_kvlen);
  saved.after(actual_seq_qlen);
  saved.after(atten_mask_);
  saved.after(head_num);
  saved.after(key_);
  saved.after(query_);
  saved.after(scale_value);
  saved.after(select_block_count);
  saved.after(select_block_size);
  saved.after(topk_indices_);
  saved.after(value_);
  saved.after(result0_);
  saved.after(result1_);
  saved.after(result2_);
  return output_result;
#endif
}

static variable_list NpuNsaCompressAttentionBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,3> needs_input_grad, c10::OptionalArray<int64_t>& actual_cmp_seq_kvlen, c10::OptionalArray<int64_t>& actual_seq_qlen, Tensor& atten_mask, int64_t& head_num, Tensor& key, Tensor& query, double& scale_value, Tensor& value, Tensor& result0, Tensor& result2, Tensor& result3)
{
  IndexRangeGenerator gen;
  auto query_ix = gen.range(1);
  auto key_ix = gen.range(1);
  auto value_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  if (needs_input_grad[0] || needs_input_grad[1] || needs_input_grad[2]) {

    auto grad_result = npu_fusion_attention_grad(query, key, value, grad, head_num, "TND", at::Tensor(), at::Tensor(), atten_mask, result2, result3, at::Tensor(), result0, scale_value, 1., 2147483647, 2147483647, 0, 0, 0, 0, at::IntArrayRef{}, actual_seq_qlen, actual_cmp_seq_kvlen, 1, true, false);
      if (needs_input_grad[/*query*/0]) {
        copy_range(grad_inputs, query_ix, std::get<0>(grad_result));
      }
      if (needs_input_grad[/*key*/1]) {
        copy_range(grad_inputs, key_ix, std::get<1>(grad_result));
      }
      if (needs_input_grad[/*value*/2]) {
        copy_range(grad_inputs, value_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
inline variable_list NpuNsaCompressAttentionBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 3>>();
  auto actual_cmp_seq_kvlen = packed_args.unpack<c10::OptionalArray<int64_t>>();
  auto actual_seq_qlen = packed_args.unpack<c10::OptionalArray<int64_t>>();
  auto atten_mask = packed_args.unpack<Tensor>();
  auto head_num = packed_args.unpack<int64_t>();
  auto key = packed_args.unpack<Tensor>();
  auto query = packed_args.unpack<Tensor>();
  auto scale_value = packed_args.unpack<double>();
  auto value = packed_args.unpack<Tensor>();
  auto result0 = packed_args.unpack<Tensor>();
  auto result2 = packed_args.unpack<Tensor>();
  auto result3 = packed_args.unpack<Tensor>();
  return NpuNsaCompressAttentionBackward0_apply_functional(variable_list(grads), needs_input_grad, actual_cmp_seq_kvlen, actual_seq_qlen, atten_mask, head_num, key, query, scale_value, value, result0, result2, result3);
#endif
}

variable_list NpuNsaCompressAttentionBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto atten_mask = atten_mask_.unpack();
  auto key = key_.unpack();
  auto query = query_.unpack();
  auto value = value_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto query_ix = gen.range(1);
  auto key_ix = gen.range(1);
  auto value_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 3>{
    task_should_compute_output({ query_ix }),
    task_should_compute_output({ key_ix }),
    task_should_compute_output({ value_ix }),
  };
  return NpuNsaCompressAttentionBackward0_apply_functional(std::move(grads), needs_input_grad, actual_cmp_seq_kvlen, actual_seq_qlen, atten_mask, head_num, key, query, scale_value, value, result0, result2, result3);
}

void NpuNsaCompressAttentionBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(actual_cmp_seq_kvlen);
    args.collect(actual_seq_qlen);
    args.collect(atten_mask_, false);
    args.collect(head_num);
    args.collect(key_, false);
    args.collect(query_, false);
    args.collect(scale_value);
    args.collect(value_, false);
    args.collect(result0_, true);
    args.collect(result2_, true);
    args.collect(result3_, true);
}
variable_list NpuNsaCompressAttentionBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(actual_cmp_seq_kvlen);
  saved.before(actual_seq_qlen);
  saved.before(atten_mask_);
  saved.before(head_num);
  saved.before(key_);
  saved.before(query_);
  saved.before(scale_value);
  saved.before(value_);
  saved.before(result0_);
  saved.before(result2_);
  saved.before(result3_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 3>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<c10::OptionalArray<int64_t>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<c10::OptionalArray<int64_t>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<int64_t>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<double>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuNsaCompressAttentionBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto atten_mask = atten_mask_.unpack();
  auto key = key_.unpack();
  auto query = query_.unpack();
  auto value = value_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto query_ix = gen.range(1);
  auto key_ix = gen.range(1);
  auto value_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 3>{
    task_should_compute_output({ query_ix }),
    task_should_compute_output({ key_ix }),
    task_should_compute_output({ value_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(actual_cmp_seq_kvlen);
  packed_args.pack(actual_seq_qlen);
  packed_args.pack(atten_mask);
  packed_args.pack(head_num);
  packed_args.pack(key);
  packed_args.pack(query);
  packed_args.pack(scale_value);
  packed_args.pack(value);
  packed_args.pack(result0);
  packed_args.pack(result2);
  packed_args.pack(result3);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(actual_cmp_seq_kvlen);
  saved.after(actual_seq_qlen);
  saved.after(atten_mask_);
  saved.after(head_num);
  saved.after(key_);
  saved.after(query_);
  saved.after(scale_value);
  saved.after(value_);
  saved.after(result0_);
  saved.after(result2_);
  saved.after(result3_);
  return output_result;
#endif
}

static variable_list InverseBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,1> needs_input_grad, Tensor& result)
{
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (needs_input_grad[/*self*/0]) {
    auto grad_result = any_grad_defined ? (-at::matmul(result.mH(), at::matmul(grad, result.mH()))) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
inline variable_list InverseBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 1>>();
  auto result = packed_args.unpack<Tensor>();
  return InverseBackward0_apply_functional(variable_list(grads), needs_input_grad, result);
#endif
}

variable_list InverseBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto result = result_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  return InverseBackward0_apply_functional(std::move(grads), needs_input_grad, result);
}

void InverseBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(result_, true);
}
variable_list InverseBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(result_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 1>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), InverseBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto result = result_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ self_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(result);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(result_);
  return output_result;
#endif
}

static variable_list NpuMoeTokenUnpermuteWithRoutingMapBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,2> needs_input_grad, bool& drop_and_pad, Tensor& permuted_tokens, Tensor& probs, std::vector<int64_t>& restore_shape, Tensor& routing_map, Tensor& sorted_indices, Tensor& result1, Tensor& result2)
{
  IndexRangeGenerator gen;
  auto permuted_tokens_ix = gen.range(1);
  auto probs_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  if (needs_input_grad[0] || needs_input_grad[1]) {

    auto grad_result = drop_and_pad ? npu_moe_token_unpermute_with_routing_map_grad(grad, result1, result2, routing_map, permuted_tokens, probs, drop_and_pad, restore_shape) : npu_moe_token_unpermute_with_routing_map_grad(grad, sorted_indices, sorted_indices, routing_map, permuted_tokens, probs, drop_and_pad, restore_shape);
      if (needs_input_grad[/*permuted_tokens*/0]) {
        copy_range(grad_inputs, permuted_tokens_ix, std::get<0>(grad_result));
      }
      if (needs_input_grad[/*probs*/1]) {
        copy_range(grad_inputs, probs_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
inline variable_list NpuMoeTokenUnpermuteWithRoutingMapBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 2>>();
  auto drop_and_pad = packed_args.unpack<bool>();
  auto permuted_tokens = packed_args.unpack<Tensor>();
  auto probs = packed_args.unpack<Tensor>();
  auto restore_shape = packed_args.unpack<std::vector<int64_t>>();
  auto routing_map = packed_args.unpack<Tensor>();
  auto sorted_indices = packed_args.unpack<Tensor>();
  auto result1 = packed_args.unpack<Tensor>();
  auto result2 = packed_args.unpack<Tensor>();
  return NpuMoeTokenUnpermuteWithRoutingMapBackward0_apply_functional(variable_list(grads), needs_input_grad, drop_and_pad, permuted_tokens, probs, restore_shape, routing_map, sorted_indices, result1, result2);
#endif
}

variable_list NpuMoeTokenUnpermuteWithRoutingMapBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto permuted_tokens = permuted_tokens_.unpack();
  auto probs = probs_.unpack();
  auto routing_map = routing_map_.unpack();
  auto sorted_indices = sorted_indices_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto permuted_tokens_ix = gen.range(1);
  auto probs_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 2>{
    task_should_compute_output({ permuted_tokens_ix }),
    task_should_compute_output({ probs_ix }),
  };
  return NpuMoeTokenUnpermuteWithRoutingMapBackward0_apply_functional(std::move(grads), needs_input_grad, drop_and_pad, permuted_tokens, probs, restore_shape, routing_map, sorted_indices, result1, result2);
}

void NpuMoeTokenUnpermuteWithRoutingMapBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(drop_and_pad);
    args.collect(permuted_tokens_, false);
    args.collect(probs_, false);
    args.collect(restore_shape);
    args.collect(routing_map_, false);
    args.collect(sorted_indices_, false);
    args.collect(result1_, true);
    args.collect(result2_, true);
}
variable_list NpuMoeTokenUnpermuteWithRoutingMapBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(drop_and_pad);
  saved.before(permuted_tokens_);
  saved.before(probs_);
  saved.before(restore_shape);
  saved.before(routing_map_);
  saved.before(sorted_indices_);
  saved.before(result1_);
  saved.before(result2_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 2>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<bool>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<std::vector<int64_t>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuMoeTokenUnpermuteWithRoutingMapBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto permuted_tokens = permuted_tokens_.unpack();
  auto probs = probs_.unpack();
  auto routing_map = routing_map_.unpack();
  auto sorted_indices = sorted_indices_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto permuted_tokens_ix = gen.range(1);
  auto probs_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 2>{
    task_should_compute_output({ permuted_tokens_ix }),
    task_should_compute_output({ probs_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(drop_and_pad);
  packed_args.pack(permuted_tokens);
  packed_args.pack(probs);
  packed_args.pack(restore_shape);
  packed_args.pack(routing_map);
  packed_args.pack(sorted_indices);
  packed_args.pack(result1);
  packed_args.pack(result2);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(drop_and_pad);
  saved.after(permuted_tokens_);
  saved.after(probs_);
  saved.after(restore_shape);
  saved.after(routing_map_);
  saved.after(sorted_indices_);
  saved.after(result1_);
  saved.after(result2_);
  return output_result;
#endif
}

static variable_list NpuMoeTokenPermuteBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,1> needs_input_grad, Tensor& indices, bool& padded_mode, Tensor& tokens, Tensor& result1)
{
  IndexRangeGenerator gen;
  auto tokens_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (needs_input_grad[/*tokens*/0]) {
    auto grad_result = any_grad_defined ? (npu_moe_token_permute_grad(tokens, grad, indices, result1, padded_mode)) : Tensor();
    copy_range(grad_inputs, tokens_ix, grad_result);
  }
  return grad_inputs;
}
inline variable_list NpuMoeTokenPermuteBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 1>>();
  auto indices = packed_args.unpack<Tensor>();
  auto padded_mode = packed_args.unpack<bool>();
  auto tokens = packed_args.unpack<Tensor>();
  auto result1 = packed_args.unpack<Tensor>();
  return NpuMoeTokenPermuteBackward0_apply_functional(variable_list(grads), needs_input_grad, indices, padded_mode, tokens, result1);
#endif
}

variable_list NpuMoeTokenPermuteBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto indices = indices_.unpack();
  auto tokens = tokens_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto tokens_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ tokens_ix }),
  };
  return NpuMoeTokenPermuteBackward0_apply_functional(std::move(grads), needs_input_grad, indices, padded_mode, tokens, result1);
}

void NpuMoeTokenPermuteBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(indices_, false);
    args.collect(padded_mode);
    args.collect(tokens_, false);
    args.collect(result1_, true);
}
variable_list NpuMoeTokenPermuteBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(indices_);
  saved.before(padded_mode);
  saved.before(tokens_);
  saved.before(result1_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 1>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<bool>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuMoeTokenPermuteBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto indices = indices_.unpack();
  auto tokens = tokens_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  IndexRangeGenerator gen;
  auto tokens_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 1>{
    task_should_compute_output({ tokens_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(indices);
  packed_args.pack(padded_mode);
  packed_args.pack(tokens);
  packed_args.pack(result1);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(indices_);
  saved.after(padded_mode);
  saved.after(tokens_);
  saved.after(result1_);
  return output_result;
#endif
}

static variable_list NpuMoeTokenUnpermuteBackward0_apply_functional(
  variable_list&& grads,
  std::array<bool,2> needs_input_grad, bool& padded_mode, Tensor& permuted_tokens, Tensor& probs, c10::OptionalArray<int64_t>& restore_shape, Tensor& sorted_indices)
{
  IndexRangeGenerator gen;
  auto permuted_tokens_ix = gen.range(1);
  auto probs_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  if (needs_input_grad[0] || needs_input_grad[1]) {

    auto grad_result = npu_moe_token_unpermute_grad(permuted_tokens, grad, sorted_indices, probs, padded_mode, restore_shape);
      if (needs_input_grad[/*permuted_tokens*/0]) {
        copy_range(grad_inputs, permuted_tokens_ix, std::get<0>(grad_result));
      }
      if (needs_input_grad[/*probs*/1]) {
        copy_range(grad_inputs, probs_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
inline variable_list NpuMoeTokenUnpermuteBackward0_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, 2>>();
  auto padded_mode = packed_args.unpack<bool>();
  auto permuted_tokens = packed_args.unpack<Tensor>();
  auto probs = packed_args.unpack<Tensor>();
  auto restore_shape = packed_args.unpack<c10::OptionalArray<int64_t>>();
  auto sorted_indices = packed_args.unpack<Tensor>();
  return NpuMoeTokenUnpermuteBackward0_apply_functional(variable_list(grads), needs_input_grad, padded_mode, permuted_tokens, probs, restore_shape, sorted_indices);
#endif
}

variable_list NpuMoeTokenUnpermuteBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto permuted_tokens = permuted_tokens_.unpack();
  auto probs = probs_.unpack();
  auto sorted_indices = sorted_indices_.unpack();
  IndexRangeGenerator gen;
  auto permuted_tokens_ix = gen.range(1);
  auto probs_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 2>{
    task_should_compute_output({ permuted_tokens_ix }),
    task_should_compute_output({ probs_ix }),
  };
  return NpuMoeTokenUnpermuteBackward0_apply_functional(std::move(grads), needs_input_grad, padded_mode, permuted_tokens, probs, restore_shape, sorted_indices);
}

void NpuMoeTokenUnpermuteBackward0::compiled_args(CompiledNodeArgs& args) const {
    args.collect(padded_mode);
    args.collect(permuted_tokens_, false);
    args.collect(probs_, false);
    args.collect(restore_shape);
    args.collect(sorted_indices_, false);
}
variable_list NpuMoeTokenUnpermuteBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  saved.before(padded_mode);
  saved.before(permuted_tokens_);
  saved.before(probs_);
  saved.before(restore_shape);
  saved.before(sorted_indices_);

  static bool called = false;
  if (!called) {
    called = true;
    std::vector<at::TypePtr> schema = {
      torch::dynamo::autograd::IValuePacker<std::array<bool, 2>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<bool>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
      torch::dynamo::autograd::IValuePacker<c10::OptionalArray<int64_t>>::packed_type(),
      torch::dynamo::autograd::IValuePacker<Tensor>::packed_type(),
    };
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), NpuMoeTokenUnpermuteBackward0_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;

  auto permuted_tokens = permuted_tokens_.unpack();
  auto probs = probs_.unpack();
  auto sorted_indices = sorted_indices_.unpack();
  IndexRangeGenerator gen;
  auto permuted_tokens_ix = gen.range(1);
  auto probs_ix = gen.range(1);
  auto needs_input_grad = std::array<bool, 2>{
    task_should_compute_output({ permuted_tokens_ix }),
    task_should_compute_output({ probs_ix }),
  };
  packed_args.pack(needs_input_grad);
  packed_args.pack(padded_mode);
  packed_args.pack(permuted_tokens);
  packed_args.pack(probs);
  packed_args.pack(restore_shape);
  packed_args.pack(sorted_indices);

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  saved.after(padded_mode);
  saved.after(permuted_tokens_);
  saved.after(probs_);
  saved.after(restore_shape);
  saved.after(sorted_indices_);
  return output_result;
#endif
}

}}} // namespace at_npu::autograd::generated
