#pragma once

#include <string>
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/triton_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
class FlashAttentionKernel final : public onnxruntime::cuda::CudaKernel {
 public:
  FlashAttentionKernel(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  bool causal_;
  float softmax_scale_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
