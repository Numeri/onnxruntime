#include "contrib_ops/cuda/flash_attention_triton.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

template <typename T>
std::string GetTypeAsString();

#define TYPE_TO_STRING(T, S)          \
  template <>                         \
  std::string GetTypeAsString<T>() {  \
    return S;                         \
  }

TYPE_TO_STRING(MLFloat16, "fp16");
TYPE_TO_STRING(float, "fp32");

template <typename T>
std::string GetFlashAttentionFunctionName() {
  std::string ret = "flash_attention_kernel_";
  ret += GetTypeAsString<T>();
  return ret;
}

}  // end of namespace

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      FlashAttentionKernel,                                       \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      FlashAttentionKernel<T>);

REGISTER_KERNEL_TYPED(MLFloat16);
REGISTER_KERNEL_TYPED(float);

template <typename T>
FlashAttentionKernel<T>::FlashAttentionKernel(const OpKernelInfo& info) : CudaKernel{info} {
  causal_ = static_cast<bool>(info.GetAttrOrDefault<int>("causal", 0));
  softmax_scale_ = info.GetAttrOrDefault<float>("softmax_scale", 1.0f);
}

template <typename T>
Status FlashAttentionKernel<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* Q = ctx->Input<Tensor>(0);
  const Tensor* K = ctx->Input<Tensor>(1);
  const Tensor* V = ctx->Input<Tensor>(2);
  const Tensor* Bias = ctx->Input<Tensor>(3);

  ORT_RETURN_IF_NOT(Q != nullptr, "Input tensor Q is null");
  ORT_RETURN_IF_NOT(K != nullptr, "Input tensor K is null");
  ORT_RETURN_IF_NOT(V != nullptr, "Input tensor V is null");

  const TensorShape& Q_shape = Q->Shape();
  const TensorShape& K_shape = K->Shape();
  const TensorShape& V_shape = V->Shape();

  ORT_RETURN_IF_NOT(Q_shape.NumDimensions() == 4, "Input tensor Q must have 4 dimensions");
  ORT_RETURN_IF_NOT(K_shape.NumDimensions() == 4, "Input tensor K must have 4 dimensions");
  ORT_RETURN_IF_NOT(V_shape.NumDimensions() == 4, "Input tensor V must have 4 dimensions");

  Tensor* Output = ctx->Output(0, Q_shape);
  ORT_RETURN_IF_NOT(Output != nullptr, "Failed to create output tensor");

  std::string function_name = GetFlashAttentionFunctionName<T>();
  cudaStream_t stream = Stream(ctx);

  struct KernelArgs {
    T* output_ptr;
    const T* q_ptr;
    const T* k_ptr;
    const T* v_ptr;
    const T* bias_ptr;
    int32_t batch;
    int32_t seqlen_q;
    int32_t seqlen_k;
    int32_t nheads;
    int32_t d;
    bool causal;
    float softmax_scale;
  } args = {
    Output->MutableData<T>(),
    Q->Data<T>(),
    K->Data<T>(),
    V->Data<T>(),
    Bias ? Bias->Data<T>() : nullptr,
    static_cast<int32_t>(Q_shape[0]),
    static_cast<int32_t>(Q_shape[1]),
    static_cast<int32_t>(K_shape[1]),
    static_cast<int32_t>(Q_shape[2]),
    static_cast<int32_t>(Q_shape[3]),
    causal_,
    softmax_scale_
  };

  int64_t grid_size = (Q_shape[1] + 127) / 128;  // Assuming BLOCK size is 128
  return onnxruntime::cuda::LaunchTritonKernel(stream, function_name, grid_size, 1, 1, &args, sizeof(args));
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
