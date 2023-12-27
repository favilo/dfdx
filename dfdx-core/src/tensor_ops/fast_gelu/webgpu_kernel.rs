use crate::prelude::webgpu_kernels::webgpu_unary;

const WGSL: &[u8] = b"TODO";

webgpu_unary!(
    super::FastGeLUKernelOp,
    f32,
    WGSL,
    "fast_gelu_fwd_f32",
    "fast_gelu_bwd_f32",
);
