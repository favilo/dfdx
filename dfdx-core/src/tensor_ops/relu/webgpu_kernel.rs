use crate::prelude::webgpu_kernels::webgpu_unary;

const WGSL: &[u8] = b"TODO";

webgpu_unary!(
    super::ReLUKernelOp,
    f32,
    WGSL,
    "relu_fwd_f32",
    "relu_bwd_f32",
);
