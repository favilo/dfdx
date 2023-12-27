use crate::prelude::webgpu_kernels::webgpu_unary;

const WGSL: &[u8] = b"TODO";

webgpu_unary!(
    super::ClampKernelOp<f32>,
    f32,
    WGSL,
    "clamp_fwd_f32",
    "clamp_bwd_f32",
);
