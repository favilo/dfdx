use crate::prelude::webgpu_kernels::webgpu_unary;

const WGSL: &[u8] = b"TODO";

webgpu_unary!(super::CosKernelOp, f32, WGSL, "cos_fwd_f32", "cos_bwd_f32",);
