use crate::prelude::webgpu_kernels::webgpu_unary;

const WGSL: &[u8] = b"TODO";

webgpu_unary!(super::LnKernelOp, f32, WGSL, "ln_fwd_f32", "ln_bwd_f32",);
