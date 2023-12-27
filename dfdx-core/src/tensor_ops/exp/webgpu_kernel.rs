use crate::prelude::webgpu_kernels::webgpu_unary;

const WGSL: &[u8] = b"TODO";

webgpu_unary!(super::ExpKernelOp, f32, WGSL, "exp_fwd_f32", "exp_bwd_f32",);
