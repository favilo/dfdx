use crate::prelude::webgpu_kernels::webgpu_unary;

const WGSL: &[u8] = b"TODO";

webgpu_unary!(df(f(x))
    super::SqrtKernelOp,
    f32,
    WGSL,
    "sqrt_fwd_f32",
    "sqrt_bwd_f32",
);
