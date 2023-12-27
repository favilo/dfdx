use crate::prelude::webgpu_kernels::webgpu_unary;

const WGSL: &[u8] = b"TODO";

webgpu_unary!(df(f(x))
    super::SigmoidKernelOp,
    f32,
    WGSL,
    "sigmoid_fwd_f32",
    "sigmoid_bwd_f32",
);
