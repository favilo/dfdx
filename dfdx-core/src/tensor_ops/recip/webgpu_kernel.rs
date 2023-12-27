use crate::prelude::webgpu_kernels::webgpu_unary;

const WGSL: &[u8] = b"TODO";

webgpu_unary!(df(f(x))
    super::RecipKernelOp,
    f32,
    WGSL,
    "recip_fwd_f32",
    "recip_bwd_f32",
);
