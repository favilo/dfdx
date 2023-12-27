use crate::prelude::webgpu_kernels::webgpu_unary;

const WGSL: &[u8] = b"TODO";

webgpu_unary!(
    super::SquareKernelOp,
    f32,
    WGSL,
    "square_fwd_f32",
    "square_bwd_f32",
);
