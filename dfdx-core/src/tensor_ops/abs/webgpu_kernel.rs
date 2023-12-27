use super::AbsKernelOp;
use crate::tensor_ops::webgpu_kernels::webgpu_unary;

const SPV_FWD: &[u8] = include_bytes!(env!("abs.spv"));

webgpu_unary!(AbsKernelOp, f32, SPV_FWD, "abs_fwd_f32", "abs_bwd_f32");

#[cfg(test)]
mod tests {
    use crate::{prelude::*, tensor::*, tests::*};

    #[test]
    fn test_webgpu_abs() {
        let dev: Webgpu = Default::default();
        let x = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.leaky_trace().abs();
        assert_close_to_literal!(r, [2.0, 1.0, 0.0, 1.0, 2.0]);
        // TODO: Add mean back in
        // let g = r.mean().backward();
        // assert_close_to_literal!(g.get(&x), [-0.2, -0.2, 0.0, 0.2, 0.2]);
    }
}
