#![feature(asm_experimental_arch)]
#![cfg_attr(target_arch = "spirv", no_std)]
// HACK(eddyb) can't easily see warnings otherwise from `spirv-builder` builds.
#![deny(warnings)]

use glam::UVec3;

extern crate spirv_std;
use spirv_std::{glam, num_traits::Float, spirv};

fn abs<E>(idx: usize, input: &[E], output: &mut [E])
where
    E: spirv_std::num_traits::Float,
{
    if input.len() > 0 {
        output[idx] = input[idx].abs();
    } else {
        output[idx] = output[idx].abs();
    }
}

#[spirv(compute(threads(128)))]
pub fn abs_fwd_f32(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &mut [f32],
) {
    let index = id.x as usize;
    abs(index, input, output);
}

#[spirv(compute(threads(128)))]
pub fn abs_bwd_f32(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] _output: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] inp_grad: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] out_grad: &[f32],
) {
    let index = id.x as usize;
    let dx = input[index].signum();
    inp_grad[index] += out_grad[index] * dx;
}

// #[spirv(compute(threads(128)))]
// pub fn abs_fwd_f64(
//     #[spirv(global_invocation_id)] id: UVec3,
//     #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] input: &[f64],
//     #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &mut [f64],
// ) {
//     unsafe {
//         spirv_std::asm! {
//             "OpCapability Float64"
//         }
//     };
//     let index = id.x as usize;
//     abs(index, input, output);
// }
