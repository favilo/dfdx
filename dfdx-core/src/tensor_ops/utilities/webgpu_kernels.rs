use crate::{
    shapes::{Dtype, Shape},
    tensor::*,
    tensor_ops::ops::{BinaryKernel, UnaryKernel},
};
use core::any::TypeId;
use std::{borrow::Cow, marker::PhantomData, sync::Arc, vec::Vec};

pub(crate) trait UnaryOpWebgpuKernel<E> {
    const DF_USES_FX: bool;
    const HAS_CONST_DF: bool;

    // /// Unique name for the kernel
    // const MODULE_NAME: &'static str;

    /// SpirV source code
    const SPV_SOURCE: &'static [u8];

    /// Name of the forward function
    const FWD_FN_NAME: &'static str;

    /// Name of the backward function
    const BWD_FN_NAME: &'static str;
}

macro_rules! webgpu_unary {
    ($Op:path, $TypeName:ty, $SPV:tt, $Fwd:tt, $Bwd:tt$(,)?) => {
        impl crate::tensor_ops::webgpu_kernels::UnaryOpWebgpuKernel<$TypeName> for $Op {
            const DF_USES_FX: bool = false;
            const HAS_CONST_DF: bool = false;
            // const MODULE_NAME: &'static str = stringify!($Op);
            const SPV_SOURCE: &'static [u8] = $SPV;
            const FWD_FN_NAME: &'static str = $Fwd;
            const BWD_FN_NAME: &'static str = $Bwd;
        }
    };
    (df(f(x)) $Op:path, $TypeName:ty, $SPV:tt, $Fwd:tt, $Bwd:tt$(,)?) => {
        impl crate::tensor_ops::webgpu_kernels::UnaryOpWebgpuKernel<$TypeName> for $Op {
            const DF_USES_FX: bool = true;
            const HAS_CONST_DF: bool = false;
            // const MODULE_NAME: &'static str = $Fwd;
            const SPV_SOURCE: &'static [u8] = $SPV;
            const FWD_FN_NAME: &'static str = $Fwd;
            const BWD_FN_NAME: &'static str = $Bwd;
        }
    };
    (const_df() $Op:path, $TypeName:ty,$SPV:tt, $Fwd:tt, $Bwd:tt$(,)?) => {
        impl crate::tensor_ops::webgpu_kernels::UnaryOpWebgpuKernel<$TypeName> for $Op {
            const DF_USES_FX: bool = false;
            const HAS_CONST_DF: bool = true;
            // const MODULE_NAME: &'static str = $Fwd;
            const SPV_SOURCE: &'static [u8] = $SPV;
            const FWD_FN_NAME: &'static str = $Fwd;
            const BWD_FN_NAME: &'static str = $Bwd;
        }
    };
}

pub(crate) use webgpu_unary;
use wgpu::{BindingType, ComputePipelineDescriptor, ShaderStages};

impl<E: Dtype, K: UnaryOpWebgpuKernel<E> + 'static> UnaryKernel<K, E> for Webgpu {
    const BACKWARD_WITHOUT_INP: bool = K::DF_USES_FX;
    const BACKWARD_WITHOUT_DATA: bool = K::HAS_CONST_DF;

    fn forward<S: Shape>(
        &self,
        op: K,
        inp: Cow<Tensor<S, E, Self>>,
    ) -> Result<Tensor<S, E, Self>, Error> {
        if !self.shader_module_loaded(TypeId::of::<K>()) {
            self.load_shader_module::<E>(TypeId::of::<K>(), K::SPV_SOURCE);
        }

        let cs_module = self
            .get_shader_module(TypeId::of::<K>())
            .ok_or(Error::WebgpuSourceLoadError)?;
        let bind_group_layout = self.create_bind_group_layout_fwd::<E, K>();
        let pipeline = self.create_pipeline::<E, K>(bind_group_layout, cs_module);
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let op_storage = self.alloc_init::<K>(&[op])?;
        let numel = inp.data.len::<E>();
        let num_blocks = (numel + 128 - 1) / 128;
        let storage = self.alloc_empty::<E>(numel)?;
        let empty = self.alloc_empty::<E>(0)?;
        let mut entries = Vec::new();
        // WGSL doesn't support empty structs, so don't bind the empty buffer
        if std::mem::size_of::<K>() > 0 {
            entries.push(wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(op_storage.as_entire_buffer_binding()),
            });
        }

        match inp {
            Cow::Borrowed(inp) => {
                entries.push(wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(inp.data.as_entire_buffer_binding()),
                });
                entries.push(wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(storage.as_entire_buffer_binding()),
                });
                let binding_group = self.dev.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &bind_group_layout,
                    entries: &entries,
                });
                let mut encoder = self
                    .dev
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&pipeline);
                    cpass.set_bind_group(0, &binding_group, &[]);
                    cpass.dispatch_workgroups(num_blocks as u32, 1, 1);
                }
                self.queue.submit(Some(encoder.finish()));
                Ok(self.build_tensor(inp.shape, inp.strides, storage))
            }
            Cow::Owned(mut inp) => {
                entries.push(wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(empty.as_entire_buffer_binding()),
                });
                entries.push(wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(
                        Arc::make_mut(&mut inp.data).as_entire_buffer_binding(),
                    ),
                });
                let binding_group = self.dev.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &bind_group_layout,
                    entries: &entries,
                });
                let mut encoder = self
                    .dev
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&pipeline);
                    cpass.set_bind_group(0, &binding_group, &[]);
                    cpass.dispatch_workgroups(num_blocks as u32, 1, 1);
                }
                self.queue.submit(Some(encoder.finish()));
                Ok(inp)
            }
        }
    }

    fn backward<S: Shape>(
        &self,
        op: K,
        inp: &impl Tensorlike<S, E, Self>,
        grad_inp: &mut Self::Vec,
        out: &impl Tensorlike<S, E, Self>,
        grad_out: &Self::Vec,
    ) -> Result<(), Error> {
        if !self.shader_module_loaded(TypeId::of::<K>()) {
            self.load_shader_module::<E>(TypeId::of::<K>(), K::SPV_SOURCE);
        }

        let cs_module = self
            .get_shader_module(TypeId::of::<K>())
            .ok_or(Error::WebgpuSourceLoadError)?;
        let bind_group_layout = self.create_bind_group_layout_bwd::<E, K>();
        let pipeline = self.create_pipeline::<E, K>(bind_group_layout, cs_module);
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let op_storage = self.alloc_init::<K>(&[op])?;
        let numel = inp.len();
        let empty_inp = self.alloc_empty::<E>(0)?;
        let empty_out = self.alloc_empty::<E>(0)?;
        let mut entries = Vec::new();
        // WGSL doesn't support empty structs, so don't bind the empty buffer
        if std::mem::size_of::<K>() > 0 {
            entries.push(wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(op_storage.as_entire_buffer_binding()),
            });
        }
        match (inp.data(), out.data()) {
            (None, None) => {
                entries.push(wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(empty_inp.as_entire_buffer_binding()),
                });
                entries.push(wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(empty_out.as_entire_buffer_binding()),
                });
                entries.push(wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(grad_inp.as_entire_buffer_binding()),
                });
                entries.push(wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(grad_out.as_entire_buffer_binding()),
                });
            }
            (None, Some(out)) => {
                entries.push(wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(empty_inp.as_entire_buffer_binding()),
                });
                entries.push(wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(out.as_entire_buffer_binding()),
                });
                entries.push(wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(grad_inp.as_entire_buffer_binding()),
                });
                entries.push(wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(grad_out.as_entire_buffer_binding()),
                });
            }
            (Some(inp), None) => {
                entries.push(wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(inp.as_entire_buffer_binding()),
                });
                entries.push(wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(empty_out.as_entire_buffer_binding()),
                });
                entries.push(wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(grad_inp.as_entire_buffer_binding()),
                });
                entries.push(wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(grad_out.as_entire_buffer_binding()),
                });
            }
            _ => unreachable!(),
        };
        let binding_group = self.dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &entries,
        });
        let mut encoder = self
            .dev
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &binding_group, &[]);
            cpass.dispatch_workgroups(numel as u32, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }
}

impl Webgpu {
    fn create_bind_group_layout_fwd<E: Dtype, K: UnaryOpWebgpuKernel<E> + 'static>(
        &self,
    ) -> wgpu::BindGroupLayout {
        let mut entries = Vec::new();
        if std::mem::size_of::<K>() > 0 {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    has_dynamic_offset: false,
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    min_binding_size: None,
                },
                count: None,
            });
        }
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                has_dynamic_offset: false,
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                min_binding_size: None,
            },
            count: None,
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                has_dynamic_offset: false,
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                min_binding_size: None,
            },
            count: None,
        });

        let bind_group_layout =
            self.dev
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &entries,
                });
        bind_group_layout
    }

    fn create_bind_group_layout_bwd<E: Dtype, K: UnaryOpWebgpuKernel<E> + 'static>(
        &self,
    ) -> wgpu::BindGroupLayout {
        let mut entries = Vec::new();
        if std::mem::size_of::<K>() > 0 {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    has_dynamic_offset: false,
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    min_binding_size: None,
                },
                count: None,
            });
        }
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                has_dynamic_offset: false,
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                min_binding_size: None,
            },
            count: None,
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                has_dynamic_offset: false,
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                min_binding_size: None,
            },
            count: None,
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                has_dynamic_offset: false,
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                min_binding_size: None,
            },
            count: None,
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                has_dynamic_offset: false,
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                min_binding_size: None,
            },
            count: None,
        });

        let bind_group_layout =
            self.dev
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &entries,
                });
        bind_group_layout
    }

    fn create_pipeline<E: Dtype, K: UnaryOpWebgpuKernel<E> + 'static>(
        &self,
        bind_group_layout: wgpu::BindGroupLayout,
        cs_module: Arc<wgpu::ShaderModule>,
    ) -> wgpu::ComputePipeline {
        let pipeline_layout = self
            .dev
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
        let pipeline_desc = ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &cs_module,
            entry_point: K::FWD_FN_NAME,
        };
        let pipeline = self.dev.create_compute_pipeline(&pipeline_desc);
        pipeline
    }
}
