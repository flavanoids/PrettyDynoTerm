use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::*;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Uniforms {
    pub time: f32,
    pub frame: u32,
    pub width: u32,
    pub height: u32,
    pub cam_pos: [f32; 3],
    pub scene_id: u32,
    pub cam_dir: [f32; 3],
    pub benchmark_mode: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct DownsampleUniforms {
    pub src_width: u32,
    pub src_height: u32,
    pub dst_width: u32,
    pub dst_height: u32,
}

pub struct ComputePipelineManager {
    device: Arc<Device>,
    queue: Arc<Queue>,
    pipeline: ComputePipeline,
    bind_group: BindGroup,
    uniform_buffer: Buffer,
    storage_buffer: Buffer,
    staging_buffer: Buffer,
    pub width: u32,
    pub height: u32,

    // GPU-accelerated preview downsampling pipeline
    downsample_pipeline: ComputePipeline,
    downsample_bind_group_layout: BindGroupLayout,
    downsample_bind_group: BindGroup,
    downsample_uniform_buffer: Buffer,
    preview_storage_buffer: Buffer,
    preview_staging_buffer: Buffer,
    pub preview_width: u32,
    pub preview_height: u32,
}

impl ComputePipelineManager {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        shader_src: &str,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        Self::new_with_stats(device, queue, shader_src, width, height, None)
    }

    pub fn new_with_stats(
        device: Arc<Device>,
        queue: Arc<Queue>,
        shader_src: &str,
        width: u32,
        height: u32,
        stats: Option<&crate::telemetry::HardwareStats>,
    ) -> Result<Self, String> {
        // Pre-Flight VRAM & Buffer Limit Guard: validate against device limits and available VRAM headroom before buffer creation
        let vram_check =
            crate::benchmark::failsafe::VramHeadroomCheck::evaluate(&device, width, height, stats)?;
        if let Some(ref warn) = vram_check.warning {
            eprintln!(
                "\x1b[38;2;255;180;0;1m⚠ VRAM Headroom Warning:\x1b[0m {}",
                warn
            );
        }

        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("dyno_term_compute_shader"),
            source: ShaderSource::Wgsl(shader_src.into()),
        });

        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pixel_count = (width as u64) * (height as u64);
        let buffer_size = pixel_count.saturating_mul(4); // 4 bytes per pixel (RGBA8)

        let storage_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Storage Buffer (Pixels)"),
            size: buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Staging Readback Buffer"),
            size: buffer_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout"),
            entries: &[
                // Uniforms binding 0
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output pixel buffer binding 1
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: storage_buffer.as_entire_binding(),
                },
            ],
        });

        // Initialize Downsample Pipeline
        let downsample_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("dyno_term_downsample_shader"),
            source: ShaderSource::Wgsl(crate::shaders::DOWNSAMPLE_SHADER.into()),
        });

        let downsample_uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Downsample Uniform Buffer"),
            size: std::mem::size_of::<DownsampleUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let downsample_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Downsample Bind Group Layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let downsample_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Downsample Pipeline Layout"),
            bind_group_layouts: &[Some(&downsample_bind_group_layout)],
            immediate_size: 0,
        });

        let downsample_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Downsample Pipeline"),
            layout: Some(&downsample_pipeline_layout),
            module: &downsample_shader,
            entry_point: Some("downsample_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        let preview_width = 160u32;
        let preview_height = 100u32;
        let preview_buffer_size = (preview_width * preview_height * 4) as u64;

        let preview_storage_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Preview Storage Buffer"),
            size: preview_buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let preview_staging_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Preview Staging Buffer"),
            size: preview_buffer_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let downsample_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Downsample Bind Group"),
            layout: &downsample_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: downsample_uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: storage_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: preview_storage_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group,
            uniform_buffer,
            storage_buffer,
            staging_buffer,
            width,
            height,
            downsample_pipeline,
            downsample_bind_group_layout,
            downsample_bind_group,
            downsample_uniform_buffer,
            preview_storage_buffer,
            preview_staging_buffer,
            preview_width,
            preview_height,
        })
    }

    pub fn update_preview_buffers(&mut self, dst_width: u32, dst_height: u32) {
        let dst_width = dst_width.clamp(1, 16384);
        let dst_height = dst_height.clamp(1, 16384);
        if self.preview_width == dst_width && self.preview_height == dst_height {
            return;
        }

        let preview_size = match (dst_width as u64)
            .checked_mul(dst_height as u64)
            .and_then(|wh| wh.checked_mul(4))
        {
            Some(s) => s,
            None => return,
        };

        if preview_size > self.device.limits().max_buffer_size {
            return;
        }

        self.preview_width = dst_width;
        self.preview_height = dst_height;

        self.preview_storage_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Preview Storage Buffer"),
            size: preview_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        self.preview_staging_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Preview Staging Buffer"),
            size: preview_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.downsample_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Downsample Bind Group"),
            layout: &self.downsample_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.downsample_uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: self.storage_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: self.preview_storage_buffer.as_entire_binding(),
                },
            ],
        });
    }

    /// Safely attempts to resize the compute pipeline, validating against VRAM and hardware limits.
    /// Returns Ok(()) if successful, or Err if requested dimensions violate limits or exceed VRAM.
    pub fn try_resize(&mut self, new_width: u32, new_height: u32) -> Result<(), String> {
        if self.width == new_width && self.height == new_height {
            return Ok(());
        }

        // Validate new dimensions through pre-flight VRAM and OOB headroom guard
        crate::benchmark::failsafe::VramHeadroomCheck::evaluate(
            &self.device,
            new_width,
            new_height,
            None,
        )?;

        let buffer_size = (new_width as u64)
            .checked_mul(new_height as u64)
            .and_then(|wh| wh.checked_mul(4))
            .ok_or_else(|| "Integer overflow in buffer size calculation".to_string())?;

        let new_storage_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Storage Buffer (Pixels)"),
            size: buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let new_staging_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Staging Readback Buffer"),
            size: buffer_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = self.pipeline.get_bind_group_layout(0);
        let new_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Compute Bind Group (Resized)"),
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: new_storage_buffer.as_entire_binding(),
                },
            ],
        });

        let new_downsample_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Downsample Bind Group (Resized)"),
            layout: &self.downsample_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.downsample_uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: new_storage_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: self.preview_storage_buffer.as_entire_binding(),
                },
            ],
        });

        // Commit newly allocated resources safely
        self.width = new_width;
        self.height = new_height;
        self.storage_buffer = new_storage_buffer;
        self.staging_buffer = new_staging_buffer;
        self.bind_group = new_bind_group;
        self.downsample_bind_group = new_downsample_bind_group;

        Ok(())
    }

    #[allow(dead_code)]
    pub fn resize(&mut self, new_width: u32, new_height: u32) {
        if let Err(err) = self.try_resize(new_width, new_height) {
            eprintln!(
                "\x1b[38;2;255;100;100;1m⚠ Resize rejected by VRAM/OOB failsafe:\x1b[0m {}",
                err
            );
        }
    }

    /// Renders frame on GPU, downsamples directly in VRAM to preview resolution,
    /// and copies ONLY the tiny terminal preview buffer over PCIe into dst_buf.
    pub fn render_with_preview(
        &mut self,
        uniforms: &Uniforms,
        dst_width: u32,
        dst_height: u32,
        dst_buf: &mut [u8],
    ) {
        if dst_width == 0 || dst_height == 0 || dst_buf.is_empty() {
            return;
        }

        self.update_preview_buffers(dst_width, dst_height);

        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(uniforms));

        let ds_uniforms = DownsampleUniforms {
            src_width: self.width,
            src_height: self.height,
            dst_width,
            dst_height,
        };
        self.queue.write_buffer(
            &self.downsample_uniform_buffer,
            0,
            bytemuck::bytes_of(&ds_uniforms),
        );

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Render With Preview Encoder"),
            });

        // 1. Dispatch main 16K compute pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Main Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            let workgroups_x = self.width.div_ceil(16);
            let workgroups_y = self.height.div_ceil(16);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // 2. Dispatch GPU downsample compute pass in VRAM
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Downsample Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.downsample_pipeline);
            compute_pass.set_bind_group(0, &self.downsample_bind_group, &[]);
            let workgroups_x = dst_width.div_ceil(16);
            let workgroups_y = dst_height.div_ceil(16);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // 3. Copy tiny downsampled buffer (~38 KB) to staging buffer
        let preview_size = (dst_width * dst_height * 4) as u64;
        encoder.copy_buffer_to_buffer(
            &self.preview_storage_buffer,
            0,
            &self.preview_staging_buffer,
            0,
            preview_size,
        );

        self.queue.submit(Some(encoder.finish()));

        // 4. Map preview staging buffer and copy directly into dst_buf (zero heap allocations)
        let buffer_slice = self.preview_staging_buffer.slice(..preview_size);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        self.device.poll(PollType::wait_indefinitely()).unwrap();
        rx.recv()
            .unwrap()
            .expect("Failed to map preview staging buffer");

        let mapped = buffer_slice
            .get_mapped_range()
            .expect("Failed to get mapped range");
        let copy_len = dst_buf.len().min(mapped.len());
        dst_buf[..copy_len].copy_from_slice(&mapped[..copy_len]);
        drop(mapped);
        self.preview_staging_buffer.unmap();
    }

    /// Pure GPU compute dispatch with hardware completion wait (no PCIe staging readback).
    pub fn render_compute_sync(&self, uniforms: &Uniforms) {
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(uniforms));

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Compute Sync Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            let workgroups_x = self.width.div_ceil(16);
            let workgroups_y = self.height.div_ceil(16);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        self.queue.submit(Some(encoder.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        self.queue.on_submitted_work_done(move || {
            let _ = tx.send(());
        });
        self.device.poll(PollType::wait_indefinitely()).unwrap();
        let _ = rx.recv();
    }

    /// Asynchronous pure GPU compute dispatch (maximizes queue depth for uncapped stress testing).
    #[allow(dead_code)]
    pub fn render_compute_async(&self, uniforms: &Uniforms) {
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(uniforms));

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Compute Async Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            let workgroups_x = self.width.div_ceil(16);
            let workgroups_y = self.height.div_ceil(16);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    /// Full-resolution CPU readback (retained for backward compatibility).
    #[allow(dead_code)]
    pub fn render_frame(&self, uniforms: &Uniforms) -> Vec<u8> {
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(uniforms));

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Compute Frame Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);

            let workgroups_x = self.width.div_ceil(16);
            let workgroups_y = self.height.div_ceil(16);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        let buffer_size = (self.width * self.height * 4) as u64;
        encoder.copy_buffer_to_buffer(
            &self.storage_buffer,
            0,
            &self.staging_buffer,
            0,
            buffer_size,
        );

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = self.staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();

        buffer_slice.map_async(MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        self.device.poll(PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().expect("Failed to map staging buffer");

        let mapped = buffer_slice
            .get_mapped_range()
            .expect("Failed to get mapped range");
        let pixel_bytes = mapped.to_vec();
        drop(mapped);
        self.staging_buffer.unmap();

        pixel_bytes
    }
}
