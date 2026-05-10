
// gpu.rs — GPU pipeline boilerplate
// wraps the wgpu surface/device/queue/config into one struct so lib.rs State doesn't have to carry them all separately.
// helper fns at the bottom (write_texture, uniform_bgl etc) 

use std::sync::Arc;
use wgpu::util::DeviceExt;  // for uniform_buf helper below
use winit::window::Window;

// GpuCtx - owns all the low-level gpu handles everything that touches wgpu goes through this
pub struct GpuCtx
{
    pub surface:       wgpu::Surface<'static>,
    pub device:        wgpu::Device,
    pub queue:         wgpu::Queue,
    pub config:        wgpu::SurfaceConfiguration,
    // set to true after the first resize(), guard against drawing before configure()
    pub is_configured: bool,
}

impl GpuCtx
{
    pub async fn new(window: Arc<Window>) -> Self
    {
        let size = window.inner_size();

        // wasm path for browser rendering
        #[cfg(target_arch = "wasm32")]
        let (surface, adapter) = {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends:                 wgpu::Backends::GL,
                flags:                    wgpu::InstanceFlags::default(),
                memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
                backend_options:          wgpu::BackendOptions::default(),
                display:                  None,
            });
            let surface = instance.create_surface(window.clone()).unwrap();
            let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference:       wgpu::PowerPreference::HighPerformance,
                compatible_surface:     Some(&surface),
                force_fallback_adapter: false,
            }).await.expect("no webgl2 adapter found");
            (surface, adapter)
        };

        // native: try vulkan first, fall back to GL
        #[cfg(not(target_arch = "wasm32"))]
        let (surface, adapter) = {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends:                 wgpu::Backends::VULKAN,
                flags:                    wgpu::InstanceFlags::default(),
                memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
                backend_options:          wgpu::BackendOptions::default(),
                display:                  None,  // wgpu grabs from window at create_surface
            });
            let surface = instance.create_surface(window.clone()).unwrap();
            match instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference:       wgpu::PowerPreference::HighPerformance,
                compatible_surface:     Some(&surface),
                force_fallback_adapter: false,
            }).await {
                Ok(a) => (surface, a),
                Err(_) => {
                    // vulkan not available -> go to GL llvmpipe should work on basically anything
                    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                        backends:                 wgpu::Backends::GL,
                        flags:                    wgpu::InstanceFlags::default(),
                        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
                        backend_options:          wgpu::BackendOptions::default(),
                        display:                  None,
                    });
                    let surface = instance.create_surface(window.clone()).unwrap();
                    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference:       wgpu::PowerPreference::HighPerformance,
                        compatible_surface:     Some(&surface),
                        force_fallback_adapter: false,
                    }).await.expect("no adapter found (tried vulkan then GL)");
                    (surface, adapter)
                }
            }
        };

        // device = logical gpu handle, queue = command submission channel
        // on wasm use webgl2 limits (max tex size 2048 etc)
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            label:             Some("Device"),
            required_features: wgpu::Features::empty(),
            required_limits:   if cfg!(target_arch = "wasm32") {
                wgpu::Limits::downlevel_webgl2_defaults()
            } else {
                wgpu::Limits::default()
            },
            ..Default::default()
        }).await.expect("device request failed");

        // pixel format +  other surface config
        let caps   = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().find(|f| f.is_srgb()).copied()
                     .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage:                         wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width:                         size.width.max(1),
            height:                        size.height.max(1),
            present_mode:                  wgpu::PresentMode::Fifo,  // vsync
            alpha_mode:                    caps.alpha_modes[0],
            view_formats:                  vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        Self { surface, device, queue, config, is_configured: true }
    }

    pub fn resize(&mut self, width: u32, height: u32)
    {
        if width > 0 && height > 0 {
            self.config.width  = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.is_configured = true;
        }
    }

    // surface texture format here
    pub fn format(&self) -> wgpu::TextureFormat { self.config.format }

    // aspect ratio for projection math
    pub fn aspect(&self) -> f32 { self.config.width as f32 / self.config.height.max(1) as f32 }
}

// texture upload helper
// copies RGBA8 pixel buffer into mip 0 of 2d gpu texture
// bytes.len() must = w * h * 4
pub fn write_texture(queue: &wgpu::Queue, tex: &wgpu::Texture, bytes: &[u8], w: u32, h: u32)
{
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture:   tex,
            mip_level: 0,
            origin:    wgpu::Origin3d::ZERO,
            aspect:    wgpu::TextureAspect::All,
        },
        bytes,
        wgpu::TexelCopyBufferLayout {
            offset:         0,
            bytes_per_row:  Some(w * 4),  // 4 bytes/pixel rgba
            rows_per_image: Some(h),
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
}

// uniformbuffer helper fns
// manages layout, buffer, bind groups
// creates a single-binding uniform BGL visible to the given shader stages
pub fn uniform_bgl(
    device: &wgpu::Device,
    label:  &str,
    vis:    wgpu::ShaderStages,
) -> wgpu::BindGroupLayout
{
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label:   Some(label),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding:    0,
            visibility: vis,
            ty: wgpu::BindingType::Buffer {
                ty:                 wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size:   None,
            },
            count: None,
        }],
    })
}

// allocates a UNIFORM | COPY_DST buffer pre-filled with data
pub fn uniform_buf(device: &wgpu::Device, data: &[u8]) -> wgpu::Buffer
{
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("Uniform Buffer"),
        contents: data,
        usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

// wires buf → binding 0 in layout
pub fn uniform_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buf:    &wgpu::Buffer,
) -> wgpu::BindGroup
{
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label:   Some("Uniform BG"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding:  0,
            resource: buf.as_entire_binding(),
        }],
    })
}
