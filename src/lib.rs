// CMSC427 Lab 02 — Parametric Objects

// draws procedurally generated 3d meshes under an orbiting camera
// press 1-4 to switch mesh, +/- to scale
// 1 — Barn (pentagonal prism)
// 2 — Cone (n-sided lathed circle)
// 3 — Vase (sine-profile lathe surface)
// 4 — Disc (flat polar grid)
// + or =  for scale up, — for scale down 

// code added in lab2 vs lab1
// 3d vertex positions (x,y,z) instead of 2d NDC coords
// index buffers — GPU reuses shared corner verts, cheaper than repeating them
// MVP matrix via uniform buffer — vertex shader transforms each vert from object space to clip space in one matrix multiply per frame
// model matrix — separate from view/proj so we can scale/rotate without touching the vert buffer (64 bytes uniform vs ~768 bytes of vert data)
// continuous animation — camera orbits, no input needed to see all sides

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
fn set_status(msg: &str) {
    if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
        if let Some(el) = doc.get_element_by_id("status") {
            el.set_text_content(Some(msg));
        }
    }
}

pub mod gpu;
pub mod mesh;

use std::sync::Arc;
use web_time::Instant;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};
use gpu::GpuCtx;
use mesh::{GpuMesh, Vertex};

// MVP uniform = Model × View × Projection, combined into one 4x4 matrix
// Model: object space → world space. + uniform scale controlled by +/- keys

// View: world space → camera space. computed with Mat4::look_at_rh(eye, target, up)
// Projection: camera space → clip space (NDC after divide by w)
// Mat4::perspective_rh(fov_y, aspect, near, far)

// vert shader does: clip_pos = mvp * vec4(position, 1.0)
// the w=1=point
// normals/directions use w=0

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms { mvp: [[f32; 4]; 4] }

// WGSL shader
// one uniform bind group: @group(0) @binding(0) — MVP matrix, vertex stage only
// frag shader just passes colour through
const SHADER: &str = r#"
// Uniforms layout must match the Rust struct byte-for-byte
// mat4x4<f32> = 64 bytes = 4 columns × 4 rows × 4 bytes, matches [[f32;4];4]
struct Uniforms { mvp: mat4x4<f32> };
@group(0) @binding(0) var<uniform> u: Uniforms;

// VIn: inputs from the vertex buffer — must match Vertex::layout() attr locations
struct VIn  { @location(0) position: vec3<f32>, @location(1) color: vec3<f32> };
// VOut: data passed from vertex stage to fragment stage, interpolated per pixel
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) color: vec3<f32> };

@vertex
fn vs_main(in: VIn) -> VOut {
    var out: VOut;
    // MVP transform: moves vert from object space → clip space in one multiply
    // w=1.0 makes position a point (translations from model/view apply)
    out.clip  = u.mvp * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    // just output the interpolated vertex colour, fully opaque
    return vec4<f32>(in.color, 1.0);
}
"#;

// make_depth_view — creates a Depth32Float texture sized to the given dimensions and returns a view
// called at startup and again on every resize, depth texture must match swap chain dimensions exactly
fn make_depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView
{
    device.create_texture(&wgpu::TextureDescriptor {
        label:           Some("Depth"),
        size:            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count:    1,
        dimension:       wgpu::TextureDimension::D2,
        format:          wgpu::TextureFormat::Depth32Float,
        usage:           wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats:    &[],
    }).create_view(&wgpu::TextureViewDescriptor::default())
}

// State - owns all GPU resources + frame animation state
struct State
{
    window:       Arc<Window>,
    gpu:          GpuCtx,
    pipeline:     wgpu::RenderPipeline,
    uniform_buf:  wgpu::Buffer,
    bind_group:   wgpu::BindGroup,
    depth_view:   wgpu::TextureView,
    // all four meshes are uploaded at startup — switching just changes current_mesh
    meshes:       Vec<GpuMesh>,
    current_mesh: usize,
    scale:        f32,   // uniform scaling thru model matrix
    start_time:   Instant,
}

impl State
{
    async fn new(window: Arc<Window>) -> Self
    {
        let gpu = GpuCtx::new(window.clone()).await;

        // bind group layout declares what the shader expects at group(0)
        let bgl = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("Uniform BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding:    0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count: None,
            }],
        });

        // pre-fill with identity MVP
        let uniform_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("Uniform Buffer"),
            contents: bytemuck::bytes_of(&Uniforms { mvp: Mat4::IDENTITY.to_cols_array_2d() }),
            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // bind group links the buffer to the layout
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("Uniform BG"),
            layout:  &bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() }],
        });

        // compile + link shaders
        let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        // pipeline layout tells wgpu which bind group slots this pipeline uses, i.e shader and processing order
        let pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:              Some("Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size:     0,
        });
        let pipeline = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module:              &shader,
                entry_point:         Some("vs_main"),
                buffers:             &[Vertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:      &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format:     gpu.format(),
                    blend:      Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology:  wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format:              wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare:       Some(wgpu::CompareFunction::Less),
                stencil:             wgpu::StencilState::default(),
                bias:                wgpu::DepthBiasState::default(),
            }),
            multisample:    wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache:          None,
        });

        // generate all four meshes on CPU and upload to GPU once at startup
        // switching via changing ibuf/vbuf being bound to pipeline
        let (bv, bi) = mesh::make_barn();
        let (cv, ci) = mesh::make_cone(32);
        let (vv, vi) = mesh::make_vase(24, 32);
        let (dv, di) = mesh::make_disc(8, 48);
        let meshes = vec![
            GpuMesh::upload(&gpu.device, &bv, &bi),
            GpuMesh::upload(&gpu.device, &cv, &ci),
            GpuMesh::upload(&gpu.device, &vv, &vi),
            GpuMesh::upload(&gpu.device, &dv, &di),
        ];

        // depth texture starts at the initial window size; recreated in resize()
        let depth_view = make_depth_view(&gpu.device, gpu.config.width, gpu.config.height);

        Self {
            window,
            gpu,
            pipeline,
            uniform_buf,
            bind_group,
            depth_view,
            meshes,
            current_mesh: 0,
            scale: 1.0,
            start_time: Instant::now(),
        }
    }

    fn resize(&mut self, width: u32, height: u32)
    {
        self.gpu.resize(width, height);
        self.depth_view = make_depth_view(&self.gpu.device, width, height);
    }

    fn set_shape(&mut self, idx: usize)
    {
        self.current_mesh = idx.min(self.meshes.len() - 1);
    }

    // multiply current scale by delta
    fn set_scale(&mut self, delta: f32)
    {
        self.scale = (self.scale * delta).clamp(0.1, 10.0);
    }

    fn render(&mut self) -> Result<(), ()>
    {
        self.window.request_redraw();

        // build MVP: Projection * View * Model
        let t     = self.start_time.elapsed().as_secs_f32();
        let angle = t * 0.6;
        let eye   = Vec3::new(2.5 * angle.cos(), 1.2, 2.5 * angle.sin());
        let proj  = Mat4::perspective_rh(45_f32.to_radians(), self.gpu.aspect(), 0.1, 100.0);
        let view  = Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y);
        let model = Mat4::from_scale(Vec3::splat(self.scale));  // uniform scale from +/- keys
        let mvp   = proj * view * model;

        // overwrite the ubuf with new matrix
        self.gpu.queue.write_buffer(
            &self.uniform_buf, 0,
            bytemuck::bytes_of(&Uniforms { mvp: mvp.to_cols_array_2d() }),
        );

        // get the next swap chain img to draw into, texturing code, just boilerplate for now
        let output = match self.gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(o)
            | wgpu::CurrentSurfaceTexture::Suboptimal(o) => o,
            wgpu::CurrentSurfaceTexture::Outdated
            | wgpu::CurrentSurfaceTexture::Lost => {
                self.gpu.surface.configure(&self.gpu.device, &self.gpu.config);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded => return Ok(()),
            wgpu::CurrentSurfaceTexture::Validation => return Err(()),
        };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Encoder") }
        );

        {
            let mut rpass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view:           &view,
                    depth_slice:    None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(wgpu::Color { r: 0.05, g: 0.06, b: 0.10, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view:         &self.depth_view,
                    depth_ops:    Some(wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(1.0),  // 1.0 = far plane, everything passes initially
                        store: wgpu::StoreOp::Discard,    // depth not needed after this pass
                    }),
                    stencil_ops:  None,
                }),
                occlusion_query_set:      None,
                timestamp_writes:         None,
                multiview_mask:           None,
            });

            // drawing triangles
            let mesh = &self.meshes[self.current_mesh];
            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.set_vertex_buffer(0, mesh.vbuf.slice(..));
            rpass.set_index_buffer(mesh.ibuf.slice(..), wgpu::IndexFormat::Uint16);
            rpass.draw_indexed(0..mesh.index_count, 0, 0..1);
        }

        self.gpu.queue.submit(std::iter::once(enc.finish()));
        output.present();
        Ok(())
    }
}

// App (OS event handling)
// native: State lives directly in App as Option<State>
// wasm: State lives in Rc<RefCell<Option<State>>> 
// with_state() abstracts the platform difference behind a closure, this is why rust is the goat

pub struct App
{
    window: Option<Arc<Window>>,
    #[cfg(not(target_arch = "wasm32"))]
    state: Option<State>,
    #[cfg(target_arch = "wasm32")]
    state: std::rc::Rc<std::cell::RefCell<Option<State>>>,
}

impl App
{
    pub fn new() -> Self
    {
        Self {
            window: None,
            #[cfg(not(target_arch = "wasm32"))]
            state: None,
            #[cfg(target_arch = "wasm32")]
            state: std::rc::Rc::new(std::cell::RefCell::new(None)),
        }
    }

    fn with_state<R>(&mut self, f: impl FnOnce(&mut State) -> R) -> Option<R>
    {
        #[cfg(not(target_arch = "wasm32"))]
        { self.state.as_mut().map(f) }
        #[cfg(target_arch = "wasm32")]
        { self.state.borrow_mut().as_mut().map(f) }
    }
}

impl ApplicationHandler for App
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop)
    {
        if self.with_state(|_| ()).is_some() { return; }

        let window = Arc::new(event_loop.create_window(
            Window::default_attributes()
                .with_title("CMSC427 Lab 02 – Parametric Objects")
                .with_inner_size(winit::dpi::PhysicalSize::new(1024u32, 720u32))
        ).unwrap());
        self.window = Some(window.clone());

        #[cfg(target_arch = "wasm32")]
        {
            use winit::platform::web::WindowExtWebSys;
            web_sys::window()
                .and_then(|w| w.document())
                .and_then(|d| d.get_element_by_id("canvas-host")
                    .or_else(|| d.body().map(|b| b.into())))
                .and_then(|host| window.canvas()
                    .and_then(|c| host.append_child(&c).ok()));
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            self.state = Some(pollster::block_on(State::new(window)));
        }

        #[cfg(target_arch = "wasm32")]
        {
            let cell   = self.state.clone();
            let window = window.clone();
            set_status("Loading renderer\u{2026}");
            wasm_bindgen_futures::spawn_local(async move {
                let state = State::new(window.clone()).await;
                set_status("Renderer ready.");
                *cell.borrow_mut() = Some(state);
                window.request_redraw();
            });
        }
    }

    // kbm + events here
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id:        winit::window::WindowId,
        event:      WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(s) => {
                self.with_state(|state| state.resize(s.width, s.height));
            }

            WindowEvent::RedrawRequested => {
                if self.with_state(|state| { let _ = state.render(); }).is_none() {
                    if let Some(w) = &self.window { w.request_redraw(); }
                }
            }

            // ElementState::Pressed fires once
            // change to ElementState::Released for continuous if I can figure out repeated presses later
            WindowEvent::KeyboardInput {
                event: winit::event::KeyEvent {
                    physical_key: PhysicalKey::Code(key),
                    state:        ElementState::Pressed,
                    ..
                },
                ..
            } => match key {
                KeyCode::Escape  => event_loop.exit(),
                KeyCode::Digit1  => { self.with_state(|s| s.set_shape(0)); }
                KeyCode::Digit2  => { self.with_state(|s| s.set_shape(1)); }
                KeyCode::Digit3  => { self.with_state(|s| s.set_shape(2)); }
                KeyCode::Digit4  => { self.with_state(|s| s.set_shape(3)); }
                KeyCode::Equal   => { self.with_state(|s| s.set_scale(1.2)); }
                KeyCode::Minus   => { self.with_state(|s| s.set_scale(1.0 / 1.2)); }
                _ => {}
            },

            _ => {}
        }
    }
}

// Entry points

pub fn run()
{
    let event_loop = EventLoop::new().unwrap();
    let mut app    = App::new();
    event_loop.run_app(&mut app).unwrap();
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn start()
{
    console_log::init_with_level(log::Level::Warn).ok();
    console_error_panic_hook::set_once();

    use winit::platform::web::EventLoopExtWebSys;
    let event_loop = EventLoop::new().unwrap();
    event_loop.spawn_app(App::new());
}
