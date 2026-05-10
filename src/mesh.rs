// mesh.rs — vertex type, GPU mesh handle, and procedural mesh generators for lab02

// meshes use index buffers (Uint16):
// each unique vertex stored once in the VBO, index list describes which to connect
// shared corners (e.g. the apex of the cone) appear once in the VBO but many times in the index list
// Uint16 caps at 65535 unique verts — fine for these meshes, half the size of Uint32
//
// each make_*() returns (Vec<Vertex>, Vec<u16>): verts + indices
// caller passes both to GpuMesh::upload() which sends them to the GPU

// Objects made:
// make_barn() — pentagonal prism extruded in Z, fan-triangulated caps (bit simpler than the drawing but meshes are hard okay)
// make_cone(n) — n-sided cone, apex + base ring + base centre
// make_vase(s, sl) — lathe surface, r(t) = 0.08 + 0.38·sin²(t·π)
// make_disc(r, s) — flat polar grid, centre vert + concentric rings

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

// Vertex
// 3d posn + RGB colour
// colour in at generation time — no textures or normals interp for now

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Vertex
{
    pub position: [f32; 3],  // object-space XYZ
    pub color:    [f32; 3],  // RGB in [0.0, 1.0]
}

impl Vertex
{
    // vbo layout
    pub fn layout() -> wgpu::VertexBufferLayout<'static>
    {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset:          0,
                    shader_location: 0,
                    format:          wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset:          std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format:          wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

// shorthand so mesh generators don't repeat vertex struct literal
fn v(x: f32, y: f32, z: f32, r: f32, g: f32, b: f32) -> Vertex
{
    Vertex { position: [x, y, z], color: [r, g, b] }
}

// GpuMesh: pair of uploaded GPU buffers (vertex + index) + index count
// vbuf holds the unique vertices, ibuf holds the triangle index list
// index_count is passed to draw_indexed() as the range end
// both buffers in VRAM until GpuMesh is dropped via rust ownership
pub struct GpuMesh
{
    pub vbuf:        wgpu::Buffer,
    pub ibuf:        wgpu::Buffer,
    pub index_count: u32,
}

impl GpuMesh
{
    // sends verts and indices to the GPU and returns handles to both buffers
    // VERTEX / INDEX usage flags tell wgpu how the pipeline will access them
    pub fn upload(device: &wgpu::Device, verts: &[Vertex], indices: &[u16]) -> Self
    {
        let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("VBuf"),
            contents: bytemuck::cast_slice(verts),
            usage:    wgpu::BufferUsages::VERTEX,
        });
        let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("IBuf"),
            contents: bytemuck::cast_slice(indices),
            usage:    wgpu::BufferUsages::INDEX,
        });
        Self { vbuf, ibuf, index_count: indices.len() as u32 }
    }
}


// Procedural mesh generators

// Barn — pentagonal prism
// 5-point cross-section (a barn silhouette) extruded from z=-0.55 to z=+0.55
// profile vertices (XY plane):
// p0 = ( 0.0,  0.65) — roof top edge
// p1 = (-0.5,  0.15) — left eave
// p2 = (-0.5, -0.55) — bottom-left corner
// p3 = ( 0.5, -0.55) — bottom-right corner
// p4 = ( 0.5,  0.15) — right eave
// caps fan triangulated from v0: triangles (0,1,2), (0,2,3), (0,3,4)
// back cap uses reversed winding 
// 5 side quads each get their own 4 verts for independent colors
pub fn make_barn() -> (Vec<Vertex>, Vec<u16>)
{
    let profile: &[(f32, f32)] = &[
        ( 0.0,  0.65),  // p0
        (-0.5,  0.15),  // p1
        (-0.5, -0.55),  // p2
        ( 0.5, -0.55),  // p3
        ( 0.5,  0.15),  // p4
    ];
    let n = profile.len() as u16;

    let front_col = [0.75_f32, 0.30, 0.15];
    let back_col  = [0.60_f32, 0.20, 0.10];
    let wall_cols: &[[f32; 3]] = &[
        [0.55, 0.40, 0.25],  // roof left wall
        [0.45, 0.45, 0.45],  // left wall
        [0.40, 0.40, 0.40],  // base
        [0.45, 0.45, 0.45],  // right wall
        [0.55, 0.40, 0.25],  // roof right wall
    ];

    let mut verts: Vec<Vertex> = Vec::new();
    let mut idx:   Vec<u16>    = Vec::new();

    // front and back
    for &(x, y) in profile { verts.push(v(x, y, -0.55, front_col[0], front_col[1], front_col[2])); }
    for &(x, y) in profile { verts.push(v(x, y,  0.55, back_col[0],  back_col[1],  back_col[2]));  }

    // front cap — fan from vert 0 (CCW when viewed from -Z)
    for i in 1..(n - 1) { idx.extend_from_slice(&[0, i, i + 1]); }

    // back cap — fan from vert n, reversed winding to face outward (+Z)
    for i in 1..(n - 1) { idx.extend_from_slice(&[n, n + i + 1, n + i]); }

    // side quads
    // brightness variation (0.8x, 0.9x) fakes ambient occlusion on different walls, lil trick I learned in game dev club
    for i in 0..(n as usize) {
        let j    = (i + 1) % n as usize;
        let c    = wall_cols[i];
        let fi   = profile[i];
        let fj   = profile[j];
        let base = verts.len() as u16;
        verts.push(v(fi.0, fi.1, -0.55, c[0],     c[1],     c[2]));
        verts.push(v(fj.0, fj.1, -0.55, c[0]*0.8, c[1]*0.8, c[2]*0.8));
        verts.push(v(fj.0, fj.1,  0.55, c[0]*0.9, c[1]*0.9, c[2]*0.9));
        verts.push(v(fi.0, fi.1,  0.55, c[0],     c[1],     c[2]));
        // split quad into two CCW triangles
        idx.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
    }

    (verts, idx)
}

// Cone — n-sided cone

// tip at (0, 0.5, 0), circular base at y=-0.5 with radius 0.45
// n controls how many triangular side faces approximate the cone
// increase n for a rounder silhouette (32 is a good tradeoff)

// vert layout:
//   0          — tip (bright gold)
//   1          — base centre (dark gold)
//   2..n+2     — base ring, colour cycles orange→yellow around circumference

// side triangles: {apex, ring[i], ring[i+1]}
// base triangles: {base_ctr, ring[i+1], ring[i]} — reversed winding to face down
pub fn make_cone(n: u32) -> (Vec<Vertex>, Vec<u16>)
{
    let mut verts = Vec::new();
    let mut idx   = Vec::new();

    verts.push(v(0.0,  0.5, 0.0, 1.0, 0.8, 0.0));  // tip
    verts.push(v(0.0, -0.5, 0.0, 0.7, 0.3, 0.0));  // base center

    // base ring: n vertices evenly spaced around a circle at y=-0.5
    for i in 0..n {
        let a  = i as f32 / n as f32 * std::f32::consts::TAU;
        let x  = 0.45 * a.cos();
        let z  = 0.45 * a.sin();
        let t  = i as f32 / n as f32;
        let cr = [0.9 - 0.4 * t, 0.5 + 0.2 * t, 0.1];
        verts.push(v(x, -0.5, z, cr[0], cr[1], cr[2]));
    }

    for i in 0..n {
        let a = 2 + i;
        let b = 2 + (i + 1) % n;
        idx.extend_from_slice(&[0, a as u16, b as u16]);  // side face 
        idx.extend_from_slice(&[1, b as u16, a as u16]);  // base face
    }

    (verts, idx)
}

// Vase — lathe surface/surface of revolution

// rotates 2D profile curve around the Y axis to produce a 3D surface
// profile radius function:
// r(t) = 0.08 + 0.38 · sin^2(t*pi),

// stacks — rings along height axis 
// slices — segments around circumference
pub fn make_vase(stacks: u32, slices: u32) -> (Vec<Vertex>, Vec<u16>)
{
    let mut verts = Vec::new();
    let mut idx   = Vec::new();

    for st in 0..=stacks {
        let t = st as f32 / stacks as f32;   
        let y = -0.7 + t * 1.4;                
        let s = (t * std::f32::consts::PI).sin();
        let r = 0.08 + 0.38 * s * s;  

        for sl in 0..slices {
            let a = sl as f32 / slices as f32 * std::f32::consts::TAU;
            let c = [0.3 + 0.5 * t, 0.7 - 0.2 * t, 0.8 - 0.6 * t];  // color gradient
            verts.push(v(r * a.cos(), y, r * a.sin(), c[0], c[1], c[2]));
        }
    }

    // connect adjacent stack rings into quads, then split each quad into 2 triangles
    // CCW from outside
    for st in 0..stacks {
        for sl in 0..slices {
            let ns = (sl + 1) % slices;
            let a  = (st * slices + sl)        as u16;
            let b  = (st * slices + ns)        as u16;
            let c  = ((st + 1) * slices + sl)  as u16;
            let d  = ((st + 1) * slices + ns)  as u16;
            idx.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }

    (verts, idx)
}

// Disc — flat polar grid at y=0
// centre vertex + concentric rings, each subdivided into sectors/arcs
// innermost ring fans from center
// outer rings use quads, same connection pattern as make_vase but in XZ plane
pub fn make_disc(rings: u32, sectors: u32) -> (Vec<Vertex>, Vec<u16>)
{
    let mut verts = Vec::new();
    let mut idx   = Vec::new();

    verts.push(v(0.0, 0.0, 0.0, 1.0, 1.0, 0.6));

    for ri in 1..=rings {
        let r  = ri as f32 / rings as f32 * 0.6; 
        let t  = ri as f32 / rings as f32; 
        for si in 0..sectors {
            let a  = si as f32 / sectors as f32 * std::f32::consts::TAU;
            let ht = si as f32 / sectors as f32; // rainbow effect
            let c  = [
                (ht * std::f32::consts::TAU).sin() * 0.5 + 0.5,
                0.2 + 0.4 * t,
                (ht * std::f32::consts::TAU).cos() * 0.5 + 0.5,
            ];
            verts.push(v(r * a.cos(), 0.0, r * a.sin(), c[0], c[1], c[2]));
        }
    }

    // inner rings
    for si in 0..sectors {
        let a = 1 + si as u16;
        let b = 1 + ((si + 1) % sectors) as u16;
        idx.extend_from_slice(&[0, a, b]);
    }

    // remaining rings — connect adj rings with quads
    // might be smarter way to do this
    for ri in 1..rings {
        for si in 0..sectors {
            let ns = (si + 1) % sectors;
            let a  = (1 + (ri - 1) * sectors + si)  as u16;
            let b  = (1 + (ri - 1) * sectors + ns)  as u16;
            let c  = (1 + ri       * sectors + si)   as u16;
            let d  = (1 + ri       * sectors + ns)   as u16;
            idx.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }

    (verts, idx)
}
