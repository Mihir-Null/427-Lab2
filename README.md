# CMSC427 Lab 02 Report: Parametric Objects

**Live demo:** https://mihir-null.github.io/427-Lab2/

This lab extends the pipeline from lab 01 into three dimensions. It adds index buffers, a combined Model-View-Projection matrix uploaded as a uniform, and four procedurally generated meshes that can be switched at runtime. I also added a scale keybind (+/-) that multiplies the Model matrix to shrink or grow the object without touching vertex data. Infrastructure is ported from lab 0/1 and extended with a dedicated `mesh.rs` module and the `glam` math library for matrix operations. 

As before, I used the [learn-wgpu tutorial](https://sotrh.github.io/learn-wgpu/) as the primary reference and AI tools for understanding and mapping concepts, not for writing code. VSCode extensions were used for linting.

## MVP Matrices

The core new concept in this lab is the combined Model-View-Projection matrix. All 3 transforms are multiplied together on the CPU each frame and uploaded as a single 64-byte uniform buffer:

```rust
let proj  = Mat4::perspective_rh(45_f32.to_radians(), self.gpu.aspect(), 0.1, 100.0);
let view  = Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y);
let model = Mat4::from_scale(Vec3::splat(self.scale));
let mvp   = proj * view * model;
```

The vertex shader then transforms each vertex with a single multiply: `clip = mvp * vec4(pos, 1.0)`. The `w=1.0` makes it a point so translations from the Model and View matrices apply. Using `w=0.0` for normals or directions would suppress translations.

The three matrices serve different roles: Model moves the mesh from object space to world space (here just applies scale), View positions the world relative to the orbiting camera, and Projection maps the view frustum to clip space. Multiplying them right-to-left means Model runs first.

## Index Buffers

All four meshes use index buffers. The vertex buffer stores each unique position once. The index buffer is a flat list of `u16` values describing which vertices to connect into triangles. `draw_indexed()` reads indices from the index buffer, then looks up each vertex in the vertex buffer.

```rust
rpass.set_index_buffer(mesh.ibuf.slice(..), wgpu::IndexFormat::Uint16);
rpass.draw_indexed(0..mesh.index_count, 0, 0..1);
```

`Uint16` caps at 65535 unique vertices, which is enough for all four meshes and uses half the memory of `Uint32`. The barn and disc use fan-triangulated caps where a single centre vertex is shared by many triangles, making index reuse especially worthwhile.

## Procedural Meshes

Four meshes are generated at startup in `mesh.rs` and all uploaded to GPU at once. Switching between them just changes which buffer is bound at draw time.

**Barn** — a pentagonal prism extruded along the Z axis. The 5-point profile is defined in the XY plane, then copied to z=±0.55. Caps are fan-triangulated from vertex 0. Each of the five side walls gets its own four vertices (not shared) so each wall can have a distinct colour tint.

**Cone** — an n-sided approximation to a smooth cone. The apex is at `(0, 0.5, 0)`, the base ring sits at y=−0.5. Increasing n (currently 32) makes the silhouette rounder.

**Vase** — a surface of revolution (lathe). A 2D profile curve is rotated around the Y axis:

```rust
let r = 0.08 + 0.38 * (t * PI).sin().powi(2);
```

At `t=0` and `t=1` the radius is 0.08 (narrow neck), at `t=0.5` it reaches 0.46 (wide belly). Stacks and slices control how finely the surface is subdivided.

**Disc** — a flat polar grid at y=0. One centre vertex fans out to the first ring; subsequent rings connect to each other with quads split into two triangles, same pattern as the vase.

## Scale Keybind

`=`/`+` increases scale by 20%, `-` decreases by 20%, both clamped to [0.1, 10.0]:

```rust
fn set_scale(&mut self, delta: f32) {
    self.scale = (self.scale * delta).clamp(0.1, 10.0);
}
```

The scale is applied via the Model matrix each frame rather than baked into the vertex buffer. This is the correct approach: changing vertex data would require re-uploading the entire buffer on every keypress, while updating the 64-byte uniform is nearly free.

## wgpu Porting Notes

The main difference from a WebGL setup is how uniform variables are handled. In wgpu, the uniform lives in a GPU buffer, and a bind group wires that buffer to a specific shader slot `@group(0) @binding(0)`. Changing the data means calling `queue.write_buffer()`, not a uniform call. The bind group itself doesn't change.

The mesh topology (TriangleList) is baked into the pipeline object in wgpu rather than passed as an argument to the draw call. This means a different primitive topology requires a different pipeline, hence the two pipelines in lab 01 existed. Trianglelist is performant and intuitive so I'm sticking with that

## Result

The final result is a 3D mesh viewer showing four procedurally generated objects under a continuously orbiting camera. Pressing 1-4 switches meshes immediately, and +/- resizes the current mesh by scaling the Model matrix. The orbiting camera shows all sides of each mesh without any input, which makes it easy to spot geometry bugs.
