// Impostor billboard: one camera-facing quad per far boid, sampling the
// preprocessed impostor atlas. The layout and the albedo/normal pairing
// mirror src/preprocess/atlas.rs (see docs/plans/impostor-billboard-lod.md
// for the design): each pose's albedo cells band the top half, view-space
// normals sit at the whole-texture 180-degree rotation (nuv = 1 - uv).
//
// Per-instance data rides the engine's MeshTag channel (automatic
// instancing): low 16 bits = boid yaw as a fraction of tau, bits 16..24 =
// pose index. The vertex stage folds the yaw into the view azimuth
// (exact in geometry because the models are upright — see the atlas docs)
// and picks the nearest baked view cell; the sprite itself is always
// displayed upright, because world up projects up-image in every baked
// cell (any azimuth) and up-screen here — yaw lives entirely in which
// cell gets picked. The fragment stage lights the unlit albedo with the
// mirrored normal.

#import bevy_pbr::{
    mesh_functions,
    forward_io::Vertex,
    view_transformations::position_world_to_clip,
}
#import bevy_pbr::mesh_view_bindings::{view, lights}
#import bevy_core_pipeline::tonemapping::tone_mapping

// --- atlas layout: mirror of src/preprocess/atlas.rs (keep in sync) -----
const ATLAS_W: u32 = 7u;                 // width in cells
const HALF_ROWS: u32 = 7u;               // albedo rows per pose
const POSE_COUNT: u32 = 1u;
const GRID_H: u32 = 2u * HALF_ROWS * POSE_COUNT;
const MAX_POLAR: f32 = 30.0 * PI / 180.0;
const RING_COUNT: u32 = 3u;
// Azimuth slots per ring, and each ring's first flat view index (the
// nadir cell is index 0; ring r starts at 1 + previous ring widths).
const RING_SLOTS: array<u32, RING_COUNT> = array<u32, RING_COUNT>(8u, 16u, 24u);
const RING_START: array<u32, RING_COUNT> = array<u32, RING_COUNT>(1u, 9u, 25u);

const TAU: f32 = 6.283185307179586;
const PI: f32 = 3.141592653589793;

// xyz: the variation's bake centre in model-local space (usually zero;
// shifts the quad to where the mesh's bulk sits relative to the origin),
// w: quad side in world metres (the bake frustum width).
@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> center_span: vec4<f32>;
// Manual match knob from BillboardTuning.brightness; 1.0 = PBR parity.
// x carries the value, yzw are padding: WebGL2 (wgpu's GL fallback)
// rejects uniform bindings whose size is not a multiple of 16 bytes —
// a bare f32 here killed pipeline creation on browsers without WebGPU.
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<uniform> brightness: vec4<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var atlas_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var atlas_sampler: sampler;

struct BillboardOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vertex(v: Vertex) -> BillboardOut {
    let tag = mesh_functions::get_tag(v.instance_index);
    let yaw = f32(tag & 0xffffu) / 65536.0 * TAU;
    let pose = (tag >> 16u) & 0xffu;

    let world_from_local = mesh_functions::get_world_from_local(v.instance_index);
    let position = world_from_local[3].xyz;

    // View direction camera -> boid, matching atlas::view_direction: polar
    // is measured from straight down, azimuth from +X toward +Z.
    let look = normalize(position - view.world_position);
    let polar = acos(clamp(-look.y, -1.0, 1.0));
    let folded = atan2(look.z, look.x) - yaw;

    // Nearest baked view (mirror of atlas::view_index, cone clamp
    // included: views steeper than 30 degrees read the outer ring).
    let ring = clamp(u32(round(polar / MAX_POLAR * f32(RING_COUNT))), 0u, RING_COUNT);
    var view_i = 0u;
    if (ring > 0u) {
        let slots = RING_SLOTS[ring - 1u];
        let azimuth = (folded % TAU + TAU) % TAU; // wrap negatives
        let slot = u32(azimuth * f32(slots) / TAU) % slots;
        view_i = RING_START[ring - 1u] + slot;
    }

    // View right/up axes in world space: the rows of view_from_world.
    let vf = view.view_from_world;
    let right = normalize(vec3<f32>(vf[0].x, vf[1].x, vf[2].x));
    let up = normalize(vec3<f32>(vf[0].y, vf[1].y, vf[2].y));

    // No spin: the baked capsule is upright in every cell (world up
    // projects up-image in the bake for any azimuth) and world up projects
    // up-screen here, so the sprite is displayed as-is — the folded yaw
    // above already selected the correct view. X maps straight through
    // (image u grows along screen-right, matching the bake camera's right
    // axis — negating it would display the baked view mirrored, invisible
    // on symmetric albedo but a horizontal flip of the baked normals
    // against the runtime view basis, lighting the sprite from the wrong
    // side). Y negates: atlas v grows down the image while +corner.y
    // places vertices screen-up.
    let corner = v.position.xz; // quad mesh: +/-0.5 corners
    let spun = vec2<f32>(corner.x, -corner.y);

    // Yaw-rotated bake centre, so leaning models sit in their quad where
    // the mesh bulk sat in the bake frustum.
    let cy = cos(yaw);
    let sy = sin(yaw);
    let center_offset = vec3<f32>(
        center_span.x * cy - center_span.z * sy,
        center_span.y,
        center_span.x * sy + center_span.z * cy,
    );

    let world = position + center_offset + (right * spun.x + up * spun.y) * center_span.w;

    var out: BillboardOut;
    out.position = position_world_to_clip(world);
    let cell = vec2<f32>(
        f32(view_i % ATLAS_W) + corner.x + 0.5,
        f32(HALF_ROWS * pose + view_i / ATLAS_W) + corner.y + 0.5,
    );
    out.uv = cell / vec2<f32>(f32(ATLAS_W), f32(GRID_H));
    return out;
}

@fragment
fn fragment(in: BillboardOut) -> @location(0) vec4<f32> {
    let albedo = textureSample(atlas_texture, atlas_sampler, in.uv);
    // The cutout: `AlphaMode::Mask` only routes the pipeline into the
    // opaque binning phase (where instancing lives) — the discard is the
    // custom shader's job (the standard PBR fragment does it from the
    // material's alpha_cutoff). Without it the atlas's transparent
    // background (0,0,0,0) writes black behind every sprite. Threshold
    // mirrors BillboardMaterial::alpha_mode's Mask(0.5).
    if (albedo.a < 0.5) {
        discard;
    }
    // Mirrored view-space normal; the sRGB texture decode returns it to
    // linear, then n = 2v - 1 (the bake stores n * 0.5 + 0.5).
    let normal =
        textureSample(atlas_texture, atlas_sampler, vec2<f32>(1.0) - in.uv).xyz * 2.0 - 1.0;

    // Lighting comes straight from the engine's view bind group — the
    // same `lights` uniform the PBR meshes read — so ambient strength,
    // sun colour (premultiplied by illuminance), sun direction, light
    // count and any future day/night state apply to billboards and
    // meshes from one source, with no sync system in between. Zero
    // directional lights (night) leaves the ambient term only.
    var diffuse_light = lights.ambient_color.rgb;
    for (var i: u32 = 0u; i < lights.n_directional_lights; i = i + 1u) {
        let l = &lights.directional_lights[i];
        // Direction TO the light, in the runtime view basis — matching
        // the baked normal up to the documented yaw-fold approximation
        // (see the atlas docs' orientation-folding note).
        let l_view = normalize((view.view_from_world * vec4<f32>((*l).direction_to_light, 0.0)).xyz);
        // PBR's diffuse for our matte capsules is Burley with roughness
        // 0.25, whose Schlick terms sit within a few percent of 1 — the
        // Lambert limit. Specular (a dielectric F0 of ~0.04 sheen) is
        // below the billboard's perceptual floor and skipped.
        diffuse_light += (*l).color.rgb * max(dot(normal, l_view), 0.0) / PI;
    }

    // The `lights` uniform is photometric HDR (sun colour carries ~10⁴
    // lux) and every PBR fragment scales its summed light by
    // `view.exposure` before writing (pbr_functions.wgsl: `view.exposure *
    // (transmitted_light + direct_light + indirect_light)`). The default
    // camera exposure (`Exposure::BLENDER`, EV100 9.7 ≈ ×1.4e-3) is what
    // brings daylight down to display scale; without it the sprite rides
    // the tonemap shoulder ~700× too high — colour survives at grazing
    // sun angles but crushes to a white blob wherever NdotL nears 1
    // (the 302 m bench view). `brightness` is the manual match knob on
    // top (1.0 = parity). Distance fog and deband dither from the
    // standard chain are skipped — neither is used by this game's views.
    diffuse_light *= view.exposure * brightness.x;

    // The same in-shader post-lighting step the PBR fragment performs on
    // non-HDR cameras: without tonemapping the raw value saturates the
    // sRGB target and the whole sprite clips to white.
    var color = vec4<f32>(albedo.rgb * diffuse_light, albedo.a);
#ifdef TONEMAP_IN_SHADER
    color = tone_mapping(color, view.color_grading);
#endif
    return color;
}
