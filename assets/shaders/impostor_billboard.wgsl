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
#import bevy_pbr::mesh_view_bindings::view

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

// x: quad side in world metres (the bake frustum width), yzw: direction
// TO the sun in world space.
@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> span_sun: vec4<f32>;
// rgb: sun colour, w: ambient fraction.
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<uniform> light: vec4<f32>;
// xyz: the variation's bake centre in model-local space (usually zero;
// shifts the quad to where the mesh's bulk sits relative to the origin).
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var<uniform> bake_center: vec4<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var atlas_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(4) var atlas_sampler: sampler;

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
    // above already selected the correct view. The 180-degree negation
    // maps the image's top edge (atlas v_min) onto the screen-top corner:
    // atlas v grows down the image while +corner.y places vertices
    // screen-up.
    let corner = v.position.xz; // quad mesh: +/-0.5 corners
    let spun = -corner;

    // Yaw-rotated bake centre, so leaning models sit in their quad where
    // the mesh bulk sat in the bake frustum.
    let cy = cos(yaw);
    let sy = sin(yaw);
    let center_offset = vec3<f32>(
        bake_center.x * cy - bake_center.z * sy,
        bake_center.y,
        bake_center.x * sy + bake_center.z * cy,
    );

    let world = position + center_offset + (right * spun.x + up * spun.y) * span_sun.x;

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
    // Mirrored view-space normal; the sRGB texture decode returns it to
    // linear, then n = 2v - 1 (the bake stores n * 0.5 + 0.5).
    let normal =
        textureSample(atlas_texture, atlas_sampler, vec2<f32>(1.0) - in.uv).xyz * 2.0 - 1.0;
    let sun_view = normalize((view.view_from_world * vec4<f32>(span_sun.yzw, 0.0)).xyz);
    let lambert = max(dot(normal, sun_view), 0.0);
    let lit = albedo.rgb * (light.rgb * lambert + vec3<f32>(light.w));
    // The cutout itself: `AlphaMode::Mask` only routes the pipeline into
    // the opaque binning phase (where instancing lives) — the discard is
    // the custom shader's job (the standard PBR fragment does it from the
    // material's alpha_cutoff). Without it the atlas's transparent
    // background (0,0,0,0) writes black behind every sprite. Threshold
    // mirrors BillboardMaterial::alpha_mode's Mask(0.5).
    if (albedo.a < 0.5) {
        discard;
    }
    return vec4<f32>(lit, albedo.a);
}
