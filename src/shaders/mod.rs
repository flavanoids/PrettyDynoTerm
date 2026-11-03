pub const UNIFIED_SHADER: &str = r#"
struct Uniforms {
    time: f32,
    frame: u32,
    width: u32,
    height: u32,
    cam_pos: vec3<f32>,
    scene_id: u32,
    cam_dir: vec3<f32>,
    benchmark_mode: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> output_pixels: array<u32>;

fn pack_rgba(c: vec4<f32>) -> u32 {
    let r = u32(clamp(c.r * 255.0, 0.0, 255.0));
    let g = u32(clamp(c.g * 255.0, 0.0, 255.0));
    let b = u32(clamp(c.b * 255.0, 0.0, 255.0));
    let a = u32(clamp(c.a * 255.0, 0.0, 255.0));
    return r | (g << 8u) | (b << 16u) | (a << 24u);
}

// -------------------------------------------------------------
// SCENE 0: 3D TROCHOIDAL GERSTNER WAVES & HENYEY-GREENSTEIN ATMOSPHERE
// -------------------------------------------------------------

fn wave_height(p_in: vec2<f32>, t: f32) -> f32 {
    var h = 0.0;
    var freq = 0.28;
    var amp = 0.65;
    var speed = 0.55;
    var p = p_in;

    // 14 Gerstner / fBm wave octaves (high trigonometric pressure)
    for (var i = 0; i < 14; i = i + 1) {
        let angle = f32(i) * 1.396263;
        let dir = vec2<f32>(cos(angle), sin(angle));
        let phase = dot(dir, p) * freq + t * speed;
        let c = cos(phase);
        let s = sin(phase);

        // Trochoidal sharpening + non-linear peaking
        h = h + s * amp;
        h = h + abs(c) * amp * 0.38;

        p = p + dir * 0.32;
        freq = freq * 1.64;
        amp = amp * 0.51;
        speed = speed * 1.14;
    }
    return h;
}

// High-precision 4-tap central differences normal
fn sea_normal(p: vec3<f32>, t: f32) -> vec3<f32> {
    let eps = 0.02;
    let h1 = wave_height(p.xz + vec2<f32>(eps, 0.0), t);
    let h2 = wave_height(p.xz - vec2<f32>(eps, 0.0), t);
    let h3 = wave_height(p.xz + vec2<f32>(0.0, eps), t);
    let h4 = wave_height(p.xz - vec2<f32>(0.0, eps), t);
    let dx = (h1 - h2) / (2.0 * eps);
    let dz = (h3 - h4) / (2.0 * eps);
    return normalize(vec3<f32>(-dx, 1.0, -dz));
}

fn render_waves(ro_in: vec3<f32>, rd_in: vec3<f32>) -> vec4<f32> {
    let t = u.time * 0.40;
    let sun_dir = normalize(vec3<f32>(0.4, 0.22, -0.85));

    // Sky base colors
    let horizon_col = vec3<f32>(0.95, 0.12, 0.50);
    let zenith_col = vec3<f32>(0.02, 0.03, 0.10);
    let sky_factor = clamp(rd_in.y * 1.8 + 0.12, 0.0, 1.0);
    var col = mix(horizon_col, zenith_col, sky_factor);

    // Sun disc & atmospheric corona
    let sun_dot = max(dot(rd_in, sun_dir), 0.0);
    col = col + vec3<f32>(1.0, 0.55, 0.2) * pow(sun_dot, 36.0) * 1.2;

    // Dual-Phase Henyey-Greenstein Mie + Rayleigh Atmospheric Radiative Transfer
    let g = 0.72;
    let g2 = g * g;
    let denom = max(1.0 + g2 - 2.0 * g * sun_dot, 0.001);
    let phase_hg = (1.0 - g2) / (4.0 * 3.14159265 * pow(denom, 1.5));
    let phase_rayleigh = 0.75 * (1.0 + sun_dot * sun_dot);
    let phase = mix(phase_rayleigh, phase_hg, 0.65);

    var transmittance = 1.0;
    var mist_accum = vec3<f32>(0.0);
    let sun_col = vec3<f32>(1.0, 0.75, 0.45);
    let ambient_col = vec3<f32>(0.15, 0.22, 0.52);

    for (var m = 0; m < 48; m = m + 1) {
        let p_mist = ro_in + rd_in * (f32(m) * 1.5 + 1.2);
        let height_decay = exp(-max(p_mist.y, 0.0) * 0.35);
        let density = max(sin(p_mist.x * 0.18 + t) * cos(p_mist.z * 0.18 + t * 0.5) - 0.15, 0.0) * height_decay;
        let step_extinction = density * 0.055;
        let in_scatter = density * (phase * sun_col * 2.2 + ambient_col) * 0.055;
        mist_accum = mist_accum + in_scatter * transmittance;
        transmittance = transmittance * exp(-step_extinction);
    }
    col = col * transmittance + mist_accum;

    // Volumetric Cumulus Cloud Deck & Adiabatic Condensation (Sky load balancing)
    if (rd_in.y >= -0.01) {
        let cloud_base = 2.8;
        let cloud_top = 8.5;
        let t_cloud_start = max((cloud_base - ro_in.y) / max(rd_in.y, 0.01), 1.0);
        let t_cloud_end = min((cloud_top - ro_in.y) / max(rd_in.y, 0.01), 65.0);

        let c_steps = 48;
        let dt_c = (t_cloud_end - t_cloud_start) / f32(c_steps);

        for (var cs = 0; cs < c_steps; cs = cs + 1) {
            let p_c = ro_in + rd_in * (t_cloud_start + f32(cs) * dt_c);
            // 6-Octave 3D fractal Brownian motion cloud density
            var d_c = 0.0;
            var c_f = 0.16;
            var c_a = 0.58;
            for (var oct = 0; oct < 6; oct = oct + 1) {
                d_c = d_c + (sin(p_c.x * c_f + t * 0.35) * cos(p_c.z * c_f + t * 0.22) * sin(p_c.y * c_f * 1.4)) * c_a;
                c_f = c_f * 2.12;
                c_a = c_a * 0.48;
            }
            d_c = clamp(d_c - 0.10, 0.0, 1.0);

            if (d_c > 0.01) {
                let sun_scat = pow(max(dot(rd_in, sun_dir), 0.0), 16.0) * 2.0;
                let cloud_tone = mix(vec3<f32>(0.22, 0.16, 0.32), vec3<f32>(1.0, 0.88, 0.70), sun_scat);
                let ext = d_c * dt_c * 0.14;
                col = col + cloud_tone * (d_c * transmittance * 0.16);
                transmittance = transmittance * exp(-ext);
            }
        }
    }

    var t_ray = 0.0;
    var p = ro_in;
    var hit = false;

    // 112 Adaptive raymarch steps for water
    if (rd_in.y < -0.01) {
        let t_approx = (0.0 - ro_in.y) / rd_in.y;
        t_ray = max(t_approx - 10.0, 0.5);

        for (var step = 0; step < 112; step = step + 1) {
            p = ro_in + rd_in * t_ray;
            let h = wave_height(p.xz, t);
            let dist = p.y - h;
            if (dist < 0.035) {
                hit = true;
                break;
            }
            t_ray = t_ray + max(dist * 0.40, 0.035);
            if (t_ray > 110.0) { break; }
        }
    }

    if (hit) {
        let n = sea_normal(p, t);
        let fresnel = pow(1.0 - max(dot(-rd_in, n), 0.0), 4.5);
        
        let deep_water = vec3<f32>(0.008, 0.04, 0.14);
        let shallow_crest = vec3<f32>(0.0, 0.98, 0.82); // Bioluminescent cyan
        
        let crest = clamp((p.y - 0.22) * 1.7, 0.0, 1.0);
        let water_base = mix(deep_water, shallow_crest, crest * 0.75);

        // Water specular highlight & subsurface scattering
        let ref_dir = reflect(rd_in, n);
        let spec = pow(max(dot(ref_dir, sun_dir), 0.0), 48.0);
        let sss = pow(max(dot(sun_dir, -rd_in), 0.0), 4.0) * crest;

        // Secondary raycast: Trace water reflections
        var ref_col = col;
        let ref_p = p + ref_dir * 0.5;
        let ref_h = wave_height(ref_p.xz, t);
        if (ref_p.y < ref_h + 0.1) {
            ref_col = shallow_crest * 0.8;
        }

        col = mix(water_base, ref_col, fresnel * 0.65);
        col = col + vec3<f32>(1.0, 0.85, 0.65) * spec * 1.1;
        col = col + shallow_crest * sss * 1.2;

        let fog = clamp(t_ray / 95.0, 0.0, 1.0);
        col = mix(col, horizon_col, fog * 0.78);
    }

    return vec4<f32>(col, 1.0);
}

// -------------------------------------------------------------
// SCENE 1: 3D MANDELBULB WITH HUBBARD-DOUADY DERIVATIVES & CONE-TRACED AO
// -------------------------------------------------------------

fn mandelbulb_de(pos: vec3<f32>) -> vec2<f32> {
    var z = pos;
    var dr = 1.0;
    var r = 0.0;
    var orbit_trap = 1e10;

    let power = 8.0;

    // 24 Unrolled polynomial iterations per step (Deep micro-fractal precision)
    for (var i = 0; i < 24; i = i + 1) {
        r = length(z);
        if (r > 3.2) { break; }

        orbit_trap = min(orbit_trap, length(z.xz));

        var theta = acos(clamp(z.z / max(r, 1e-6), -1.0, 1.0));
        var phi = atan2(z.y, z.x);

        dr = pow(r, power - 1.0) * power * dr + 1.0;

        let zr = pow(r, power);
        theta = theta * power;
        phi = phi * power;

        z = zr * vec3<f32>(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)) + pos;
    }

    let r_safe = max(r, 1e-6);
    let dist = 0.5 * log(r_safe) * r_safe / max(dr, 1e-6);
    return vec2<f32>(dist, orbit_trap);
}

fn mandelbulb_normal(pos: vec3<f32>) -> vec3<f32> {
    let eps = 0.002;
    let d = mandelbulb_de(pos).x;
    let nx = mandelbulb_de(pos + vec3<f32>(eps, 0.0, 0.0)).x - d;
    let ny = mandelbulb_de(pos + vec3<f32>(0.0, eps, 0.0)).x - d;
    let nz = mandelbulb_de(pos + vec3<f32>(0.0, 0.0, eps)).x - d;
    return normalize(vec3<f32>(nx, ny, nz));
}

// 10-Tap Cone-Traced Ambient Occlusion using the Distance Field
fn mandelbulb_ao(pos: vec3<f32>, nor: vec3<f32>) -> f32 {
    var occ = 0.0;
    var sca = 1.0;
    for (var i = 1; i <= 10; i = i + 1) {
        let hr = 0.010 * f32(i);
        let aopos = pos + nor * hr;
        let dd = mandelbulb_de(aopos).x;
        occ = occ + (hr - dd) * sca;
        sca = sca * 0.72;
    }
    return clamp(1.0 - 3.8 * occ, 0.0, 1.0);
}

// 24-Step Soft shadow raymarch towards light
fn mandelbulb_shadow(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
    var res = 1.0;
    var t = 0.015;
    for (var i = 0; i < 24; i = i + 1) {
        let d = mandelbulb_de(ro + rd * t).x;
        if (d < 0.0015) {
            return 0.12;
        }
        res = min(res, 16.0 * d / t);
        t = t + max(d * 0.65, 0.010);
        if (t > 2.8) { break; }
    }
    return clamp(res, 0.12, 1.0);
}

fn render_mandelbulb(uv_x: f32, uv_y: f32) -> vec4<f32> {
    // Cinematic orbital camera path around the Mandelbulb core
    let t_cam = u.time * 0.14;
    let cam_dist = 2.22 + 0.12 * sin(t_cam * 0.7);
    let cam_pos = vec3<f32>(sin(t_cam) * cam_dist, 0.38 + 0.18 * cos(t_cam * 0.5), cos(t_cam) * cam_dist);
    let cam_target = vec3<f32>(0.0, 0.0, 0.0);
    let fwd = normalize(cam_target - cam_pos);
    let right = normalize(cross(fwd, vec3<f32>(0.0, 1.0, 0.0)));
    let up = cross(right, fwd);
    let fov_scale = 1.40;
    let rd = normalize(fwd * fov_scale + right * uv_x + up * uv_y);
    let ro = cam_pos;

    // Bounding sphere intersection (R = 2.2) to focus raymarch precisely on the fractal
    let r_bound = 2.2;
    let b = dot(ro, rd);
    let c = dot(ro, ro) - r_bound * r_bound;
    let disc = b * b - c;

    var t = 0.05;
    var t_max = 5.5;
    if (disc > 0.0) {
        let t_enter = -b - sqrt(disc);
        let t_exit = -b + sqrt(disc);
        t = max(t_enter, 0.05);
        t_max = max(t_exit, t);
    }

    var hit = false;
    var trap = 1.0;
    var glow = vec3<f32>(0.0);
    var p_hit = vec3<f32>(0.0);

    // 192 Raymarching steps with volumetric field accumulation
    for (var i = 0; i < 192; i = i + 1) {
        let p = ro + rd * t;
        let res = mandelbulb_de(p);
        let dist = res.x;
        trap = min(trap, res.y);

        // Volumetric aura / orbit-trap glow accumulation along ray
        let density = exp(-max(dist, 0.0) * 8.0);
        let aura_col = mix(vec3<f32>(0.0, 0.85, 1.0), vec3<f32>(1.0, 0.38, 0.08), clamp(res.y * 2.2, 0.0, 1.0));
        glow = glow + aura_col * (density * 0.018);

        if (dist < 0.0020) {
            hit = true;
            p_hit = p;
            break;
        }

        t = t + max(dist * 0.48, 0.0035);
        if (t > t_max) {
            break;
        }
    }

    var col = vec3<f32>(0.0);

    if (hit) {
        let n = mandelbulb_normal(p_hit);
        let light_dir = normalize(vec3<f32>(1.2, 1.6, 0.8));
        let diff = max(dot(n, light_dir), 0.0);
        let ao = mandelbulb_ao(p_hit, n);
        let shadow = mandelbulb_shadow(p_hit + n * 0.004, light_dir);

        let base_cyan = vec3<f32>(0.02, 0.85, 1.0);
        let molten_gold = vec3<f32>(1.0, 0.52, 0.08);
        let rim_purple = vec3<f32>(0.85, 0.12, 0.95);

        let fractal_color = mix(base_cyan, molten_gold, clamp(trap * 2.2, 0.0, 1.0));
        let rim = pow(1.0 - max(dot(-rd, n), 0.0), 3.0);
        let ref_dir = reflect(rd, n);
        let spec = pow(max(dot(ref_dir, light_dir), 0.0), 24.0);

        let lit_surface = (fractal_color * (diff * shadow * 0.85 + 0.15) + rim_purple * rim * 0.9 + vec3<f32>(1.0, 0.95, 0.85) * spec * shadow) * ao;
        col = lit_surface + glow;
    } else {
        let bg_stars = pow(clamp(sin(rd.x * 55.0) * sin(rd.y * 55.0) * sin(rd.z * 55.0), 0.0, 1.0), 22.0) * 0.7;
        let deep_space = vec3<f32>(0.005, 0.003, 0.015) + vec3<f32>(bg_stars);
        col = deep_space + glow * 1.5;
    }

    return vec4<f32>(col, 1.0);
}

// -------------------------------------------------------------
// SCENE 2: RELATIVISTIC KERR-SCHWARZSCHILD GEODESIC RK4 INTEGRATOR
// -------------------------------------------------------------

fn blackhole_accel(p: vec3<f32>) -> vec3<f32> {
    let r_sq = dot(p, p);
    let r = sqrt(r_sq);
    let r_quad = r_sq * r_sq;
    let r_event = 1.0;
    // Schwarzschild 2nd-order general relativistic curvature acceleration
    let a_grav = -1.5 * r_event * p / (r_quad + 0.001);

    // Kerr Frame-Dragging (Lense-Thirring effect) spin vector S = (0, 0.72, 0)
    let spin = vec3<f32>(0.0, 0.72, 0.0);
    let a_spin = 1.6 * cross(spin, p) / (r_quad * r + 0.001);

    return a_grav + a_spin;
}

fn render_blackhole(ro_in: vec3<f32>, rd_in: vec3<f32>) -> vec4<f32> {
    var p = ro_in;
    var dir = rd_in;
    let r_event = 1.0;
    let disk_r_in = 2.1;
    let disk_r_out = 8.2;

    var disk_accum = vec3<f32>(0.0);
    var escaped = false;

    let dt = 0.12;

    // 80 4th-Order Runge-Kutta (RK4) Relativistic Geodesic Steps
    for (var i = 0; i < 80; i = i + 1) {
        let r = length(p);
        if (r < r_event) {
            break;
        }
        if (r > 22.0) {
            escaped = true;
            break;
        }

        // Runge-Kutta 4 (RK4) geodesic integration
        let k1_v = blackhole_accel(p);
        let k1_p = dir;

        let p_half1 = p + k1_p * (0.5 * dt);
        let v_half1 = normalize(dir + k1_v * (0.5 * dt));
        let k2_v = blackhole_accel(p_half1);
        let k2_p = v_half1;

        let p_half2 = p + k2_p * (0.5 * dt);
        let v_half2 = normalize(dir + k2_v * (0.5 * dt));
        let k3_v = blackhole_accel(p_half2);
        let k3_p = v_half2;

        let p_full = p + k3_p * dt;
        let v_full = normalize(dir + k3_v * dt);
        let k4_v = blackhole_accel(p_full);
        let k4_p = v_full;

        let p_next = p + (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p) * (dt / 6.0);
        let dir_next = normalize(dir + (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) * (dt / 6.0));

        // Thin accretion disk equatorial plane crossing (y = 0)
        if ((p.y > 0.0 && p_next.y <= 0.0) || (p.y < 0.0 && p_next.y >= 0.0)) {
            let r_disk = length(p.xz);
            if (r_disk >= disk_r_in && r_disk <= disk_r_out) {
                let phi = atan2(p.z, p.x);
                let v_orbit = 1.0 / sqrt(r_disk);
                let v_rot = vec3<f32>(-sin(phi), 0.0, cos(phi)) * v_orbit;
                let v_dot_d = dot(v_rot, dir);

                // Relativistic Doppler Boost factor
                let gamma = 1.0 / sqrt(max(1.0 - v_orbit * v_orbit, 0.01));
                let doppler_factor = 1.0 / (gamma * max(1.0 - v_dot_d, 0.05));

                // General Relativistic Gravitational Redshift: g_grav = sqrt(1 - R_s / r)
                let g_grav = sqrt(max(1.0 - r_event / r_disk, 0.01));

                // Total Lorentz-invariant spectral shift: g = g_doppler * g_grav
                let g_total = doppler_factor * g_grav;

                // Liouville's theorem in curved spacetime: I_obs = g^4 * I_emit
                let beaming = pow(clamp(g_total, 0.1, 3.2), 4.0);

                // Novikov-Thorne standard thin-disk emissivity profile
                let nt_flux = (1.0 - sqrt(disk_r_in / r_disk)) / (r_disk * r_disk * r_disk + 0.1);
                let swirl = sin(r_disk * 4.8 - phi * 3.8 + u.time * 0.75);
                let intensity = nt_flux * 45.0 * beaming;

                let blue_plasma = vec3<f32>(0.2, 0.8, 1.0) * 2.2;
                let red_plasma = vec3<f32>(1.0, 0.18, 0.03);
                let plasma_col = mix(red_plasma, blue_plasma, clamp(v_dot_d * 1.2 + 0.5, 0.0, 1.0));

                disk_accum = disk_accum + plasma_col * intensity * (0.6 + 0.4 * swirl) * 0.75;
            }
        }

        p = p_next;
        dir = dir_next;
    }

    var bg = vec3<f32>(0.0);
    if (escaped) {
        let star_dir = dir;
        let star = pow(clamp(sin(star_dir.x * 45.0) * sin(star_dir.y * 45.0) * sin(star_dir.z * 45.0), 0.0, 1.0), 18.0);
        bg = vec3<f32>(star * 0.95);
    }

    let final_col = disk_accum + bg;
    return vec4<f32>(final_col, 1.0);
}

// -------------------------------------------------------------
// SCENE 3: TORNADIC SUPERCELL & BURGERS-ROTT MESOCYCLONE DYNAMICS
// -------------------------------------------------------------

fn tornado_wind_field(p: vec3<f32>, t: f32) -> vec4<f32> {
    let r_sq = dot(p.xz, p.xz);
    let r = sqrt(r_sq);
    let y = p.y;

    // Height-dependent core radius: widening funnel with altitude (hyperboloid of revolution)
    let r_c = 0.42 + 0.32 * max(y, 0.0) + 0.12 * pow(max(y - 2.2, 0.0), 2.0);

    // Burgers-Rott vortex tangential velocity profile
    let gamma = 6.2; // Circulation
    let v_theta = (gamma / (6.283185 * max(r, 0.08))) * (1.0 - exp(-r_sq / (r_c * r_c)));

    // Cyclostrophic pressure deficit: dp/dr = rho * v_theta^2 / r
    // Integrating yields the central pressure drop:
    let delta_p = -3.8 / (1.0 + (r_sq / (r_c * r_c)));

    // Thermodynamic dew-point condensation threshold
    // When pressure drops below condensation isobar, water vapor condenses
    let cond_threshold = -0.85;
    let condensation = clamp((-delta_p + cond_threshold) * 0.85, 0.0, 1.0);

    // Spiraling helical inflow bands
    let theta = atan2(p.z, p.x);
    let spiral = sin(theta * 3.0 - 2.2 * log(max(r, 0.15)) + y * 0.9 - t * 3.2);

    return vec4<f32>(v_theta, delta_p, condensation, spiral);
}

fn render_tornado(ro_in: vec3<f32>, rd_in: vec3<f32>) -> vec4<f32> {
    let t = u.time * 0.65;
    
    // Storm chaser perspective
    let ro = ro_in + vec3<f32>(0.0, -0.4, -1.8);
    let rd = rd_in;

    var col = vec3<f32>(0.06, 0.08, 0.12); // Ominous bruised supercell sky
    let sun_dir = normalize(vec3<f32>(0.8, 0.35, -0.6));

    // Distant lightning flash potential
    let flash_phase = sin(u.time * 1.8);
    let is_lightning = step(0.96, sin(u.time * 14.5) * flash_phase);
    let lightning_col = vec3<f32>(0.75, 0.90, 1.0) * is_lightning * 1.6;

    var transmittance = 1.0;
    var storm_accum = vec3<f32>(0.0);

    let t_start = 0.6;
    let t_end = 19.0;
    let steps = 112;
    let dt = (t_end - t_start) / f32(steps);

    for (var i = 0; i < steps; i = i + 1) {
        let p = ro + rd * (t_start + f32(i) * dt);

        if (p.y < 0.0 || p.y > 9.5) {
            continue;
        }

        let vortex = tornado_wind_field(p, t);
        let v_theta = vortex.x;
        let cond = vortex.z;
        let spiral = vortex.w;

        // Ground debris & dust collar friction boundary layer (y < 0.8)
        let debris_layer = clamp(1.0 - p.y / 0.85, 0.0, 1.0);
        let debris_r = length(p.xz);
        let debris_density = debris_layer * clamp(1.8 - debris_r * 0.9, 0.0, 1.0) * (0.6 + 0.4 * sin(p.x * 5.0 + t * 4.0));

        // Total cloud droplet + debris extinction coefficient
        let cloud_density = cond * (0.75 + 0.25 * spiral) + debris_density * 1.4;

        if (cloud_density > 0.015) {
            // Severe thunderstorm coloration:
            let core_col = vec3<f32>(0.12, 0.13, 0.16);
            let hail_green = vec3<f32>(0.08, 0.32, 0.28);
            let debris_col = vec3<f32>(0.35, 0.24, 0.14);

            var parcel_col = mix(core_col, hail_green, clamp((v_theta - 2.5) * 0.35, 0.0, 1.0));
            parcel_col = mix(parcel_col, debris_col, debris_layer * 0.85);

            // Forward Mie solar scattering along vortex rim
            let sun_rim = pow(max(dot(rd, sun_dir), 0.0), 8.0) * 0.6;
            parcel_col = parcel_col + vec3<f32>(1.0, 0.8, 0.5) * sun_rim;
            parcel_col = parcel_col + lightning_col * (0.4 + 0.6 * cond);

            let step_ext = cloud_density * dt * 0.38;
            storm_accum = storm_accum + parcel_col * (cloud_density * transmittance * 0.22);
            transmittance = transmittance * exp(-step_ext);

            if (transmittance < 0.02) {
                break;
            }
        }
    }

    col = col * transmittance + storm_accum;

    // Ground plane horizon with rain sheet
    if (rd.y < -0.01) {
        let t_ground = (0.0 - ro.y) / rd.y;
        if (t_ground > 0.0 && t_ground < t_end) {
            let p_g = ro + rd * t_ground;
            let wet_mud = vec3<f32>(0.04, 0.035, 0.03);
            let ground_fog = clamp(t_ground / 15.0, 0.0, 1.0);
            col = mix(wet_mud, col, ground_fog * 0.85);
        }
    }

    return vec4<f32>(col, 1.0);
}

// -------------------------------------------------------------
// SCENE 4: CARDIAC VENTRICULAR ARRHYTHMIA & ALIEV-PANFILOV ELECTROPHYSIOLOGY
// -------------------------------------------------------------

// Aliev-Panfilov 2-Variable Reaction-Diffusion System
// Simulates 3D scroll-wave re-entry rotors in ventricular fibrillation
fn aliev_panfilov_potential(pos: vec3<f32>, t: f32) -> vec2<f32> {
    let r = length(pos.xz);
    let phi = atan2(pos.z, pos.x);

    // Myocardial fiber helical rotation: fibers rotate ~120 degrees transmurally
    let fiber_pitch1 = pos.y * 1.15;
    let fiber_pitch2 = -pos.y * 0.85;

    // Primary counter-rotating scroll wave rotor (Ventricle anterior wall)
    let speed1 = 2.2;
    let phase1 = phi + fiber_pitch1 - t * speed1 + 1.65 * log(max(r, 0.08));
    let u1 = 1.0 / (1.0 + exp(-12.0 * sin(phase1)));
    let v1 = 0.5 * (1.0 + cos(phase1 - 1.35));

    // Secondary re-entrant rotor (Ventricle posterior septum: polymorphic arrhythmia)
    let speed2 = 1.75;
    let phase2 = -phi * 1.5 + fiber_pitch2 - t * speed2 + 2.1 * log(max(r, 0.06));
    let u2 = 1.0 / (1.0 + exp(-10.0 * sin(phase2)));
    let v2 = 0.5 * (1.0 + cos(phase2 - 1.10));

    // Superposed transmembrane potential with refractory suppression
    let u_total = clamp(max(u1, u2 * 0.85) - 0.35 * (v1 + v2 * 0.5), 0.0, 1.0);
    let v_total = clamp(0.6 * v1 + 0.4 * v2, 0.0, 1.0);

    return vec2<f32>(u_total, v_total);
}

// Thick-walled left ventricle anatomical signed distance field (SDF)
fn ventricle_sdf(pos: vec3<f32>) -> f32 {
    let p = pos;
    // Outer epicardial surface (ellipsoid)
    let r_epi = length(vec3<f32>(p.x / 1.08, p.y / 1.48, p.z / 1.08)) - 1.0;
    // Inner endocardial cavity (blood chamber)
    let r_endo = length(vec3<f32>(p.x / 0.72, (p.y + 0.12) / 1.18, p.z / 0.72)) - 1.0;

    // Truncated base at top (valve plane y > 1.15)
    let cap = p.y - 1.15;

    // Thick myocardial muscle wall: between endo and epi
    let wall = max(r_epi, -r_endo);
    return max(wall, cap);
}

fn ventricle_normal(pos: vec3<f32>) -> vec3<f32> {
    let eps = 0.0025;
    let d = ventricle_sdf(pos);
    let nx = ventricle_sdf(pos + vec3<f32>(eps, 0.0, 0.0)) - d;
    let ny = ventricle_sdf(pos + vec3<f32>(0.0, eps, 0.0)) - d;
    let nz = ventricle_sdf(pos + vec3<f32>(0.0, 0.0, eps)) - d;
    return normalize(vec3<f32>(nx, ny, nz));
}

// 10-Tap tissue ambient occlusion
fn ventricle_ao(pos: vec3<f32>, nor: vec3<f32>) -> f32 {
    var occ = 0.0;
    var sca = 1.0;
    for (var i = 1; i <= 10; i = i + 1) {
        let hr = 0.012 * f32(i);
        let aopos = pos + nor * hr;
        let dd = ventricle_sdf(aopos);
        occ = occ + (hr - dd) * sca;
        sca = sca * 0.75;
    }
    return clamp(1.0 - 3.8 * occ, 0.0, 1.0);
}

fn render_cardiac(uv_x: f32, uv_y: f32) -> vec4<f32> {
    let t_cam = u.time * 0.18;
    // Close orbital camera so the ventricle fills 75-80% of the viewport
    let cam_dist = 1.95;
    let cam_pos = vec3<f32>(sin(t_cam) * cam_dist, 0.35 + 0.15 * cos(t_cam * 0.6), cos(t_cam) * cam_dist);
    let cam_target = vec3<f32>(0.0, 0.0, 0.0);
    let fwd = normalize(cam_target - cam_pos);
    let right = normalize(cross(fwd, vec3<f32>(0.0, 1.0, 0.0)));
    let up = cross(right, fwd);
    let fov = 1.55;
    let rd = normalize(fwd * fov + right * uv_x + up * uv_y);
    let ro = cam_pos;

    // Bounding sphere intersection (R = 1.75)
    let r_b = 1.75;
    let b = dot(ro, rd);
    let c = dot(ro, ro) - r_b * r_b;
    let disc = b * b - c;

    var t = 0.05;
    var t_max = 4.5;
    if (disc > 0.0) {
        let t_enter = -b - sqrt(disc);
        let t_exit = -b + sqrt(disc);
        t = max(t_enter, 0.05);
        t_max = max(t_exit, t);
    }

    var hit = false;
    var p_hit = vec3<f32>(0.0);
    var accum_glow = vec3<f32>(0.0);

    // 160 Raymarching steps with volumetric electrophysiological glow
    for (var i = 0; i < 160; i = i + 1) {
        let p = ro + rd * t;
        let dist = ventricle_sdf(p);

        // Volumetric optical mapping laser fluorescence glow
        let ap = aliev_panfilov_potential(p, u.time);
        let u_pot = ap.x;
        let v_rec = ap.y;

        // Fluorescent depolarization wave aura (cyan/mint 40 mV peak)
        let depol_col = vec3<f32>(0.0, 0.95, 0.82) * u_pot;
        let repol_col = vec3<f32>(1.0, 0.48, 0.10) * v_rec;
        let wave_aura = (depol_col + repol_col) * exp(-max(dist, 0.0) * 7.5);
        accum_glow = accum_glow + wave_aura * 0.016;

        if (dist < 0.0022) {
            hit = true;
            p_hit = p;
            break;
        }

        t = t + max(dist * 0.38, 0.0035);
        if (t > t_max) {
            break;
        }
    }

    var col = vec3<f32>(0.01, 0.015, 0.035); // Clinical optical mapping dark field

    if (hit) {
        let n = ventricle_normal(p_hit);
        let ao = ventricle_ao(p_hit, n);
        let light_dir = normalize(vec3<f32>(1.2, 1.8, 1.0));
        let diff = max(dot(n, light_dir), 0.0);
        let rim = pow(1.0 - max(dot(-rd, n), 0.0), 3.0);

        let ap = aliev_panfilov_potential(p_hit, u.time);
        let u_pot = ap.x;
        let v_rec = ap.y;

        // Optical mapping false-color voltage gradient:
        // Resting myocardium (-85 mV): Deep crimson vascular muscle
        // Depolarization wavefront (+30 mV): Bioluminescent cyan-mint
        // Plateau / Refractory phase (0 mV): Amber-gold
        let resting_myocardium = vec3<f32>(0.38, 0.03, 0.05);
        let depol_wave = vec3<f32>(0.0, 1.0, 0.85);
        let refractory_wave = vec3<f32>(1.0, 0.55, 0.08);

        var tissue_col = mix(resting_myocardium, refractory_wave, v_rec * 0.75);
        tissue_col = mix(tissue_col, depol_wave, u_pot * 0.95);

        // Coronary vessel branching surface relief
        let vessel = sin(p_hit.x * 16.0 + p_hit.y * 12.0) * cos(p_hit.z * 16.0);
        let vessel_factor = smoothstep(0.72, 0.95, abs(vessel));
        tissue_col = mix(tissue_col, vec3<f32>(0.85, 0.08, 0.02), vessel_factor * 0.6);

        // Transmural myocardial light penetration (20 subsurface scattering steps through muscle wall)
        var sss_accum = vec3<f32>(0.0);
        for (var s = 1; s <= 20; s = s + 1) {
            let p_sub = p_hit - n * (f32(s) * 0.008);
            let ap_sub = aliev_panfilov_potential(p_sub, u.time);
            let sub_glow = mix(vec3<f32>(0.5, 0.05, 0.02), vec3<f32>(0.0, 0.9, 0.7), ap_sub.x * 0.8);
            sss_accum = sss_accum + sub_glow * exp(-f32(s) * 0.18);
        }

        let lit = tissue_col * (diff * 0.75 + 0.25) * ao + sss_accum * 0.18;
        let spec = pow(max(dot(reflect(rd, n), light_dir), 0.0), 24.0) * 0.85;

        col = lit + vec3<f32>(1.0, 0.95, 0.85) * spec + vec3<f32>(0.2, 0.8, 1.0) * rim * 0.6 + accum_glow;
    } else {
        col = col + accum_glow * 1.4;
    }

    return vec4<f32>(col, 1.0);
}

// -------------------------------------------------------------
// MAIN COMPUTE DISPATCH
// -------------------------------------------------------------

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= u.width || y >= u.height) {
        return;
    }

    let aspect = f32(u.width) / f32(u.height);
    let uv_x = (f32(x) / f32(u.width) - 0.5) * 2.0 * aspect;
    let uv_y = (0.5 - f32(y) / f32(u.height)) * 2.0;

    let ro = u.cam_pos;
    let fwd = normalize(u.cam_dir);
    let right = normalize(cross(fwd, vec3<f32>(0.0, 1.0, 0.0)));
    let up = cross(right, fwd);
    let rd = normalize(fwd + right * uv_x + up * uv_y);

    var pixel_color = vec4<f32>(0.0, 0.0, 0.0, 1.0);

    if (u.scene_id == 0u) {
        pixel_color = render_waves(ro, rd);
    } else if (u.scene_id == 1u) {
        pixel_color = render_mandelbulb(uv_x, uv_y);
    } else if (u.scene_id == 2u) {
        pixel_color = render_blackhole(ro, rd);
    } else if (u.scene_id == 3u) {
        pixel_color = render_tornado(ro, rd);
    } else {
        pixel_color = render_cardiac(uv_x, uv_y);
    }

    let pixel_idx = y * u.width + x;
    output_pixels[pixel_idx] = pack_rgba(pixel_color);
}
"#;

pub const DOWNSAMPLE_SHADER: &str = r#"
struct DownsampleUniforms {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
};

@group(0) @binding(0) var<uniform> ds: DownsampleUniforms;
@group(0) @binding(1) var<storage, read> src_pixels: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst_pixels: array<u32>;

@compute @workgroup_size(16, 16)
fn downsample_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dx = id.x;
    let dy = id.y;
    if (dx >= ds.dst_width || dy >= ds.dst_height) {
        return;
    }
    let sx = (dx * ds.src_width) / ds.dst_width;
    let sy = (dy * ds.src_height) / ds.dst_height;
    let src_idx = sy * ds.src_width + sx;
    let dst_idx = dy * ds.dst_width + dx;
    dst_pixels[dst_idx] = src_pixels[src_idx];
}
"#;
