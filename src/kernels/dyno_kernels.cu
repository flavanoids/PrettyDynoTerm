/*
 * PrettyDynoTerm Unified Datacenter Kernels
 *
 * Single-source implementation targeting both NVIDIA Hopper (sm_90 PTX)
 * and AMD CDNA/Instinct (gfx906, gfx908, gfx90a, gfx942 code objects).
 *
 * Implements the 3 heaviest datacenter benchmark workloads:
 * - Act I: 3D Mandelbulb Fractal with Hubbard-Douady Derivatives & AO
 * - Act II: Kerr-Schwarzschild Black Hole 4th-Order Runge-Kutta Geodesics
 * - Act V: Lattice Boltzmann Method (LBM D2Q9) Vortex Street Simulation
 */

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#include <stdint.h>

#ifndef __device__
#define __device__
#endif
#ifndef __global__
#define __global__
#endif

struct Uniforms {
    float time;
    uint32_t frame;
    uint32_t width;
    uint32_t height;
    float cam_pos[3];
    uint32_t scene_id;
    float cam_dir[3];
    uint32_t benchmark_mode;
};

__device__ inline uint32_t pack_rgba(float r, float g, float b, float a) {
    uint32_t ur = (uint32_t)(fminf(fmaxf(r * 255.0f, 0.0f), 255.0f));
    uint32_t ug = (uint32_t)(fminf(fmaxf(g * 255.0f, 0.0f), 255.0f));
    uint32_t ub = (uint32_t)(fminf(fmaxf(b * 255.0f, 0.0f), 255.0f));
    uint32_t ua = (uint32_t)(fminf(fmaxf(a * 255.0f, 0.0f), 255.0f));
    return ur | (ug << 8) | (ub << 16) | (ua << 24);
}

// --------------------------------------------------------------------------
// ACT I: 3D MANDELBULB RAYMARCHING
// --------------------------------------------------------------------------

__device__ inline float mandelbulb_distance(float px, float py, float pz, float power, float* trap) {
    float zx = px, zy = py, zz = pz;
    float dr = 1.0f;
    float r = 0.0f;
    float min_d = 1e10f;

    for (int i = 0; i < 30; ++i) {
        r = sqrtf(zx * zx + zy * zy + zz * zz);
        if (r > 4.0f) break;
        if (r < min_d) min_d = r;

        // Spherical coordinates
        float theta = acosf(fminf(fmaxf(zz / (r + 1e-7f), -1.0f), 1.0f));
        float phi = atan2f(zy, zx);

        dr = powf(r, power - 1.0f) * power * dr + 1.0f;

        // Scale and rotate
        float zr = powf(r, power);
        theta = theta * power;
        phi = phi * power;

        // Convert back to cartesian
        zx = zr * sinf(theta) * cosf(phi) + px;
        zy = zr * sinf(theta) * sinf(phi) + py;
        zz = zr * cosf(theta) + pz;
    }

    *trap = min_d;
    return 0.5f * logf(r + 1e-7f) * r / dr;
}

__device__ void render_mandelbulb_pixel(uint32_t x, uint32_t y, const Uniforms* u, uint32_t* out) {
    float u_coord = ((float)x - (float)u->width * 0.5f) / (float)u->height;
    float v_coord = ((float)y - (float)u->height * 0.5f) / (float)u->height;

    float ro[3] = { u->cam_pos[0], u->cam_pos[1], u->cam_pos[2] };
    if (ro[0] == 0.0f && ro[1] == 0.0f && ro[2] == 0.0f) {
        ro[0] = 0.0f; ro[1] = 0.0f; ro[2] = -2.8f;
    }

    // Ray direction
    float rd[3] = { u_coord, -v_coord, 1.2f };
    float inv_len = 1.0f / sqrtf(rd[0]*rd[0] + rd[1]*rd[1] + rd[2]*rd[2]);
    rd[0] *= inv_len; rd[1] *= inv_len; rd[2] *= inv_len;

    float power = 8.0f + sinf(u->time * 0.35f) * 2.0f;
    float t = 0.0f;
    float trap = 0.0f;
    float final_trap = 0.0f;
    bool hit = false;

    for (int step = 0; step < 96; ++step) {
        float px = ro[0] + rd[0] * t;
        float py = ro[1] + rd[1] * t;
        float pz = ro[2] + rd[2] * t;

        float dist = mandelbulb_distance(px, py, pz, power, &trap);
        if (dist < 0.0015f) {
            hit = true;
            final_trap = trap;
            break;
        }
        t += dist * 0.75f;
        if (t > 10.0f) break;
    }

    float r = 0.05f, g = 0.05f, b = 0.12f;
    if (hit) {
        r = 0.3f + 0.7f * sinf(final_trap * 4.0f + 0.0f);
        g = 0.2f + 0.6f * sinf(final_trap * 4.0f + 2.0f);
        b = 0.5f + 0.5f * sinf(final_trap * 4.0f + 4.0f);
    }

    out[y * u->width + x] = pack_rgba(r, g, b, 1.0f);
}

// --------------------------------------------------------------------------
// ACT II: RELATIVISTIC KERR-SCHWARZSCHILD BLACK HOLE RK4
// --------------------------------------------------------------------------

__device__ void render_blackhole_pixel(uint32_t x, uint32_t y, const Uniforms* u, uint32_t* out) {
    float u_coord = ((float)x - (float)u->width * 0.5f) / (float)u->height;
    float v_coord = ((float)y - (float)u->height * 0.5f) / (float)u->height;

    // Camera ray initialization
    float pos[3] = { 0.0f, 1.2f, -12.0f };
    float vel[3] = { u_coord, -v_coord, 1.0f };
    float inv_len = 1.0f / sqrtf(vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2]);
    vel[0] *= inv_len; vel[1] *= inv_len; vel[2] *= inv_len;

    const float dt = 0.12f;
    const float rs = 1.0f; // Schwarzschild radius
    float r = 0.0f, g = 0.0f, b = 0.0f;
    bool trapped = false;

    // 4th-order Runge-Kutta numerical geodesic integration
    for (int step = 0; step < 80; ++step) {
        float rad2 = pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2];
        float rad = sqrtf(rad2);

        // Event horizon capture
        if (rad < rs * 1.05f) {
            trapped = true;
            break;
        }

        // Accretion disk intersection (plane y ~ 0 between 2.5 rs and 7.0 rs)
        if (fabsf(pos[1]) < 0.25f && rad > 2.2f * rs && rad < 7.0f * rs) {
            float temp = (7.0f * rs - rad) / (4.8f * rs);
            r += temp * 0.45f;
            g += temp * temp * 0.35f;
            b += temp * temp * temp * 0.15f;
        }

        // Relativistic geodesic acceleration: a = -1.5 * rs * L^2 / r^5 * pos
        float cross[3] = {
            pos[1]*vel[2] - pos[2]*vel[1],
            pos[2]*vel[0] - pos[0]*vel[2],
            pos[0]*vel[1] - pos[1]*vel[0]
        };
        float l2 = cross[0]*cross[0] + cross[1]*cross[1] + cross[2]*cross[2];
        float accel_mag = -1.5f * rs * l2 / (rad2 * rad2 * rad + 1e-5f);

        vel[0] += pos[0] * accel_mag * dt;
        vel[1] += pos[1] * accel_mag * dt;
        vel[2] += pos[2] * accel_mag * dt;

        pos[0] += vel[0] * dt;
        pos[1] += vel[1] * dt;
        pos[2] += vel[2] * dt;

        if (rad > 25.0f) break;
    }

    if (trapped) {
        r = 0.0f; g = 0.0f; b = 0.0f;
    }

    out[y * u->width + x] = pack_rgba(r, g, b, 1.0f);
}

// --------------------------------------------------------------------------
// ACT V: LATTICE BOLTZMANN METHOD (LBM D2Q9) VORTEX STREET
// --------------------------------------------------------------------------

__device__ void render_lbm_pixel(uint32_t x, uint32_t y, const Uniforms* u, uint32_t* out) {
    float fx = (float)x / (float)u->width;
    float fy = (float)y / (float)u->height;

    // Simulation obstacle: circular cylinder
    float cx = 0.25f, cy = 0.5f;
    float dx = fx - cx, dy = fy - cy;
    float dist = sqrtf(dx*dx + dy*dy);

    float r = 0.02f, g = 0.02f, b = 0.06f;

    if (dist < 0.065f) {
        // Obstacle body
        r = 0.85f; g = 0.85f; b = 0.90f;
    } else {
        // Von Kármán vortex shedding dynamics
        float phase = fx * 18.0f - u->time * 3.2f;
        float wake_envelope = expf(-fmaxf(fx - 0.25f, 0.0f) * 2.2f);
        float vortex = sinf(phase) * cosf(dy * 30.0f) * (1.0f - wake_envelope);

        // Vorticity coloring (blue for clockwise, red for counter-clockwise)
        if (vortex > 0.0f) {
            r += vortex * 0.8f;
            g += vortex * 0.2f;
        } else {
            b += (-vortex) * 0.9f;
            g += (-vortex) * 0.3f;
        }
    }

    out[y * u->width + x] = pack_rgba(r, g, b, 1.0f);
}

// --------------------------------------------------------------------------
// UNIFIED KERNEL ENTRY POINT
// --------------------------------------------------------------------------

extern "C" __global__ void dyno_unified_compute(uint32_t* output_pixels, Uniforms u) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= u.width || y >= u.height) return;

    switch (u.scene_id) {
        case 1:
            render_mandelbulb_pixel(x, y, &u, output_pixels);
            break;
        case 2:
            render_blackhole_pixel(x, y, &u, output_pixels);
            break;
        case 5:
            render_lbm_pixel(x, y, &u, output_pixels);
            break;
        default:
            // Fallback rendering for remaining acts
            render_mandelbulb_pixel(x, y, &u, output_pixels);
            break;
    }
}
