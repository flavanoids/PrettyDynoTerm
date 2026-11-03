use rayon::prelude::*;

#[derive(Clone, Copy)]
pub struct Particle {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub vx: f32,
    pub vy: f32,
    pub vz: f32,
    pub mass: f32,
}

pub struct NBodySimulation {
    pub particles: Vec<Particle>,
    pub width: usize,
    pub height: usize,
}

impl NBodySimulation {
    pub fn new(count: usize, width: usize, height: usize) -> Self {
        let mut particles = Vec::with_capacity(count);

        // Seed particles in two colliding spiral galaxies
        for i in 0..count {
            let arm = (i % 4) as f32 * (std::f32::consts::PI / 2.0);
            let r = ((i as f32 / count as f32).sqrt()) * 12.0 + 0.5;
            let theta = r * 1.8 + arm;

            let galaxy_offset_x = if i < count / 2 { -5.0 } else { 5.0 };
            let galaxy_offset_y = if i < count / 2 { 1.5 } else { -1.5 };
            let galaxy_dir = if i < count / 2 { 1.0 } else { -1.0 };

            let x = galaxy_offset_x + r * theta.cos();
            let y = galaxy_offset_y + r * theta.sin() * 0.4;
            let z = (i as f32 % 5.0 - 2.5) * 0.4;

            // Orbital velocity + drift towards collision
            let v_orbit = (7.0 / (r + 0.8)).sqrt();
            let vx = -theta.sin() * v_orbit * galaxy_dir + (galaxy_dir * -0.2);
            let vy = theta.cos() * v_orbit * 0.4 * galaxy_dir;
            let vz = 0.0;

            particles.push(Particle {
                x,
                y,
                z,
                vx,
                vy,
                vz,
                mass: 1.0 + (i % 7) as f32 * 0.5,
            });
        }

        Self {
            particles,
            width,
            height,
        }
    }

    /// Step simulation using all CPU threads via Rayon
    pub fn step_multithreaded(&mut self, dt: f32) {
        let n = self.particles.len();
        let eps2 = 0.9f32; // Softening factor

        // Parallel compute accelerations across all CPU cores
        let accelerations: Vec<(f32, f32, f32)> = self
            .particles
            .par_iter()
            .map(|p1| {
                let mut ax = 0.0f32;
                let mut ay = 0.0f32;
                let mut az = 0.0f32;

                // 256 Gravitational center points per particle
                let stride = (n / 256).max(1);
                for j in (0..n).step_by(stride) {
                    let p2 = unsafe { self.particles.get_unchecked(j) };
                    let dx = p2.x - p1.x;
                    let dy = p2.y - p1.y;
                    let dz = p2.z - p1.z;
                    let dist_sq = dx * dx + dy * dy + dz * dz + eps2;
                    let dist = dist_sq.sqrt();
                    let inv_cube = 1.0 / (dist_sq * dist);
                    let force = p2.mass * inv_cube;

                    ax += dx * force;
                    ay += dy * force;
                    az += dz * force;
                }

                (ax, ay, az)
            })
            .collect();

        // Integrate positions
        for (p, (ax, ay, az)) in self.particles.iter_mut().zip(accelerations.into_iter()) {
            p.vx += ax * dt;
            p.vy += ay * dt;
            p.vz += az * dt;
            p.x += p.vx * dt;
            p.y += p.vy * dt;
            p.z += p.vz * dt;
        }
    }

    /// Rasterize particles into an RGBA8 buffer for the terminal renderer
    #[allow(dead_code)]
    pub fn rasterize(&self, buffer: &mut [u8]) {
        self.rasterize_to_preview(buffer, self.width, self.height);
    }

    /// Rasterize particles directly to preview dimensions (avoids allocating and clearing 530MB framebuffers)
    pub fn rasterize_to_preview(&self, buffer: &mut [u8], width: usize, height: usize) {
        let expected_len = width * height * 4;
        let buf = if buffer.len() >= expected_len {
            &mut buffer[..expected_len]
        } else {
            buffer
        };

        // Clear buffer to dark cosmic obsidian
        for chunk in buf.chunks_exact_mut(4) {
            chunk[0] = 5;
            chunk[1] = 6;
            chunk[2] = 16;
            chunk[3] = 255;
        }

        let w = width as f32;
        let h = height as f32;
        let scale = (h * 0.42).min(w * 0.32);
        let center_x = w * 0.5;
        let center_y = h * 0.5;

        for p in &self.particles {
            let px = (center_x + p.x * scale) as isize;
            let py = (center_y + p.y * scale) as isize;

            if px >= 0 && px < width as isize && py >= 0 && py < height as isize {
                let idx = ((py as usize) * width + (px as usize)) * 4;
                let speed = (p.vx * p.vx + p.vy * p.vy + p.vz * p.vz).sqrt();

                // Velocity color grading: Cyan for high kinetic speed, amber/magenta for core
                let (r, g, b) = if speed > 2.8 {
                    (255, 230, 100) // Supernova white-gold
                } else if speed > 1.6 {
                    (0, 240, 255) // Kinetic cyan
                } else if speed > 0.9 {
                    (190, 50, 255) // Neon violet
                } else {
                    (60, 130, 230) // Deep azure
                };

                buf[idx] = buf[idx].saturating_add(r);
                buf[idx + 1] = buf[idx + 1].saturating_add(g);
                buf[idx + 2] = buf[idx + 2].saturating_add(b);
            }
        }
    }
}
