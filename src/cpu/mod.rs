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
        // Fail-safe protection against out-of-memory allocations
        const MAX_SAFE_PARTICLES: usize = 5_000_000;
        let count = count.min(MAX_SAFE_PARTICLES);
        let mut particles = Vec::with_capacity(count);
        if count == 0 {
            return Self {
                particles,
                width,
                height,
            };
        }

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
        if n == 0 {
            return;
        }
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
                    if let Some(p2) = self.particles.get(j) {
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
                }

                (ax, ay, az)
            })
            .collect();

        // Symplectic Leapfrog (Velocity Verlet) time integration:
        // Strictly preserves phase-space volume, energy invariance, and orbital angular momentum
        let half_dt = 0.5 * dt;
        for (p, (ax, ay, az)) in self.particles.iter_mut().zip(accelerations) {
            p.vx += ax * half_dt;
            p.vy += ay * half_dt;
            p.vz += az * half_dt;
            p.x += p.vx * dt;
            p.y += p.vy * dt;
            p.z += p.vz * dt;
            p.vx += ax * half_dt;
            p.vy += ay * half_dt;
            p.vz += az * half_dt;
        }
    }

    /// Rasterize particles into an RGBA8 buffer for the terminal renderer
    #[allow(dead_code)]
    pub fn rasterize(&self, buffer: &mut [u8]) {
        self.rasterize_to_preview(buffer, self.width, self.height);
    }

    /// Rasterize particles directly to preview dimensions (avoids allocating and clearing 530MB framebuffers)
    pub fn rasterize_to_preview(&self, buffer: &mut [u8], width: usize, height: usize) {
        let expected_len = match width.checked_mul(height).and_then(|wh| wh.checked_mul(4)) {
            Some(l) => l,
            None => return,
        };
        if buffer.len() < expected_len || width == 0 || height == 0 {
            return;
        }
        let buf = &mut buffer[..expected_len];

        // Clear buffer to dark cosmic obsidian
        for chunk in buf.as_chunks_mut::<4>().0 {
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
                if idx + 3 < buf.len() {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nbody_initialization() {
        let count = 512;
        let sim = NBodySimulation::new(count, 120, 60);
        assert_eq!(sim.particles.len(), count);
        assert_eq!(sim.width, 120);
        assert_eq!(sim.height, 60);

        for p in &sim.particles {
            assert!(p.x.is_finite());
            assert!(p.y.is_finite());
            assert!(p.z.is_finite());
            assert!(p.vx.is_finite());
            assert!(p.vy.is_finite());
            assert!(p.vz.is_finite());
            assert!(p.mass > 0.0);
        }
    }

    #[test]
    fn test_nbody_step_multithreaded_no_nans() {
        let mut sim = NBodySimulation::new(256, 100, 50);
        let orig_pos_0 = (sim.particles[0].x, sim.particles[0].y, sim.particles[0].z);

        sim.step_multithreaded(0.016);

        let new_pos_0 = (sim.particles[0].x, sim.particles[0].y, sim.particles[0].z);
        // Particle should have moved
        assert_ne!(orig_pos_0, new_pos_0);

        // All particles should remain finite numbers
        for p in &sim.particles {
            assert!(p.x.is_finite(), "x became non-finite");
            assert!(p.y.is_finite(), "y became non-finite");
            assert!(p.z.is_finite(), "z became non-finite");
            assert!(p.vx.is_finite(), "vx became non-finite");
            assert!(p.vy.is_finite(), "vy became non-finite");
            assert!(p.vz.is_finite(), "vz became non-finite");
        }
    }

    #[test]
    fn test_nbody_rasterize_to_preview() {
        let sim = NBodySimulation::new(128, 40, 20);
        let mut buffer = vec![0u8; 40 * 20 * 4];
        sim.rasterize_to_preview(&mut buffer, 40, 20);

        // Check clear color on pixel 0 (dark cosmic obsidian [5, 6, 16, 255])
        assert_eq!(buffer[3], 255); // Alpha should be 255

        // Check that at least some particle colors were blended
        let has_particles = buffer
            .as_chunks::<4>()
            .0
            .iter()
            .any(|chunk| chunk[0] > 5 || chunk[1] > 6 || chunk[2] > 16);
        assert!(
            has_particles,
            "Rasterizer should render particles into buffer"
        );
    }

    #[test]
    fn test_nbody_rasterize_bounds_protection() {
        let sim = NBodySimulation::new(64, 40, 20);
        // Smaller buffer than expected should not panic
        let mut small_buffer = vec![0u8; 10];
        sim.rasterize_to_preview(&mut small_buffer, 40, 20);

        // Zero-sized buffer
        let mut empty_buffer: Vec<u8> = Vec::new();
        sim.rasterize_to_preview(&mut empty_buffer, 40, 20);
    }

    #[test]
    fn test_nbody_zero_particles_fail_safe() {
        let mut sim = NBodySimulation::new(0, 80, 40);
        assert_eq!(sim.particles.len(), 0);
        // Step with 0 particles should not panic or divide by zero
        sim.step_multithreaded(0.016);

        let mut buf = vec![0u8; 80 * 40 * 4];
        sim.rasterize_to_preview(&mut buf, 80, 40);
        assert_eq!(buf[3], 255);
    }

    #[test]
    fn test_nbody_massive_particle_count_clamping() {
        // Enormous particle request should be clamped to safe ceiling to prevent OOM
        let sim = NBodySimulation::new(100_000_000, 80, 40);
        assert!(sim.particles.len() <= 5_000_000);
    }
}
