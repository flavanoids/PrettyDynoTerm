# Mathematical Specification: Shaders & Scenes

`dyno_term` features six distinct benchmark acts based on genuine physical, meteorological, astrophysical, and biomedical partial differential equations. Each act targets specific hardware execution paths (transcendental SFUs, warp/wavefront divergence, register pressure, memory bus bandwidth, and multi-core CPU IPC).

---

## Act I: "Atmospheric Ocean Waves" (GPU Compute Shader)
*Hardware Target: Hydrodynamic Gerstner Superposition, Central Finite Difference Gradients, Adiabatic Cloud Condensation & Radiative Transfer*

### Visual Concept
An endless oceanic seascape with bioluminescent cyan crests under an atmospheric twilight sky filled with convective cumulus clouds and forward-scattering crepuscular solar rays.

### Mathematical Formulation
1. **14-Octave Trochoidal Gerstner Waves**:
   The sea surface displacement $H(\vec{p}, t)$ is modeled as a superposition of 14 Gerstner wave octaves:
   $$H(\vec{p}, t) = \sum_{i=0}^{13} \left[ A_i \sin(\vec{k}_i \cdot \vec{p} + \omega_i t) + 0.38 A_i |\cos(\vec{k}_i \cdot \vec{p} + \omega_i t)| \right]$$
   where frequency $k_{i+1} = 1.64 k_i$, amplitude $A_{i+1} = 0.51 A_i$, and wave speed $\omega_{i+1} = 1.14 \omega_i$.
   Angular wave dispersion follows golden-ratio increments $\theta_i = i \times 1.396263 \text{ rad} \approx 80^\circ$ for non-repeating isotropic sea states.

2. **High-Precision 4-Tap Central Differences Normals**:
   The surface normal $\vec{n}$ is computed using symmetric 4-tap central differences:
   $$\vec{n} = \text{normalize}\left(-\frac{H(x+\epsilon, z) - H(x-\epsilon, z)}{2\epsilon}, 1.0, -\frac{H(x, z+\epsilon) - H(x, z-\epsilon)}{2\epsilon}\right)$$

3. **Adiabatic Cumulus Cloud Formation & Clausius-Clapeyron Thermodynamics**:
   To equalize GPU load across all 8.29 million pixels (preventing sky rays from idling), all sky-directed rays ($rd_y \ge -0.01$) march through 48 volumetric steps of 6-octave 3D fractal Brownian motion (fBm) cloud density with dual-phase Henyey-Greenstein Mie scattering ($g = 0.72$):
   $$P_{\text{HG}}(\cos\gamma, g) = \frac{1 - g^2}{4\pi (1 + g^2 - 2g \cos\gamma)^{3/2}}$$

4. **Secondary Water Reflection Rays**:
   Upon intersecting the water surface, a secondary reflection ray $\vec{r}_{refl} = \text{reflect}(\vec{rd}, \vec{n})$ evaluates ocean specular highlights and subsurface scattering.

---

## Act II: "The Mandelbulb Core" (GPU Compute Shader)
*Hardware Target: Hypercomplex Polynomial Algebra, Analytical Derivative Tracking, Cone-Traced AO & Soft Shadow Marching*

### Visual Concept
A 3D fractal Mandelbulb ($n = 8$) suspended in deep space. An orbiting camera glides around recursive fractal caverns, revealing iridescent metallic shells and glowing plasma interior magma chambers.

### Mathematical Formulation
1. **Order-8 Polynomial Iteration**:
   Spherical coordinate mapping:
   $$r = \sqrt{x^2 + y^2 + z^2}, \quad \theta = \arccos\left(\frac{z}{r}\right), \quad \phi = \arctan\left(\frac{y}{x}\right)$$
   $$\vec{z}_{k+1} = r^8 \begin{pmatrix} \sin(8\theta) \cos(8\phi) \\ \sin(8\theta) \sin(8\phi) \\ \cos(8\theta) \end{pmatrix} + \vec{c}$$

2. **Hubbard-Douady Potential Derivative & Distance Estimation**:
   The running derivative $dr$ is maintained across 24 unrolled loop iterations:
   $$dr_{k+1} = r^7 \cdot 8 \cdot dr_k + 1.0, \quad d = 0.5 \cdot \frac{r \ln(r)}{dr}$$

3. **Volumetric Aura & 10-Tap Cone-Traced Ambient Occlusion**:
   160 raymarching steps evaluate volumetric orbit-trap glow along the ray, coupled with 10-tap cone ambient occlusion:
   $$\text{AO} = \text{clamp}\left(1.0 - 3.8 \sum_{i=1}^{10} \frac{i \delta - \text{DE}(\vec{p} + \vec{n} i \delta)}{2^i}, 0.0, 1.0\right)$$

---

## Act III: "Relativistic Black Hole" (GPU Compute Shader)
*Hardware Target: 4th-Order Runge-Kutta (RK4) Numerical Integration, Kerr Metric Frame-Dragging & Novikov-Thorne Relativistic Disk*

### Visual Concept
A supermassive black hole with an accretion disk orbiting at relativistic speeds, surrounded by procedural starfield distortions produced by curved spacetime gravitational lensing.

### Mathematical Formulation
1. **4th-Order Runge-Kutta (RK4) Geodesic Integrator**:
   Light trajectories are integrated using a 4th-order Runge-Kutta solver over 80 steps:
   $$\vec{a}_{\text{geodesic}} = -\frac{3}{2} R_s \frac{\vec{p}}{\|\vec{p}\|^4} + 1.6 \frac{\vec{S}_{\text{Kerr}} \times \vec{p}}{\|\vec{p}\|^5}$$
   where $\vec{S}_{\text{Kerr}} = (0, 0.72, 0)$ implements Lense-Thirring frame-dragging.

2. **Novikov-Thorne Thin-Disk Emissivity & Liouville Relativistic Invariance**:
   For disk plasma crossing $y = 0$ between $r_{\text{in}} = 2.1 R_s$ and $r_{\text{out}} = 8.2 R_s$:
   $$F_{\text{NT}}(r) \propto \frac{1}{r^3} \left(1 - \sqrt{\frac{r_{\text{in}}}{r}}\right)$$
   Observed intensity follows curved-spacetime Liouville invariance:
   $$I_{\text{obs}} = g_{\text{total}}^4 \cdot F_{\text{NT}}(r), \quad g_{\text{total}} = \frac{\sqrt{1 - R_s / r}}{\gamma (1 - \vec{v} \cdot \vec{d})}$$

---

## Act IV: "Tornadic Mesocyclone" (GPU Compute Shader - Weather Model)
*Hardware Target: Fluid Dynamics Stream Functions, Burgers-Rott Vortex Vorticity, Cyclostrophic Pressure Balance & Mie Condensation*

### Visual Concept
A rotating tornadic supercell thunderstorm over the plains, complete with an adiabatic condensation funnel cloud, helical inflow bands, surface debris boundary layer, and distant lightning discharge potential.

### Mathematical Formulation
1. **Burgers-Rott Vortex Velocity Field**:
   The exact Navier-Stokes solution for an axisymmetric stretched vortex:
   $$v_\theta(r, y) = \frac{\Gamma}{2\pi r} \left(1 - \exp\left(-\frac{r^2}{r_c(y)^2}\right)\right)$$
   where core radius $r_c(y) = r_0 (1.0 + 0.32 y + 0.12 y^2)$ widens into a hyperboloid funnel cloud with altitude.

2. **Cyclostrophic Pressure Deficit & Condensation Isobar**:
   In an intense atmospheric vortex, cyclostrophic balance balances radial pressure gradient and centrifugal acceleration:
   $$\frac{1}{\rho}\frac{dp}{dr} = \frac{v_\theta^2}{r} \implies \Delta p(r, y) = -\frac{\Delta P_0}{1 + (r / r_c(y))^2}$$
   Air drawn into the core expands adiabatically. Condensation into visible cloud droplets occurs where ambient pressure drops below the local dew-point saturation pressure $P_{\text{dew}}$.

3. **Radiative Transfer & Surface Boundary Layer**:
   112 volumetric raymarch steps compute multi-scattering through cloud droplets and turbulent surface friction debris layers ($y < 0.8\text{ km}$).

---

## Act V: "Cardiac Electrophysiology" (GPU Compute Shader - Medical Model)
*Hardware Target: Reaction-Diffusion Partial Differential Equations, Anisotropic Myocardial Fibers & Optical Mapping Subsurface Scattering*

### Visual Concept
A beating human left ventricle afflicted with ventricular tachycardia / fibrillation. Fluorescent depolarization action potential wavefronts (cyan-mint, $+30\text{ mV}$) and repolarization refractory wakes (amber-gold) spiral around the thick-walled myocardial muscle.

### Mathematical Formulation
1. **Aliev-Panfilov 2-Variable Reaction-Diffusion System**:
   The gold-standard biophysical PDE model for myocardial tissue excitation and cardiac arrhythmia:
   $$\frac{\partial u}{\partial t} = \nabla \cdot (\mathbf{D} \nabla u) - k u(u - a)(u - 1) - u v$$
   $$\frac{\partial v}{\partial t} = \left(\epsilon + \frac{\mu_1 v}{u + \mu_2}\right) (-v - k u(u - b - 1))$$
   where $u$ is transmembrane action potential (depolarization) and $v$ is slow potassium/calcium recovery gating.

2. **Scroll-Wave Rotor Dynamics & Anisotropic Myocardial Pitch**:
   Simulates two interacting counter-rotating scroll-wave rotors across myocardial fibers that rotate by $\sim 120^\circ$ through the ventricle wall:
   $$\Phi_1 = \phi + \theta_{\text{fiber}}(y) - \omega_1 t + 1.65 \ln(r), \quad u = \frac{1}{1 + \exp(-12.0 \sin \Phi)}$$

3. **Transmural Subsurface Light Transport**:
   160 raymarching steps through the thick-walled ventricular ellipsoid SDF, coupled with 20 subsurface light penetration steps through myocardial tissue.

---

## Act VI: "Lattice Boltzmann Vortex Street" (GPU Compute Shader - CFD Model)
*Hardware Target: Incompressible Navier-Stokes Kinetics, D2Q9 Velocity Streaming, BGK Collision Relaxation & Vorticity Curl*

### Visual Concept
An aerodynamic flow channel inside an aerospace wind tunnel observing a circular bluff body obstacle shedding a periodic, staggered **von Kármán Vortex Street**. Alternating electric cyan (counter-clockwise) and fluorescent magenta (clockwise) swirling vortex cores drift downstream amidst golden streamline filaments and dark carbon-composite chamber boundary layers.

### Mathematical Formulation
1. **D2Q9 Velocity Lattice & Equilibrium Distribution**:
   The fluid velocity distribution is modeled across 9 discrete lattice velocities $\vec{c}_0 \dots \vec{c}_8$:
   $$f_i^{\text{eq}}(\rho, \vec{u}) = \rho w_i \left[ 1 + \frac{\vec{c}_i \cdot \vec{u}}{c_s^2} + \frac{(\vec{c}_i \cdot \vec{u})^2}{2 c_s^4} - \frac{\vec{u} \cdot \vec{u}}{2 c_s^2} \right]$$
   where $c_s^2 = 1/3$ is the lattice speed of sound, with weights $w_0 = 4/9$, $w_{1..4} = 1/9$, and $w_{5..8} = 1/36$.

2. **BGK Collision & Viscous Dissipation**:
   The Bhatnagar-Gross-Krook (BGK) collision operator computes local momentum relaxation towards equilibrium across all 9 directions:
   $$\Omega_i = -\frac{1}{\tau} \left(f_i - f_i^{\text{eq}}\right)$$
   where relaxation time $\tau = 0.65$ dictates the kinematic fluid viscosity $\nu = c_s^2 (\tau - 0.5)$.

3. **von Kármán Vortex Street Shedding ($Re \sim 250, St = 0.21$)**:
   Downstream wake instability produces periodic vortex shedding at Strouhal frequency $\omega_s$:
   $$\vec{u}_{\text{vortex}} = \sum_{k=1}^{6} \frac{\Gamma_k}{2\pi} \frac{(-(y - y_k), x - x_k)}{(x - x_k)^2 + (y - y_k)^2 + r_{\text{core}}^2}$$
   where staggered alternating circulation $\Gamma_k = (-1)^k \Gamma_0$ induces self-sustaining boundary layer separation.

4. **4-Tap Symmetric Central Differences Vorticity**:
   High-precision curl operator $\omega = \nabla \times \vec{u}$ isolates localized fluid rotation:
   $$\omega = \frac{u_y(x+\epsilon, y) - u_y(x-\epsilon, y)}{2\epsilon} - \frac{u_x(x, y+\epsilon) - u_x(x, y-\epsilon)}{2\epsilon}$$

---

## Act VII: "Kinetic Supernova" (CPU Multi-Threaded Physics)
*Hardware Target: CPU Multi-Core IPC, SIMD Vectorization & Cache Coherency*

### Concept & Formulation
A multi-threaded gravitational $N$-body collision simulating **32,000 celestial masses** interacting under Newtonian gravity, parallelized across all available hardware threads using **Rayon**.

1. **Gravitational Field Integration**:
   For particle $i$ interacting with particle $j$:
   $$\vec{F}_{ij} = \frac{G \cdot m_j \cdot (\vec{p}_j - \vec{p}_i)}{\left(|\vec{p}_j - \vec{p}_i|^2 + \epsilon^2\right)^{3/2}}$$
   where $\epsilon = 0.9$ is the gravitational softening length to prevent singular divergences.

2. **Kinetic Velocity Direct Preview Rasterization**:
   Particles are projected directly onto the preview buffer with velocity-based thermal color grading:
   - High speed ($v > 2.8$): Supernova white-gold (`#ffe664`)
   - Moderate speed ($v > 1.6$): Kinetic cyan (`#00f0ff`)
   - Base drift ($v < 1.0$): Deep galactic azure (`#3c82e6`)
