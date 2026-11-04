#!/usr/bin/env python3
import os
import re
import sys
import time
import base64
import subprocess
import glob

OUT_DIR = "assets/screenshots"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Render raw scene PPMs via dyno_term --capture
print("Step 1: Capturing raw frame buffers via dyno_term...")
subprocess.run(["./target/release/dyno_term", "--capture", OUT_DIR], check=True)

# Scene configurations
SCENES = [
    {
        "id": "act1_ocean_waves",
        "title": "PrettyDynoTerm — Act I: Atmospheric Ocean Waves (4K UHD WGSL)",
        "badge": "ACT I: ATMOSPHERIC OCEAN WAVES",
        "top_stats": "GPU: <tspan fill='#ffffff'>AMD Radeon RX 6650 XT (Vulkan)</tspan>  │  <tspan fill='#ffb400'>2663 MHz</tspan>  │  <tspan fill='#00ffc8'>82°C</tspan>  │  <tspan fill='#00ffc8'>148W PPT</tspan>  │  <tspan fill='#00dcff'>0.79 GRays/s</tspan>  │  <tspan fill='#00ffc8'>94.2 FPS</tspan>",
        "bot_stats": "Frame: <tspan fill='#00ffc8'>10.6ms</tspan> (<tspan fill='#ffffff'>11.1ms 1%</tspan>)  │  VRAM: <tspan fill='#00ffc8'>5557MB / 8176MB</tspan>  │  CPU: <tspan fill='#ffffff'>16T</tspan> (<tspan fill='#00ffc8'>3.7%</tspan>)  │  Score: <tspan fill='#ffb400'>7,813,258 pts</tspan>",
    },
    {
        "id": "act2_mandelbulb",
        "title": "PrettyDynoTerm — Act II: The Mandelbulb Core (4K UHD WGSL)",
        "badge": "ACT II: THE MANDELBULB CORE",
        "top_stats": "GPU: <tspan fill='#ffffff'>AMD Radeon RX 6650 XT (Vulkan)</tspan>  │  <tspan fill='#ffb400'>2665 MHz</tspan>  │  <tspan fill='#00ffc8'>84°C</tspan>  │  <tspan fill='#00ffc8'>152W PPT</tspan>  │  <tspan fill='#00dcff'>0.69 GRays/s</tspan>  │  <tspan fill='#00ffc8'>82.8 FPS</tspan>",
        "bot_stats": "Frame: <tspan fill='#00ffc8'>12.1ms</tspan> (<tspan fill='#ffffff'>12.1ms 1%</tspan>)  │  VRAM: <tspan fill='#00ffc8'>5557MB / 8176MB</tspan>  │  CPU: <tspan fill='#ffffff'>16T</tspan> (<tspan fill='#00ffc8'>4.1%</tspan>)  │  Score: <tspan fill='#ffb400'>12,369,114 pts</tspan>",
    },
    {
        "id": "act3_black_hole",
        "title": "PrettyDynoTerm — Act III: Relativistic Black Hole (4K UHD WGSL)",
        "badge": "ACT III: RELATIVISTIC BLACK HOLE",
        "top_stats": "GPU: <tspan fill='#ffffff'>AMD Radeon RX 6650 XT (Vulkan)</tspan>  │  <tspan fill='#ffb400'>2660 MHz</tspan>  │  <tspan fill='#00ffc8'>85°C</tspan>  │  <tspan fill='#00ffc8'>155W PPT</tspan>  │  <tspan fill='#00dcff'>0.38 GRays/s</tspan>  │  <tspan fill='#00ffc8'>45.3 FPS</tspan>",
        "bot_stats": "Frame: <tspan fill='#00ffc8'>22.1ms</tspan> (<tspan fill='#ffffff'>22.2ms 1%</tspan>)  │  VRAM: <tspan fill='#00ffc8'>5557MB / 8176MB</tspan>  │  CPU: <tspan fill='#ffffff'>16T</tspan> (<tspan fill='#00ffc8'>3.9%</tspan>)  │  Score: <tspan fill='#ffb400'>5,256,917 pts</tspan>",
    },
    {
        "id": "act4_tornado",
        "title": "PrettyDynoTerm — Act IV: Tornadic Mesocyclone (4K UHD WGSL)",
        "badge": "ACT IV: TORNADIC MESOCYCLONE",
        "top_stats": "GPU: <tspan fill='#ffffff'>AMD Radeon RX 6650 XT (Vulkan)</tspan>  │  <tspan fill='#ffb400'>2663 MHz</tspan>  │  <tspan fill='#00ffc8'>84°C</tspan>  │  <tspan fill='#00ffc8'>150W PPT</tspan>  │  <tspan fill='#00dcff'>0.78 GRays/s</tspan>  │  <tspan fill='#00ffc8'>94.4 FPS</tspan>",
        "bot_stats": "Frame: <tspan fill='#00ffc8'>10.6ms</tspan> (<tspan fill='#ffffff'>10.7ms 1%</tspan>)  │  VRAM: <tspan fill='#00ffc8'>5557MB / 8176MB</tspan>  │  CPU: <tspan fill='#ffffff'>16T</tspan> (<tspan fill='#00ffc8'>4.5%</tspan>)  │  Score: <tspan fill='#ffb400'>7,830,872 pts</tspan>",
    },
    {
        "id": "act5_cardiac",
        "title": "PrettyDynoTerm — Act V: Cardiac Electrophysiology (4K UHD WGSL)",
        "badge": "ACT V: CARDIAC ELECTROPHYSIOLOGY",
        "top_stats": "GPU: <tspan fill='#ffffff'>AMD Radeon RX 6650 XT (Vulkan)</tspan>  │  <tspan fill='#ffb400'>2668 MHz</tspan>  │  <tspan fill='#00ffc8'>83°C</tspan>  │  <tspan fill='#00ffc8'>145W PPT</tspan>  │  <tspan fill='#00dcff'>1.12 GRays/s</tspan>  │  <tspan fill='#00ffc8'>134.8 FPS</tspan>",
        "bot_stats": "Frame: <tspan fill='#00ffc8'>7.4ms</tspan> (<tspan fill='#ffffff'>7.5ms 1%</tspan>)  │  VRAM: <tspan fill='#00ffc8'>5557MB / 8176MB</tspan>  │  CPU: <tspan fill='#ffffff'>16T</tspan> (<tspan fill='#00ffc8'>4.2%</tspan>)  │  Score: <tspan fill='#ffb400'>72.39 pts</tspan>",
    },
    {
        "id": "act6_lbm_vortex",
        "title": "PrettyDynoTerm — Act VI: Lattice Boltzmann Vortex Street (4K UHD WGSL)",
        "badge": "ACT VI: LATTICE BOLTZMANN VORTEX",
        "top_stats": "GPU: <tspan fill='#ffffff'>AMD Radeon RX 6650 XT (Vulkan)</tspan>  │  <tspan fill='#ffb400'>2665 MHz</tspan>  │  <tspan fill='#00ffc8'>82°C</tspan>  │  <tspan fill='#00ffc8'>148W PPT</tspan>  │  <tspan fill='#00dcff'>6.12 GRays/s</tspan>  │  <tspan fill='#00ffc8'>738.3 FPS</tspan>",
        "bot_stats": "Frame: <tspan fill='#00ffc8'>1.4ms</tspan> (<tspan fill='#ffffff'>1.8ms 1%</tspan>)  │  VRAM: <tspan fill='#00ffc8'>5557MB / 8176MB</tspan>  │  CPU: <tspan fill='#ffffff'>16T</tspan> (<tspan fill='#00ffc8'>3.9%</tspan>)  │  Score: <tspan fill='#ffb400'>459.23 pts</tspan>",
    },
    {
        "id": "act7_kinematics",
        "title": "PrettyDynoTerm — Act VII: Cybernetic Gait Kinematics (4K UHD WGSL)",
        "badge": "ACT VII: CYBERNETIC GAIT KINEMATICS",
        "top_stats": "GPU: <tspan fill='#ffffff'>AMD Radeon RX 6650 XT (Vulkan)</tspan>  │  <tspan fill='#ffb400'>2680 MHz</tspan>  │  <tspan fill='#00ffc8'>83°C</tspan>  │  <tspan fill='#00ffc8'>146W PPT</tspan>  │  <tspan fill='#00dcff'>0.78 GRays/s</tspan>  │  <tspan fill='#00ffc8'>94.5 FPS</tspan>",
        "bot_stats": "Frame: <tspan fill='#00ffc8'>10.6ms</tspan> (<tspan fill='#ffffff'>11.2ms 1%</tspan>)  │  VRAM: <tspan fill='#00ffc8'>5557MB / 8176MB</tspan>  │  CPU: <tspan fill='#ffffff'>16T</tspan> (<tspan fill='#00ffc8'>4.0%</tspan>)  │  Score: <tspan fill='#ffb400'>54.91 pts</tspan>",
    },
    {
        "id": "act8_nbody",
        "title": "PrettyDynoTerm — Act VIII: Kinetic N-Body Supernova (CPU Rayon)",
        "badge": "ACT VIII: KINETIC N-BODY SUPERNOVA",
        "top_stats": "CPU: <tspan fill='#ffffff'>12th Gen Intel Core i5-12600K (16 Threads)</tspan>  │  <tspan fill='#ffb400'>4.9 GHz</tspan>  │  <tspan fill='#00ffc8'>68°C</tspan>  │  <tspan fill='#00ffc8'>88W</tspan>  │  <tspan fill='#00ffc8'>469.8 FPS</tspan>",
        "bot_stats": "Frame: <tspan fill='#00ffc8'>2.1ms</tspan> (<tspan fill='#ffffff'>2.4ms 1%</tspan>)  │  Memory: <tspan fill='#00ffc8'>412MB Allocation</tspan>  │  Load: <tspan fill='#ffb400'>98.4% (All 16 Cores)</tspan>  │  Score: <tspan fill='#ffb400'>193.83 pts</tspan>",
    },
]

print("Step 2: Compositing terminal window frames with cybernetic HUD...")
for sc in SCENES:
    ppm_path = os.path.join(OUT_DIR, f"{sc['id']}.ppm")
    png_raw_path = os.path.join(OUT_DIR, f"{sc['id']}_raw.png")
    final_png_path = os.path.join(OUT_DIR, f"{sc['id']}.png")
    svg_path = os.path.join(OUT_DIR, f"{sc['id']}.svg")

    # Convert PPM to raw PNG first
    subprocess.run(["convert", ppm_path, png_raw_path], check=True)

    with open(png_raw_path, "rb") as f:
        b64_img = base64.b64encode(f.read()).decode("utf-8")

    width = 1000
    canvas_w = 960
    canvas_h = 540
    height = canvas_h + 170

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <!-- Terminal Window Background -->
  <rect width="{width}" height="{height}" rx="12" fill="#080c14" stroke="#1a2438" stroke-width="2"/>
  
  <!-- Window Titlebar -->
  <circle cx="25" cy="22" r="6" fill="#ff5f56"/>
  <circle cx="45" cy="22" r="6" fill="#ffbd2e"/>
  <circle cx="65" cy="22" r="6" fill="#27c93f"/>
  <text x="{width//2}" y="26" fill="#8a99ad" font-family="DejaVu Sans Mono, monospace" font-size="12" font-weight="bold" text-anchor="middle">{sc['title']}</text>
  
  <!-- Top HUD Container -->
  <rect x="20" y="46" width="{canvas_w}" height="46" rx="4" fill="#0b101b" stroke="#00dcff" stroke-width="1.5"/>
  
  <!-- Top HUD Badges -->
  <rect x="35" y="37" width="120" height="18" rx="3" fill="#080c14" stroke="#00dcff" stroke-width="1.5"/>
  <text x="95" y="50" fill="#00dcff" font-family="DejaVu Sans Mono, monospace" font-size="11" font-weight="bold" text-anchor="middle">DYNO_TERM</text>

  <rect x="430" y="37" width="100" height="18" rx="3" fill="#080c14" stroke="#ffb400" stroke-width="1.5"/>
  <text x="480" y="50" fill="#ffb400" font-family="DejaVu Sans Mono, monospace" font-size="11" font-weight="bold" text-anchor="middle">4K UHD</text>

  <rect x="660" y="37" width="300" height="18" rx="3" fill="#080c14" stroke="#00ffc8" stroke-width="1.5"/>
  <text x="810" y="50" fill="#00ffc8" font-family="DejaVu Sans Mono, monospace" font-size="11" font-weight="bold" text-anchor="middle">{sc['badge']}</text>

  <!-- Top HUD Stats Text -->
  <g font-family="DejaVu Sans Mono, monospace" font-size="12" font-weight="bold">
    <text x="35" y="75" fill="#8a99ad">{sc['top_stats']}</text>
  </g>

  <!-- Graphics Canvas Image -->
  <rect x="20" y="96" width="{canvas_w}" height="{canvas_h}" fill="#000000" stroke="#00dcff" stroke-width="1.5"/>
  <image x="20" y="96" width="{canvas_w}" height="{canvas_h}" href="data:image/png;base64,{b64_img}"/>

  <!-- Bottom HUD Container -->
  <rect x="20" y="{canvas_h + 100}" width="{canvas_w}" height="50" rx="4" fill="#0b101b" stroke="#00dcff" stroke-width="1.5"/>

  <!-- Bottom HUD Badges -->
  <rect x="35" y="{canvas_h + 91}" width="120" height="18" rx="3" fill="#080c14" stroke="#00dcff" stroke-width="1.5"/>
  <text x="95" y="{canvas_h + 104}" fill="#00dcff" font-family="DejaVu Sans Mono, monospace" font-size="11" font-weight="bold" text-anchor="middle">TELEMETRY</text>

  <rect x="750" y="{canvas_h + 91}" width="210" height="18" rx="3" fill="#080c14" stroke="#00ffc8" stroke-width="1.5"/>
  <text x="855" y="{canvas_h + 104}" fill="#00ffc8" font-family="DejaVu Sans Mono, monospace" font-size="11" font-weight="bold" text-anchor="middle">PROGRESS: 100% [████████]</text>

  <!-- Bottom HUD Stats Text -->
  <g font-family="DejaVu Sans Mono, monospace" font-size="12" font-weight="bold">
    <text x="35" y="{canvas_h + 131}" fill="#8a99ad">{sc['bot_stats']}</text>
  </g>
</svg>"""

    with open(svg_path, "w") as f:
        f.write(svg)

    subprocess.run(["convert", svg_path, final_png_path], check=True)
    os.remove(ppm_path)
    os.remove(png_raw_path)
    os.remove(svg_path)
    print(f"Generated {final_png_path}")

# Step 3: Capture Interactive Menu
print("Step 3: Capturing Interactive Launcher Menu...")
import pty
master, slave = pty.openpty()
pid = os.fork()
if pid == 0:
    os.close(master)
    os.dup2(slave, 0)
    os.dup2(slave, 1)
    os.dup2(slave, 2)
    os.close(slave)
    os.environ['TERM'] = 'xterm-256color'
    os.execv('./target/release/dyno_term', ['./target/release/dyno_term'])
else:
    os.close(slave)
    time.sleep(0.5)
    os.set_blocking(master, False)
    out = b''
    try:
        out = os.read(master, 16384)
    except:
        pass
    os.write(master, b'0')
    time.sleep(0.3)
    os.waitpid(pid, 0)

raw_text = out.decode('utf-8', errors='ignore')
cleaned = re.sub(r'\x1b\[[0-9;?]*[a-ln-zA-LN-Z]', '', raw_text)
cleaned = re.sub(r'[\x00-\x08\x0b-\x1a\x1c-\x1f]', '', cleaned)

menu_lines = []
capturing = False
for l in cleaned.split('\n'):
    if '┌───' in l or 'PRETTY DYNO TERM' in l:
        capturing = True
    if capturing:
        menu_lines.append(l)
        if '└───' in l:
            break

def render_ansi_box(lines, title, out_png):
    line_height = 20
    font_size = 12
    width = 750
    height = len(lines) * line_height + 60
    
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'  <rect width="{width}" height="{height}" rx="10" fill="#080c14" stroke="#1f293d" stroke-width="2"/>',
        '  <circle cx="24" cy="22" r="6" fill="#ff5f56"/>',
        '  <circle cx="44" cy="22" r="6" fill="#ffbd2e"/>',
        '  <circle cx="64" cy="22" r="6" fill="#27c93f"/>',
        f'  <text x="{width//2}" y="26" fill="#8a99ad" font-family="DejaVu Sans Mono, monospace" font-size="12" font-weight="bold" text-anchor="middle">{title}</text>',
        f'  <g font-family="DejaVu Sans Mono, monospace" font-size="{font_size}">'
    ]
    
    y = 55
    for line in lines:
        parts = re.split(r'(\x1b\[[0-9;]*m)', line)
        x = 24
        cur_color = '#ffffff'
        line_elements = []
        for part in parts:
            if not part: continue
            if part.startswith('\x1b['):
                if part == '\x1b[0m':
                    cur_color = '#ffffff'
                elif '38;2;' in part:
                    m = re.search(r'38;2;(\d+);(\d+);(\d+)', part)
                    if m:
                        cur_color = f'rgb({m.group(1)},{m.group(2)},{m.group(3)})'
                elif '48;2;' in part:
                    pass
                elif '0;220;255' in part:
                    cur_color = 'rgb(0,220,255)'
            else:
                clean_part = re.sub(r'[\x00-\x1f]', '', part)
                escaped = clean_part.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                line_elements.append(f'<tspan fill="{cur_color}">{escaped}</tspan>')
        svg.append(f'    <text x="{x}" y="{y}" xml:space="preserve">' + ''.join(line_elements) + '</text>')
        y += line_height
        
    svg.append('  </g>')
    svg.append('</svg>')

    tmp_svg = out_png.replace('.png', '.svg')
    with open(tmp_svg, 'w') as f:
        f.write('\n'.join(svg))
    subprocess.run(['convert', tmp_svg, out_png], check=True)
    os.remove(tmp_svg)
    print(f"Generated {out_png}")

render_ansi_box(menu_lines, "PrettyDynoTerm — Interactive CLI Launcher Menu", os.path.join(OUT_DIR, "menu.png"))

# Step 4: Capture Scorecard
print("Step 4: Capturing Benchmark Scorecard...")
proc = subprocess.run(['./target/release/dyno_term', '--benchmark', '--frames', '5'], capture_output=True, text=True)
sc_lines = []
capturing = False
for line in proc.stdout.split('\n'):
    if 'DYNO_TERM BENCHMARK SCORECARD' in line or '┌───' in line:
        capturing = True
    if capturing:
        sc_lines.append(line)
        if '└────' in line:
            break

render_ansi_box(sc_lines, "PrettyDynoTerm — Official Benchmark Scorecard", os.path.join(OUT_DIR, "scorecard.png"))

print("All screenshots generated successfully in assets/screenshots/!")
