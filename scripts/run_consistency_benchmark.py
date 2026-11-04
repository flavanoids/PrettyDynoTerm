#!/usr/bin/env python3
"""
Dyno Term Consistency Benchmark Suite
Runs 5 iterations of ~1-minute benchmark runs with 60-second idle cooldown pauses,
collecting deterministic scores, per-act FPS, frametimes, and thermal metrics to verify consistency.
"""

import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

RUNS = 5
FRAMES = 880  # Calibrated for ~60s at 4K on RX 6650 XT across 7 acts
RES = "4k"
IDLE_PAUSE_SEC = 60
OUTPUT_DIR = Path("consistency_runs")
BINARY = Path("./target/release/dyno_term")


def calc_stats(values):
    n = len(values)
    if n == 0:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "cv_pct": 0}
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(variance)
    cv = (std / mean * 100.0) if mean != 0 else 0.0
    return {
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "cv_pct": cv,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not BINARY.exists():
        print(f"Error: binary {BINARY} not found! Run cargo build --release first.")
        sys.exit(1)

    print("=" * 80)
    print("DYNO_TERM 5-RUN CONSISTENCY BENCHMARK SUITE")
    print(f"Configuration: {RUNS} iterations, {FRAMES} frames/scene @ {RES.upper()} (~60s per run)")
    print(f"Idle Cooldown Interval: {IDLE_PAUSE_SEC} seconds between runs")
    print(f"Output Directory: {OUTPUT_DIR.resolve()}")
    print("=" * 80)

    run_data = []

    for i in range(1, RUNS + 1):
        out_file = OUTPUT_DIR / f"run_{i}.json"
        cmd = [
            str(BINARY),
            "-b",
            "-f",
            str(FRAMES),
            "--res",
            RES,
            "-o",
            str(out_file),
        ]

        start_time = time.time()
        start_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[ITERATION {i}/{RUNS}] Starting at {start_dt}...")
        print(f"Command: {' '.join(cmd)}")

        try:
            # We run subprocess and capture output
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            wall_elapsed = time.time() - start_time
            if proc.returncode != 0:
                print(f"ERROR: Iteration {i} exited with code {proc.returncode}!")
                print(proc.stdout[-500:])
                sys.exit(1)
        except Exception as e:
            print(f"ERROR running benchmark: {e}")
            sys.exit(1)

        if not out_file.exists():
            print(f"ERROR: Expected report {out_file} was not generated!")
            sys.exit(1)

        with open(out_file, "r") as f:
            data = json.load(f)

        score = data.get("overall_score", 0)
        benchmark_elapsed = data.get("total_elapsed_seconds", 0)
        gpu_temp = data.get("peak_gpu_temp_c", 0)
        hotspot_temp = data.get("peak_gpu_hotspot_c", 0)
        gpu_clock = data.get("peak_gpu_clock_mhz", 0)
        gpu_power = data.get("peak_gpu_power_w", 0)

        run_summary = {
            "iteration": i,
            "wall_elapsed_sec": round(wall_elapsed, 2),
            "benchmark_elapsed_sec": round(benchmark_elapsed, 2),
            "overall_score": score,
            "peak_gpu_temp_c": gpu_temp,
            "peak_gpu_hotspot_c": hotspot_temp,
            "peak_gpu_clock_mhz": gpu_clock,
            "peak_gpu_power_w": gpu_power,
            "scenes": {s["scene_name"]: s["avg_fps"] for s in data.get("scene_results", [])},
        }
        run_data.append(run_summary)

        print(f"[ITERATION {i}/{RUNS}] Completed in {benchmark_elapsed:.2f}s (wall {wall_elapsed:.2f}s)")
        print(f"  Score:         {score:,} pts")
        print(f"  Peak GPU Temp: {gpu_temp}°C (Hotspot: {hotspot_temp}°C)")
        print(f"  Peak Clock:    {gpu_clock} MHz | Power: {gpu_power} W")

        if i < RUNS:
            print(f"\n[IDLE COOLDOWN] Pausing for {IDLE_PAUSE_SEC}s to allow thermal stabilization...")
            for remaining in range(IDLE_PAUSE_SEC, 0, -10):
                print(f"  Cooldown remaining: {remaining}s...")
                time.sleep(10)
            print("  Cooldown complete. Resuming next run.")

    # Statistical Analysis
    scores = [r["overall_score"] for r in run_data]
    durations = [r["benchmark_elapsed_sec"] for r in run_data]
    temps = [r["peak_gpu_temp_c"] for r in run_data]
    hotspots = [r["peak_gpu_hotspot_c"] for r in run_data]
    clocks = [r["peak_gpu_clock_mhz"] for r in run_data]

    score_stats = calc_stats(scores)
    duration_stats = calc_stats(durations)
    temp_stats = calc_stats(temps)
    hotspot_stats = calc_stats(hotspots)
    clock_stats = calc_stats(clocks)

    scene_names = list(run_data[0]["scenes"].keys())
    scene_stats = {}
    for name in scene_names:
        scene_fps_list = [r["scenes"][name] for r in run_data]
        scene_stats[name] = calc_stats(scene_fps_list)

    summary = {
        "runs_analyzed": RUNS,
        "config": {
            "frames_per_scene": FRAMES,
            "resolution": RES,
            "cooldown_seconds": IDLE_PAUSE_SEC,
        },
        "score_statistics": score_stats,
        "duration_statistics": duration_stats,
        "temperature_statistics": temp_stats,
        "hotspot_statistics": hotspot_stats,
        "clock_statistics": clock_stats,
        "scene_fps_statistics": scene_stats,
        "runs": run_data,
    }

    summary_file = OUTPUT_DIR / "consistency_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Print Report
    print("\n" + "=" * 80)
    print("                     CONSISTENCY BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Iterations:        {RUNS} runs x ~60s with 60s idle pause")
    print(f"Mean Score:        {score_stats['mean']:,.0f} pts")
    print(f"Standard Dev:      ±{score_stats['std']:,.1f} pts ({score_stats['cv_pct']:.3f}% CV)")
    print(f"Min / Max Score:   {score_stats['min']:,} / {score_stats['max']:,}")
    print(f"Score Spread:      {score_stats['range']:,} pts ({(score_stats['range'] / score_stats['mean'] * 100):.3f}%)")
    print("-" * 80)
    print(f"Mean Duration:     {duration_stats['mean']:.2f}s (±{duration_stats['std']:.2f}s)")
    print(f"Peak GPU Temp:     {temp_stats['mean']:.1f}°C (Min: {temp_stats['min']}°C, Max: {temp_stats['max']}°C)")
    print(f"Peak Hotspot Temp: {hotspot_stats['mean']:.1f}°C (Min: {hotspot_stats['min']}°C, Max: {hotspot_stats['max']}°C)")
    print(f"Peak GPU Clock:    {clock_stats['mean']:.0f} MHz (Min: {clock_stats['min']} MHz, Max: {clock_stats['max']} MHz)")
    print("=" * 80)

    print("\nPER-ACT FPS STABILITY ACROSS 5 RUNS:")
    print(f"{'Scene Name':<34} | {'Mean FPS':>10} | {'Std Dev':>8} | {'CV (%)':>7} | {'Min FPS':>9} | {'Max FPS':>9}")
    print("-" * 88)
    for name, s in scene_stats.items():
        print(f"{name:<34} | {s['mean']:>10.2f} | {s['std']:>8.2f} | {s['cv_pct']:>6.2f}% | {s['min']:>9.2f} | {s['max']:>9.2f}")
    print("=" * 88)
    print(f"\nDetailed summary saved to {summary_file}")


if __name__ == "__main__":
    main()
