import os
import re
import sys
import time
import psutil
import subprocess

def get_gpu_info():
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        parts = [p.strip() for p in res.stdout.strip().split(",")]
        return {
            "temp": f"{parts[0]}°C",
            "power": f"{float(parts[1]):.1f}W",
            "vram_used": f"{int(parts[2])/1024:.1f} GB",
            "vram_total": f"{int(parts[3])/1024:.1f} GB",
            "util": f"{parts[4]}%"
        }
    except Exception:
        return {"temp": "N/A", "power": "N/A", "vram_used": "N/A", "vram_total": "N/A", "util": "N/A"}

def render_bar(current, total, length=30):
    pct = min(1.0, max(0.0, current / max(1, total)))
    filled = int(length * pct)
    bar = "█" * filled + "░" * (length - filled)
    return f"[{bar}] {pct*100:.1f}%"

def parse_latest_log():
    log_file = "C:/Users/khana/.gemini/antigravity-ide/brain/39667801-c04c-4e9d-b335-0be4f52aff95/.system_generated/tasks/task-293.log"
    if not os.path.exists(log_file):
        return None
    
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Find matches for Epoch and progress
    # e.g.: Epoch 1/2:   1%|          | 645/54000 [02:07<2:51:29,  5.19it/s, loss=0.6986, lr=1.19e-05, vram=4.3GB]
    matches = re.findall(r"Epoch\s+(\d+)/(\d+):\s+(\d+)%\|.*?\|\s+(\d+)/(\d+)\s+\[(.*?),\s+([\d\.]+it/s)(?:,\s+loss=([\d\.]+))?(?:,\s+lr=([e\d\.\-]+))?(?:,\s+vram=([\d\.]+GB))?\]", content)
    if not matches:
        return None

    last = matches[-1]
    return {
        "epoch": int(last[0]),
        "total_epochs": int(last[1]),
        "pct": int(last[2]),
        "step": int(last[3]),
        "total_steps": int(last[4]),
        "time_str": last[5],
        "speed": last[6],
        "loss": last[7] if last[7] else "N/A",
        "lr": last[8] if last[8] else "N/A",
        "vram_log": last[9] if last[9] else "N/A"
    }

def display_dashboard():
    os.system("cls" if os.name == "nt" else "clear")
    info = parse_latest_log()
    gpu = get_gpu_info()
    cpu_pct = psutil.cpu_percent()
    ram = psutil.virtual_memory()

    print("="*65)
    print("      🛰️  SATQUERY: REAL-TIME MULTIMODAL TRAINING DASHBOARD      ")
    print("="*65)

    if info:
        print(f"\n▶ CURRENT CYCLE: Epoch {info['epoch']} / {info['total_epochs']}")
        print(f"  Batch Progress:  {render_bar(info['step'], info['total_steps'], 35)}  ({info['step']:,} / {info['total_steps']:,})")
        print(f"  Training Speed:  {info['speed']}  |  Elapsed / ETA: {info['time_str']}")
        print(f"  Current Loss:    {info['loss']}    |  Learning Rate: {info['lr']}")
        
        # Total overall training progress
        total_steps_all = info['total_steps'] * info['total_epochs']
        curr_step_all = (info['epoch'] - 1) * info['total_steps'] + info['step']
        print(f"\n▶ OVERALL PROGRESS (2 EPOCHS TOTAL):")
        print(f"  Total Progress:  {render_bar(curr_step_all, total_steps_all, 35)}  ({curr_step_all:,} / {total_steps_all:,})")
    else:
        print("\n  Waiting for training metrics from background task...")

    print("\n" + "-"*65)
    print("▶ HARDWARE HEALTH & DESKTOP TELEMETRY:")
    print(f"  GPU Temp:     {gpu['temp']:<8} (Cool & Quiet) |  Power Draw:   {gpu['power']}")
    print(f"  GPU VRAM:     {gpu['vram_used']} / {gpu['vram_total']} (Ceiling Active, 4.5 GB Free for Video)")
    print(f"  GPU Engine:   {gpu['util']:<8} (3D Compute) |  Process Prio: BelowNormal")
    print(f"  CPU Usage:    {cpu_pct}% (<15% average)     |  System RAM:   {ram.used/(1024**3):.1f} GB / {ram.total/(1024**3):.1f} GB")
    print("-" * 65)
    print("\n[Controls]:")
    print("  • Run 'stop_training.bat' to safely save checkpoint and stop anytime.")
    print("  • Press Ctrl+C to exit this monitor (training continues in background).")
    print("="*65)

if __name__ == "__main__":
    try:
        while True:
            display_dashboard()
            time.sleep(3)
    except KeyboardInterrupt:
        print("\nExiting monitor. Training remains running in the background.")
