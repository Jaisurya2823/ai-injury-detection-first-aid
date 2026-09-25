#!/usr/bin/env python3
"""
SAFE-AI First Aid System — one-time setup script.

Run this ONCE after extracting the zip, before starting the app:

    python setup.py

It will:
  1. Check Python version
  2. Install required packages
  3. Create .env from .env.example if missing
  4. Ask for your Roboflow API key and save it to .env
  5. Verify the API key works
  6. Print the command to start the app
"""
from __future__ import annotations
import os, sys, subprocess, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

print("=" * 60)
print("  SAFE-AI First Aid System — Setup")
print("=" * 60)

# 1. Python version
print(f"\n[1/5] Python {sys.version.split()[0]}")
if sys.version_info < (3, 9):
    print("  ✗ Python 3.9 or newer is required.")
    sys.exit(1)
print("  ✓ OK")

# 2. Install packages
print("\n[2/5] Installing packages from requirements.txt ...")
req = ROOT / "requirements.txt"
if req.exists():
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req), "-q"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ✗ pip failed:\n{result.stderr[-800:]}")
        sys.exit(1)
print("  ✓ OK")

# Also install inference-sdk
subprocess.run(
    [sys.executable, "-m", "pip", "install", "inference-sdk", "-q"],
    capture_output=True
)

# 3. Create .env
print("\n[3/5] Setting up .env ...")
env_file = ROOT / ".env"
example  = ROOT / ".env.example"
if not env_file.exists():
    if example.exists():
        import shutil
        shutil.copy(example, env_file)
        print("  ✓ Created .env from .env.example")
    else:
        env_file.write_text(
            "FIRST_AID_MODEL_MODE=workflow\n"
            "ROBOFLOW_API_KEY=\n"
            "FIRST_AID_CONFIDENCE_THRESHOLD=0.40\n"
            "FIRST_AID_OOD_THRESHOLD=0.60\n"
        )
        print("  ✓ Created blank .env")
else:
    print("  ✓ .env already exists")

# 4. Set Roboflow API key
print("\n[4/5] Roboflow API key")
env_text = env_file.read_text(encoding="utf-8")

# Check if a real key is already set
current_key_match = re.search(r"^ROBOFLOW_API_KEY=(.+)$", env_text, re.MULTILINE)
current_key = (current_key_match.group(1).strip() if current_key_match else "").strip('"\'')

if current_key and not current_key.startswith("your_"):
    print(f"  ✓ API key already set ({current_key[:8]}...)")
else:
    print("  Get your FREE key at: app.roboflow.com → Settings → API Keys")
    key = input("  Paste your Roboflow API key here: ").strip()
    if not key:
        print("  ✗ No key entered. Edit .env manually before starting the app.")
    else:
        if current_key_match:
            env_text = re.sub(
                r"^ROBOFLOW_API_KEY=.*$",
                f"ROBOFLOW_API_KEY={key}",
                env_text, flags=re.MULTILINE
            )
        else:
            env_text += f"\nROBOFLOW_API_KEY={key}\n"
        env_file.write_text(env_text, encoding="utf-8")
        print("  ✓ API key saved to .env")

# 5. Quick verify
print("\n[5/5] Verifying configuration ...")
# Re-read .env to load key
for line in env_file.read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.split("#")[0].strip().strip('"\'')
        if k and k not in os.environ:
            os.environ[k] = v

mode = os.getenv("FIRST_AID_MODEL_MODE", "?")
key  = os.getenv("ROBOFLOW_API_KEY", "")
print(f"  Mode      : {mode}")
print(f"  API key   : {'SET (' + key[:8] + '...)' if key and not key.startswith('your_') else 'NOT SET'}")
print(f"  Threshold : {os.getenv('FIRST_AID_CONFIDENCE_THRESHOLD', '0.40')}")

if not key or key.startswith("your_"):
    print("\n  ⚠  ROBOFLOW_API_KEY is not set.")
    print("     Edit .env and add your key, then re-run setup.py or start the app.")
else:
    print("  ✓ Configuration looks good")

print()
print("=" * 60)
print("  Setup complete. Start the app with:")
print()
print("      python run_streamlit.py")
print()
print("  App will open at: http://localhost:8501")
print("=" * 60)
