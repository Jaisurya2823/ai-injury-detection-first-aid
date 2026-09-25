from __future__ import annotations
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], label: str, round_id: str) -> None:
    print(f"\n=== {label} ===", flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["FIRST_AID_DATA_DIR"] = str(ROOT / ".stress_data" / round_id)
    env["FIRST_AID_DB_PATH"] = str(ROOT / ".stress_data" / round_id / "first_aid.sqlite3")
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, timeout=60)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> None:
    stress_root = ROOT / ".stress_data"
    shutil.rmtree(stress_root, ignore_errors=True)
    stress_root.mkdir(parents=True, exist_ok=True)
    try:
        for i in range(1, 9):
            run([sys.executable, "-m", "pytest", "-q"], f"FUNCTIONAL PASS {i}/8", f"functional_{i}")
        for i in range(1, 9):
            run([
                sys.executable, "-m", "pytest", "-q",
                "tests/test_storage.py", "tests/test_quality_and_pipeline.py", "tests/test_crash_resilience.py",
            ], f"CRASH/FAULT PASS {i}/8", f"fault_{i}")
        print("\nALL 16 STRESS ROUNDS PASSED", flush=True)
    finally:
        shutil.rmtree(stress_root, ignore_errors=True)


if __name__ == "__main__":
    main()
