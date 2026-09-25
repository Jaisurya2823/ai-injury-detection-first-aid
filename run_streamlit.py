"""
Root-level Streamlit launcher.
Run this instead of `streamlit run app/streamlit_app.py` on Windows,
or whenever you get `ModuleNotFoundError: No module named 'app'`.

Usage:
    python run_streamlit.py
"""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    script = Path(__file__).parent / "app" / "streamlit_app.py"
    sys.exit(subprocess.call(["streamlit", "run", str(script)]))
