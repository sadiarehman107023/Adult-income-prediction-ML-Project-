"""
Hugging Face Spaces entry point for Streamlit.
This ensures the Streamlit app runs correctly both locally and on deployment.
"""

import os
import subprocess
from pathlib import Path

# Define the path to the main Streamlit app
current_dir = Path(__file__).parent
streamlit_app_path = current_dir / "streamlit_app.py"

if not streamlit_app_path.exists():
    raise FileNotFoundError(f"Could not find {streamlit_app_path}. Make sure streamlit_app.py exists.")

# --- Run the Streamlit app properly ---
# This launches Streamlit’s runtime rather than executing the file directly
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"   # Required for Hugging Face Spaces
os.environ["STREAMLIT_SERVER_PORT"] = "7860"       # Hugging Face default port

import sys
subprocess.run([sys.executable, "-m", "streamlit", "run", str(streamlit_app_path)])


