#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────
# JobCoach AI – one‑shot runner
# Creates (if missing) the .venv, installs deps, downloads spaCy
# model once, then launches Streamlit. Re‑run anytime; it will
# detect the existing venv and skip redundant steps.
# ----------------------------------------------------------------

set -e  # exit on first error

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_DIR"

# 1️⃣  Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
  echo "🔧 Creating virtual environment (.venv)…"
  python -m venv .venv
fi

# 2️⃣  Activate venv (works on macOS/Linux & Git‑Bash)
source .venv/bin/activate

# 3️⃣  Upgrade resolver
python -m pip install --upgrade pip wheel setuptools

# 4️⃣  Install/upgrade required packages
python -m pip install --no-cache-dir -r requirements.txt

# 5️⃣  spaCy model (runs only once)
python - << 'PY'
import importlib.util, subprocess, sys
model = 'en_core_web_sm'
if importlib.util.find_spec(model) is None:
    print('📦 Downloading spaCy model:', model)
    subprocess.check_call([sys.executable, '-m', 'spacy', 'download', model])
else:
    print('✅ spaCy model already present')
PY

# 6️⃣  Launch Streamlit
exec streamlit run app.py
