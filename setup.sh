#!/usr/bin/env bash
# Sets up a consistent environment for topophone2:
#   1) creates/updates the conda env from environment.yml
#   2) clones GeoPVI (pinned commit) into external/GeoPVI, builds its FMM solver, installs it
#   3) installs topophone2 (editable)
# Usage: ./setup.sh [--with-fwi2d]     (Linux / macOS / WSL; needs conda or mamba)
set -euo pipefail

ENV_NAME="topophone2"
ENV_YML="conda_env.yml" 

GEOPVI_URL="https://github.com/XuebinZhaoZXB/GeoPVI.git"
GEOPVI_VERSION="v1.0.0" 
GEOPVI_DIR="external/GeoPVI"
WITH_FWI2D=0
[ "${1:-}" = "--with-fwi2d" ] && WITH_FWI2D=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if command -v mamba >/dev/null 2>&1; then CONDA=mamba; else CONDA=conda; fi
command -v "$CONDA" >/dev/null 2>&1 || { echo "conda/mamba not found"; exit 1; }

# 1) Create/update conda env ===================================================================

echo "==> [1/3] Creating/updating conda env '$ENV_NAME'"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    "$CONDA" env update -n "$ENV_NAME" -f "$ENV_YML" --prune
else
    "$CONDA" env create -n "$ENV_NAME" -f "$ENV_YML"
fi

# activate the env to ensure that the rest of the script runs in the correct environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
echo "==> Conda env '$ENV_NAME' activated"

# 2) Clone/build/install GeoPVI ================================================================
echo "==> [2/3] Installing GeoPVI @ ${GEOPVI_VERSION}"
if [ ! -d "$GEOPVI_DIR/.git" ]; then
    git clone "$GEOPVI_URL" "$GEOPVI_DIR"
fi
git -C "$GEOPVI_DIR" fetch --quiet origin
git -C "$GEOPVI_DIR" checkout --quiet "$GEOPVI_VERSION"

# Some GeoPVI tags (e.g. v1.0.0) were committed with CRLF line endings, which
# breaks bash ("$'\r': command not found"). Strip them if present; no-op on
# tags that are already clean (e.g. master, v2.0, variable-density-*).
if grep -qU $'\r' "$GEOPVI_DIR/setup.sh"; then
    echo "    fixing CRLF line endings in GeoPVI's setup.sh"
    sed -i 's/\r$//' "$GEOPVI_DIR/setup.sh"
fi

# cd to GeoPVI and run the setup.sh script to build the FMM solver and install GeoPVI
cd "$GEOPVI_DIR"
bash setup.sh
# echo "Build 2d fwi solver..."
# cd geopvi/forward/fwi2d
# RESULT="$(python setup.py build_ext -i 2>&1)" && status=0 || status=$?
# if [ $status -eq 0 ]; then
#     echo "Building 2d fwi solver succeeds"
# else
#     echo "Error: $RESULT"
# fi
# cd ../../../

# echo "Build 2d FMM solver..."
# cd geopvi/forward/tomo2d
# RESULT="$(make && python setup.py build_ext -i 2>&1)" && status=0 || status=$?
# if [ $status -eq 0 ]; then
#     echo "Building 2d FMM solver succeeds"
# else
#     echo "Error: $RESULT"
# fi
# cd ../../../
echo "Install GeoPVI..."
pip install -e .

echo "==> GeoPVI installed"

# 3) Install topophone2 =======================================================================
echo "==> [3/3] Installing topophone2"

# cd back to the topophone2 directory (step 2 left us inside external/GeoPVI)
cd "$SCRIPT_DIR"

# --no-deps: dependencies come from environment.yml (requirements.txt still lists
# stdlib modules and a PyPI 'geopvi', which we do not want pip to resolve)
pip install -e . --no-deps

conda deactivate

echo ".................END of setup.sh................."

# =============================================================================================== 