#!/bin/bash
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_ROOT_DIR="$(realpath "$SCRIPT_DIR/../..")"

git config --global --add safe.directory /workspace/Dynamic-GSG
git config --global --add safe.directory /workspace/Dynamic-GSG/submodules/GroundingDINO
git config --global --add safe.directory /workspace/Dynamic-GSG/submodules/diff-gaussian-rasterization-w-depth
git config --global --add safe.directory /workspace/Dynamic-GSG/submodules/describe-anything

# Ensure all git submodules are initialized and updated
cd $PROJECT_ROOT_DIR
git submodule update --init --recursive

# Build DiffGaussianRasterizationWDepth
cd submodules/diff-gaussian-rasterization-w-depth
python3 -m pip install . 

# Build GroundingDINO
cd ../GroundingDINO
python3 -m pip install .

# Build DescribeAnything
cd ../describe-anything
python3 -m pip install -v .

# Build and install other third-party dependencies
python3 -m pip install -r ../../requirements.txt