#!/usr/bin/env bash
# Copy only the code SageMaker needs into build/source (keeps data, .git and notebooks out of the upload).
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf build/source && mkdir -p build/source
cp -R src build/source/src
cp -R sagemaker build/source/sagemaker
find build/source -name "__pycache__" -type d -prune -exec rm -rf {} +
echo "packaged build/source"
