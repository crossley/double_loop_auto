#!/bin/bash

set -eu

MODULE_VERSION="${MODULE_VERSION:-python3/3.11.0}"
VENV_DIR="${VENV_DIR:-$PWD/.venv-model_spiking_cat_90vs180_gadi}"
REQ_FILE="${REQ_FILE:-gadi/model_spiking_cat_90vs180_gadi_requirements.txt}"

module load "${MODULE_VERSION}"

python3 -m venv "${VENV_DIR}"
. "${VENV_DIR}/bin/activate"
python3 -m pip install --upgrade pip
python3 -m pip install -r "${REQ_FILE}"

echo "Environment ready at ${VENV_DIR}"
echo "Activate with: . ${VENV_DIR}/bin/activate"
