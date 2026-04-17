#!/bin/bash

set -euo pipefail

ROTATIONS="${ROTATIONS:-90}"
N_SIMULATIONS="${N_SIMULATIONS:-1}"
N_TRIALS="${N_TRIALS:-100}"
PROBE_TRIAL_ONSETS="${PROBE_TRIAL_ONSETS:-100}"
N_PROBE_TRIALS="${N_PROBE_TRIALS:-20}"
SEED="${SEED:-1}"

SCRIPTS=(
  "code/model_spiking_cat_90vs180_gadi.py"
  "code/model_spiking_cat_90vs180_vectorized.py"
  "code/model_spiking_cat_90vs180_vectorized_light.py"
)

LABELS=(
  "baseline_gadi_safe"
  "vectorized"
  "vectorized_light"
)

TIME_FILES=()
cleanup() {
  for file in "${TIME_FILES[@]:-}"; do
    [ -n "${file}" ] && [ -f "${file}" ] && rm -f "${file}"
  done
}
trap cleanup EXIT

run_one() {
  local label="$1"
  local script_path="$2"
  local time_file
  time_file="$(mktemp)"
  TIME_FILES+=("${time_file}")

  echo "Running ${label} ..." >&2
  /usr/bin/time -p -o "${time_file}" \
    python3 "${script_path}" \
      --rotations "${ROTATIONS}" \
      --n-simulations "${N_SIMULATIONS}" \
      --n-trials "${N_TRIALS}" \
      --probe-trial-onsets "${PROBE_TRIAL_ONSETS}" \
      --n-probe-trials "${N_PROBE_TRIALS}" \
      --seed "${SEED}"

  awk '/^real / {print $2}' "${time_file}"
}

printf "Benchmark settings: rotations=%s n_simulations=%s n_trials=%s probe_trial_onsets=%s n_probe_trials=%s seed=%s\n" \
  "${ROTATIONS}" "${N_SIMULATIONS}" "${N_TRIALS}" "${PROBE_TRIAL_ONSETS}" "${N_PROBE_TRIALS}" "${SEED}"
echo

declare -a RESULTS=()
for idx in "${!SCRIPTS[@]}"; do
  elapsed="$(run_one "${LABELS[$idx]}" "${SCRIPTS[$idx]}")"
  RESULTS+=("${elapsed}")
done

baseline="${RESULTS[0]}"
vectorized="${RESULTS[1]}"
light="${RESULTS[2]}"

printf "\n%-20s %12s %16s\n" "variant" "seconds" "speedup_vs_base"
printf "%-20s %12s %16s\n" "--------------------" "------------" "----------------"

for idx in "${!LABELS[@]}"; do
  seconds="${RESULTS[$idx]}"
  speedup="$(awk -v base="${baseline}" -v current="${seconds}" 'BEGIN { if (current == 0) print "inf"; else printf "%.2fx", base / current }')"
  printf "%-20s %12.2f %16s\n" "${LABELS[$idx]}" "${seconds}" "${speedup}"
done
