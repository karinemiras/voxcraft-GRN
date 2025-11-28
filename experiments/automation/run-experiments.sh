#!/usr/bin/env bash
# Run from repo root (inside docker):
#   ./experiments/automation/run-experiments.sh path/to/PARAMS.sh

# prevent silent errors
set -euo pipefail

# ---------- params(set default)----------
params_file=${1:-experiments/locomotion.sh}
source "$params_file"

# Defaults if not set in params file
: "${MAX_PARALLEL:=10}"      # how many runs to execute at once (1 = queue the rest) [max 1 in current GPU setup)
: "${WAIT_FOR_ALL:=1}"      # 1 = wait for all runs then run analysis, 0 = don't wait
: "${delay_setup_script:=10}" # seconds between queue checks if throttling

# ---------- setup ----------
mkdir -p "${out_path}/${study_name}" "${out_path}/${study_name}/analysis"

# Parse comma-separated lists from params
IFS=',' read -r -a EXP_LIST  <<< "${experiments}"
IFS=',' read -r -a TF_LIST   <<< "${tfs}"
IFS=',' read -r -a COND_LIST <<< "${env_conditions}"
IFS=',' read -r -a RUN_LIST  <<< "${runs}"

# Sanity check
if [[ ${#EXP_LIST[@]} -ne ${#TF_LIST[@]} || ${#EXP_LIST[@]} -ne ${#COND_LIST[@]} ]]; then
  echo "Error: experiments, tfs, and env_conditions must have same length (comma-separated)."
  echo "experiments='${experiments}'"
  echo "tfs='${tfs}'"
  echo "env_conditions='${env_conditions}'"
  exit 1
fi

# Helper: count running tmux sessions for this study
count_running_sessions() {
  tmux ls 2>/dev/null | grep -E "^${study_name}_" | wc -l | tr -d ' '
}

# ---------- launch ----------
for idx in "${!EXP_LIST[@]}"; do
  exp="${EXP_LIST[$idx]}"
  tf="${TF_LIST[$idx]}"
  cond="${COND_LIST[$idx]}"

  for run in "${RUN_LIST[@]}"; do
    logfile="${out_path}/${study_name}/${exp}_${run}.log"
    session="${study_name}_${exp}_r${run}"
    session=${session//[^a-zA-Z0-9_-]/-}

    # Throttle: wait until < MAX_PARALLEL sessions are running
    while [[ $(count_running_sessions) -ge ${MAX_PARALLEL} ]]; do
      sleep "${delay_setup_script}"
    done

    # Skip if already exists
    if tmux has-session -t "$session" 2>/dev/null; then
      echo "Session exists, skipping: $session"
      continue
    fi

    # Build command (NOTE: python3 -u for live logging)
    cmd=(
      python3 -u "${docker_path}/algorithms/${algorithm}.py"
      --out_path "${out_path}"
      --experiment_name "${exp}"
      --env_conditions "${cond}"
      --run "${run}"
      --study_name "${study_name}"
      --algorithm "${algorithm}"
      --fitness_metric "${fitness_metric}"
      --num_generations "${num_generations}"
      --population_size "${population_size}"
      --offspring_size "${offspring_size}"
      --simulation_time "${simulation_time}"
      --plastic "${plastic}"
      --docker_path "${docker_path}"
      --crossover_prob "${crossover_prob}"
      --mutation_prob "${mutation_prob}"
      --max_voxels "${max_voxels}"
      --tfs "${tf}"
      --cube_face_size "${cube_face_size}"
      --run_simulation "${run_simulation}"
    )

    echo "Launching $session  ->  $logfile"
    tmux new-session -d -s "$session" \
      "mkdir -p '${out_path}/${study_name}'; exec ${cmd[*]} >>'$logfile' 2>&1"
  done
done

echo ""
echo ">> Launched runs with concurrency limit = ${MAX_PARALLEL}"
echo ">> Attach: tmux attach -t ${study_name}_<exp>_r<run>"
echo ">> Logs:   tail -f ${out_path}/${study_name}/<exp>_<run>.log"

# ---------- optional: wait for all, then analyze ----------
if [[ "${WAIT_FOR_ALL}" -eq 1 ]]; then
  echo ">> Waiting for all ${study_name} sessions to finish..."
  while [[ $(count_running_sessions) -gt 0 ]]; do
    sleep 10
  done
  echo ">> All runs finished. Starting analysis..."
  ./experiments/automation/run-analysis.sh "$params_file"
fi
