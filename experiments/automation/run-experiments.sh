#!/usr/bin/env bash
# Run from repo root (inside docker):
#   ./experiments/automation/run-experiments.sh path/to/PARAMS.sh
set -euo pipefail

# ---------- params ----------
params_file=${1:-experiments/locomotion.sh}
source "$params_file"

# Paths
fatal_flag="${out_path}/${study_name}/FATAL"
recover_dir="${out_path}/${study_name}/recoverable"
mkdir -p "${out_path}/${study_name}" "${out_path}/${study_name}/analysis" "$recover_dir"

# Defaults
: "${MAX_PARALLEL:=1}"
: "${WAIT_FOR_ALL:=1}"
: "${delay_setup_script:=10}"
: "${docker_path:=.}"

# ---------- helpers ----------
timestamp() { date +"%Y-%m-%d_%H-%M-%S"; }

count_running_sessions() {
  tmux ls 2>/dev/null | grep -E "^${study_name}_" | wc -l | tr -d ' '
}

kill_all_sessions() {
  tmux ls 2>/dev/null | awk -F: '{print $1}' | grep -E "^${study_name}_" | while read -r s; do
    tmux kill-session -t "$s" 2>/dev/null || true
  done
}

# CUDA: driver present?
cuda_driver_ok() {
  command -v nvidia-smi >/dev/null 2>&1 || return 1
  nvidia-smi >/dev/null 2>&1 || return 1
  nvidia-smi -L >/dev/null 2>&1 || return 1
  test "$(nvidia-smi -L | wc -l | tr -d ' ')" -ge 1 || return 1
  return 0
}

# CUDA: runtime init ok? (cuInit(0) == 0)
cuda_runtime_ok() {
  python3 - <<'PY' 2>/dev/null
import ctypes, sys
try:
    lib = ctypes.CDLL('libcuda.so')
    cuInit = lib.cuInit
    cuInit.argtypes = [ctypes.c_uint]
    rc = cuInit(0)
    sys.exit(0 if rc == 0 else 2)
except Exception:
    sys.exit(1)
PY
}

# Total-kill iff driver or runtime are NOT ok
cuda_dead() {
  ! cuda_driver_ok && return 0
  ! cuda_runtime_ok && return 0
  return 1
}

# Handle sentinel:
# - If CUDA is dead -> total kill (return 0)
# - Else -> archive sentinel and continue (return 1)
handle_sentinel_if_any() {
  if [[ -f "$fatal_flag" ]]; then
    if cuda_dead; then
      echo "CUDA unavailable (driver/runtime)."
      echo "Sentinel contents:"; sed -n '1,200p' "$fatal_flag" || true
      return 0
    else
      local ts; ts=$(timestamp)
      mv -f "$fatal_flag" "${recover_dir}/recoverable_${ts}.txt" || true
      return 1
    fi
  fi
  return 1
}

# ---------- clear stale sentinel ----------
rm -f "$fatal_flag" 2>/dev/null || true

# ---------- parse lists ----------
IFS=',' read -r -a EXP_LIST  <<< "${experiments}"
IFS=',' read -r -a TF_LIST   <<< "${tfs}"
IFS=',' read -r -a COND_LIST <<< "${env_conditions}"
IFS=',' read -r -a RUN_LIST  <<< "${runs}"

if [[ ${#EXP_LIST[@]} -ne ${#TF_LIST[@]} || ${#EXP_LIST[@]} -ne ${#COND_LIST[@]} ]]; then
  echo "Error: experiments, tfs, env_conditions must have same length."
  exit 1
fi

# ---------- launch ----------
for idx in "${!EXP_LIST[@]}"; do
  exp="${EXP_LIST[$idx]}"
  tf="${TF_LIST[$idx]}"
  cond="${COND_LIST[$idx]}"

  for run in "${RUN_LIST[@]}"; do
    logfile="${out_path}/${study_name}/${exp}_${run}.log"
    session="${study_name}_${exp}_r${run}"
    session=${session//[^a-zA-Z0-9_-]/-}

    # Hard stop if CUDA dead (before launch)
    if cuda_dead; then
      echo "CUDA unavailable before launch. Aborting all launches."
      exit 1
    fi
    # If sentinel exists, decide based on CUDA status
    if handle_sentinel_if_any; then
      echo "CUDA dead via sentinel. Aborting."
      exit 1
    fi

    # Throttle
    while [[ $(count_running_sessions) -ge ${MAX_PARALLEL} ]]; do
      # While waiting: if CUDA dies -> kill all immediately, no one continues
      if cuda_dead; then
        echo "CUDA died while waiting. Killing all sessions..."
        kill_all_sessions
        exit 1
      fi
      # Recoverable sentinel -> archive, continue
      handle_sentinel_if_any || true
      sleep "${delay_setup_script}"
    done

    # Skip if exists
    if tmux has-session -t "$session" 2>/dev/null; then
      echo "Session exists, skipping: $session"
      continue
    fi

    # Close race: re-check right before launch
    if cuda_dead; then
      echo "CUDA unavailable at pre-launch. Aborting."
      exit 1
    fi
    handle_sentinel_if_any || true

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
echo ">> Launched with concurrency = ${MAX_PARALLEL}"
echo ">> Attach: tmux attach -t ${study_name}_<exp>_r<run>"

# ---------- wait & analyze ----------
if [[ "${WAIT_FOR_ALL}" -eq 1 ]]; then
  echo ">> Waiting for all ${study_name} sessions to finish..."
  while true; do
    running=$(count_running_sessions)

    # If CUDA dies mid-flight: kill all immediately; no sim continues
    if cuda_dead; then
      echo "CUDA died during wait. Killing all ${study_name} sessions..."
      kill_all_sessions
      exit 1
    fi
    # Recoverable sentinel -> archive & continue
    handle_sentinel_if_any || true

    if [[ "$running" -le 0 ]]; then
      break
    fi
    sleep 10
  done

  echo ">> All runs finished. Starting analysis..."
  ./experiments/automation/run-analysis.sh "$params_file"
fi

