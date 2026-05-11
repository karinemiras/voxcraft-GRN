#!/bin/bash

# run this script from the ROOT (inside docker): ./experiments/automation/run-analysis.sh pathPARAMSFILE/PARAMSFILE.sh
# if u want to run it out of docker, update out_path and docker_path

if [ $# -eq 0 ]
  then
    params_file="experiments/locomotion.sh"
  else
    params_file=$1
fi

set -a
source "$params_file"
set +a

mapfile -t config_param_names < <(
  python3 -c 'import ast
tree = ast.parse(open("utils/config.py").read())
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                print(arg.value[2:].replace("-", "_"))
                break'
)

common_args=()
papermill_args=()
for param_name in "${config_param_names[@]}"; do
  param_value="${!param_name-}"
  if [[ -n "$param_value" ]]; then
    common_args+=(--"$param_name" "$param_value")
    papermill_args+=(-p "$param_name" "$param_value")
  fi
done


python3 ${docker_path}/experiments/analysis/symmetry_metrics_csv.py \
  "${common_args[@]}" \
  --output_csv "${out_path}/${study_name}/analysis/additional_metrics.csv"


python3 ${docker_path}/experiments/analysis/consolidate.py \
  "${common_args[@]}"



papermill "experiments/analysis/analysis.ipynb" \
          "experiments/analysis/analysis-executed.ipynb" \
          "${papermill_args[@]}"


python3 ${docker_path}/experiments/analysis/snapshots_bests.py \
  "${common_args[@]}"


python3 ${docker_path}/experiments/analysis/bests_snap_draw.py \
  "${common_args[@]}"
