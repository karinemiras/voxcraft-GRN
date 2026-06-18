#!/bin/bash

# Run the analysis notebook manually from the repo root:
#   ./experiments/automation/notebook.sh experiments/locomotion.sh

if [ $# -eq 0 ]; then
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

papermill_args=()
for param_name in "${config_param_names[@]}"; do
  param_value="${!param_name-}"
  if [[ -n "$param_value" ]]; then
    papermill_args+=(-p "$param_name" "$param_value")
  fi
done

papermill "experiments/analysis/analysis_inter.ipynb" \
          "experiments/analysis/analysis_inter-executed.ipynb" \
          "${papermill_args[@]}"
