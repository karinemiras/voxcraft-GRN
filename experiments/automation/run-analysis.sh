#!/bin/bash

# run this script from the ROOT: ./experiments/automation/run-analysis.sh pathPARAMSFILE/PARAMSFILE.sh

if [ $# -eq 0 ]
  then
    params_file="experiments/noveltysearch.sh"
  else
    params_file=$1
fi

set -a
source "$params_file"
set +a

#python experiments/analysis/consolidate.py $study $experiments $runs $generations $out_path;

python experiments/analysis/snapshots_bests.py \
  --study_name "$study_name" \
  --experiments "$experiments" \
  --tfs "$tfs" \
  --runs "$runs" \
  --generations "$generations" \
  --out_path "$out_path" \
  --max_voxels "$max_voxels" \
  --env_conditions "$env_conditions" \
  --algorithm "$algorithm" \
  --plastic "$plastic"


#papermill "experiments/analysis/analysis.ipynb" \
#          "experiments/analysis/analysis-executed.ipynb" \
#          -p study "$study" \
#          -p experiments "$experiments" \
#          -p runs "$runs" \
#          -p generations "$generations" \
#          -p final_gen "$final_gen" \
#          -p out_path "$out_path"
