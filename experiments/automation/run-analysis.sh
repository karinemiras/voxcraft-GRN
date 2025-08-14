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
#python experiments/analysis/bests_snap_draw.py $study $experiments $runs $generations $out_path;

papermill "experiments/analysis/analysis.ipynb" \
          "experiments/analysis/analysis-executed.ipynb" \
          -p study "$study" \
          -p experiments "$experiments" \
          -p runs "$runs" \
          -p generations "$generations" \
          -p final_gen "$final_gen" \
          -p out_path "$out_path"
