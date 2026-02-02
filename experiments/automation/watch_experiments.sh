#!/bin/bash
# run this script from the ROOT (inside docker): ./experiments/automation/watch_experiments.sh pathPARAMSFILE/PARAMSFILE.sh
#set -e
#set -x

if [ $# -eq 0 ]
  then
    params_file="experiments/locomotion.sh"
  else
    params_file=$1
fi

set -a
source "$params_file"
set +a

python3 experiments/analysis/watch_robots.py \
  --study_name "$study_name" \
  --experiments "$experiments" \
  --tfs "$tfs" \
  --runs "$runs" \
  --generations "$generations" \
  --out_path "$out_path" \
  --max_voxels "$max_voxels" \
  --cube_face_size "$cube_face_size" \
  ;