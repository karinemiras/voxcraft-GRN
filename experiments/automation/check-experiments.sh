#!/bin/bash
#set -e
#set -x

# run this script from the root (revolve folder): ./experiments/default_study/run-analysis.sh pathPARAMSFILE/PARAMSFILE.sh

DIR="$(dirname "${BASH_SOURCE[0]}")"
study_path="$(basename $DIR)"

if [ $# -eq 0 ]
  then
     params_file=$DIR/paramsdefault.sh
  else
    params_file=$1
fi

source $params_file

# discover unfinished experiments

screen -list

IFS=', ' read -r -a experiments <<< "$experiments"
IFS=', ' read -r -a runs <<< "$runs"

to_do=()
for run in "${runs[@]}"
do
    for experiment in "${experiments[@]}"
    do

     file="${outputs_path}/${study}/${experiment}_${run}.log";
     printf  "\n${file}\n"

     #check experiments status
     if [[ -f "$file" ]]; then

            lastgen=$(grep -c "Finished generation" $file);
            echo " latest finished gen ${lastgen}";

     else
         # not started yet
         echo " None";
     fi

    done
done
