#!/bin/bash
# run this script from the root (folder): ./experiments/automation/run-experiments.sh pathPARAMSFILE/PARAMSFILE.sh

DIR="$(dirname "${BASH_SOURCE[0]}")"
study="$(basename $DIR)"

if [ $# -eq 0 ]
  then
     params_file=$DIR/paramsdefault.sh
  else
    params_file=$1
fi

source $params_file

mkdir ${outputs_path}/${study};
screen -d -m -S ${study}_loop -L -Logfile ${outputs_path}/${study}/setuploop.log $DIR/setup-experiments.sh ${params_file};

### CHEATS: ###

# to check all running exps screens: screen -list
# to stop all running exps: killall screen
# to check a screen: screen -r naaameee
# screen -ls  | egrep "^\s*[0-9]+.screen_" | awk -F "." '{print $1}' |  xargs kill