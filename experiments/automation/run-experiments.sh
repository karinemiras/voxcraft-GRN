#!/bin/bash
# run this script from the ROOT: ./experiments/automation/run-experiments.sh pathPARAMSFILE/PARAMSFILE.sh

if [ $# -eq 0 ]
  then
    params_file="experiments/noveltysearch.sh"
  else
    params_file=$1
fi

source "$params_file"

# when not needed, just fails
mkdir ${out_path}/${study_name};

screen -d -m -S ${study_name}_loop -L -Logfile ${out_path}/${study_name}/setuploop.log experiments/automation/setup-experiments.sh ${params_file};

### CHEATS: ###

# to check all running exps screens: screen -list
# to stop all running exps: killall screen
# to check a screen: screen -r naaameee
# screen -ls  | egrep "^\s*[0-9]+.screen_" | awk -F "." '{print $1}' |  xargs kill