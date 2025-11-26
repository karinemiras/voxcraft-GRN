#!/bin/bash
# run this script from the ROOT (inside docker): ./experiments/automation/setup-experiments.sh pathPARAMSFILE/PARAMSFILE.sh
#set -e
#set -x

if [ $# -eq 0 ]
  then
    params_file="experiments/locomotion.sh"
  else
    params_file=$1
fi

source "$params_file"

# when not needed, just fails
mkdir ${out_path}/${study_name}
mkdir ${out_path}/${study_name}/analysis


possible_screens=()

# defines possible ports for screens
for t in $(seq 1 $((${num_terminals}))); do
    possible_screens+=($t)
done

# unpack params
IFS=', ' read -r -a experiments <<< "$experiments"
IFS=', ' read -r -a tfs <<< "$tfs"

while true
	do

    printf "\n  >>>> loop ... \n"

    # discover free terminals

    active_screens=()
    free_screens=()
    active_experiments=()

    declare -a arr="$(screen -list)"

    for obj in ${arr[@]}; do

       screenstudy="$(cut -d'_' -f2 <<<"$obj")"

          if [[ "$screenstudy" == "${study_name}" ]]; then
           printf "\n screen ${obj} is on\n"
              screen="$(cut -d'_' -f3 <<<"$obj")"

              active_experiments+=("$(cut -d'_' -f4 -<<<"$obj")_$(cut -d'_' -f5 -<<<"$obj")")
              active_screens+=($screen)
          fi
    done

   for possible_screen in "${possible_screens[@]}"; do
       if [[ ! " ${active_screens[@]} " =~ " ${possible_screen} " ]]; then
           free_screens+=($possible_screen)
     fi
      done


    # discover unfinished experiments
    to_do=()
    unfinished=()
    for i in $(seq $nruns)
    do
        run=$(($i))

        for experiment in "${experiments[@]}"
        do

         file="${out_path}/${study_name}/${experiment}_${run}.log";
     
         #check experiments status
         if [[ -f "$file" ]]; then

              lastgen=$(grep -c "Finished generation" $file);

               if [ "$lastgen" != "$num_generations" ]; then

                 unfinished+=("${experiment}_${run}")

                # only if not already running
                if [[ ! " ${active_experiments[@]} " =~ " ${experiment}_${run} " ]]; then
                   to_do+=("${experiment}_${run}")
                fi
             fi
         else
               # not started yet
               # echo " None";
               unfinished+=("${experiment}_${run}")
               # only if not already running
                if [[ ! " ${active_experiments[@]} " =~ " ${experiment}_${run} " ]]; then
                   to_do+=("${experiment}_${run}")
                fi
         fi

        done
    done


    # spawns N experiments (N is according to free screens)

    max_fs=${#free_screens[@]}
    to_do_now=("${to_do[@]:0:$max_fs}")

    p=0
    for to_d in "${to_do_now[@]}"; do

        exp=$(cut -d'_' -f1 <<<"${to_d}")
        run=$(cut -d'_' -f2 <<<"${to_d}")
        idx=$( echo ${experiments[@]/${exp}//} | cut -d/ -f1 | wc -w | tr -d ' ' )

        # nice -n19 python3  experiments/${study_name}/${algorithm}.py

    #  screen -d -m -S _${study_name}_${free_screens[$p]}_${to_d} -L -Logfile ${out_path}/${study_name}/${exp}_${run}".log" \
#               python3  ${docker_path}/algorithms/${algorithm}.py --out_path ${out_path} \
#               --experiment_name ${exp} --env_conditions ${env_conditions} --run ${run} --study_name=${study_name} \
#               --num_generations ${num_generations} --population_size ${population_size} --offspring_size ${offspring_size} \
#               --simulation_time ${simulation_time} --docker_path ${docker_path} \
#               --crossover_prob ${crossover_prob} --mutation_prob ${mutation_prob}  \
#               --max_voxels ${max_voxels}  --tfs ${tfs[$idx]}  --run_simulation ${run_simulation} \
#               ;

        python3  ${docker_path}/algorithms/${algorithm}.py --out_path ${out_path} \
         --experiment_name ${exp} --env_conditions ${env_conditions} --run ${run} --study_name=${study_name} \
         --num_generations ${num_generations} --population_size ${population_size} --offspring_size ${offspring_size} \
         --simulation_time ${simulation_time} --docker_path ${docker_path} \
         --crossover_prob ${crossover_prob} --mutation_prob ${mutation_prob}  \
         --max_voxels ${max_voxels}  --tfs ${tfs[$idx]} --run_simulation ${run_simulation} \
         ;

        printf "\n >> (re)starting ${study_name}_${free_screens[$p]}_${to_d} \n\n"
        p=$((${p}+1))

    done

   # if all experiments are finished, run analysis and make videos
   # (NOTE: IF THE SCREEN IS LOCKED, YOU JUST GET VIDEO WITH A LOCKED SCREEN...)

   if [ -z "$unfinished" ]; then

      printf "\n analysis...\n"
      ./experiments/automation/run-analysis.sh $params_file

      #./experiments/automation/watch_and_record.sh $params_file

      pkill -f ${study_name}_loop
      exit;
   fi

    # use this longer period for longer experiments
    sleep $delay_setup_script;

done



