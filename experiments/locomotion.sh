#!/bin/bash


### PARAMS INI ###

# this should be the path for the output files (choose YOUR OWN dir!)
out_path="/working_data"
# /home/ripper8/projects/working_data

docker_path="/workspace"
# /home/ripper8/projects/voxcraft

# DO NOT use underline ( _ ) in the study and experiments names
# delimiter of three vars below is coma. example:
#experiments="exp1,epx2"
# exps order is the same for all three vars
# exps names should not be fully contained in each other

study_name="soft"
experiments="highfricbone,highfricNObone,lowfricbone,lowfricNObone"

# one voxel_types definition per experiment
voxel_types="withbone,nobone,withbone,nobone"

# one set of conditions per experiment
env_conditions="none,none,none,none"

ustatic="1,1,0.1,0.1"
udynamic="0.8,0.8,0.1,0.1"

####

nruns=10

runs=""
for i in $(seq 1 $nruns);
do
  runs=("${runs}${i},")
done
runs=${runs::-1}

watchruns=$runs

algorithm="basic_EA"

fitness_metric="novelty_weighted"

plastic=0

num_generations="100"

population_size="50"

offspring_size="50"

# gens for box-plots, snapshots, videos (by default the last gen)
#generations="1,$num_generations"
generations="1,$num_generations"

# max gen to filter line-plots  (by default the last gen)
final_gen="$num_generations"

mutation_prob=0.9

crossover_prob=1

max_voxels=64

cube_face_size=4

simulation_time=2

run_simulation=1

### PARAMS END ###