#!/bin/bash


### PARAMS INI ###

# this should be the path for the output files (choose YOUR OWN dir!)
out_path="/working_data"
docker_path="/workspace"

# DO NOT use underline ( _ ) in the study and experiments names
# delimiter of three vars below is coma. example:
#experiments="exp1,epx2"
# exps order is the same for all three vars
# exps names should not be fully contained in each other

study_name="voxlocnov"
experiments="crosspropxyv4s25"

# one tf definition per experiment
tfs="reg2"

# one set of conditions per experiment
env_conditions="none"

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

fitness_metric="local_novelty"

plastic=0

num_generations="25"

population_size="50"

offspring_size="50"

# gens for box-plots, snapshots, videos (by default the last gen)
#generations="1,$num_generations"
generations="1,$num_generations"

# max gen to filter line-plots  (by default the last gen)
final_gen="$num_generations"

mutation_prob=0.9

crossover_prob=1

max_voxels=25

cube_face_size=4

simulation_time=0

run_simulation=0

### PARAMS END ###