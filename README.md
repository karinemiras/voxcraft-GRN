# How to use 

- Install the simulator Voxcraft within Docker: [voxcraft-sim](https://github.com/voxcraft/voxcraft-sim)
 [note that GoogleCollab is NOT necessary]

- install requirements.txt

> To generate a voxel robot represented by a random numpy use:  random_np_to_voxels.py 

> To generate a voxel robot represented by a random Gene Regulatory Network use:  random_grn_to_voxels.py 

> To run a single full experiment use: python algorithms/basic_EA.py  --out_path mydatapath  --docker_path mydockerpath

> To run a batch of experiments and analysis according to a parameter file use:
./experiments/automation/run-experiments.sh experiments/locomotion.sh 

> Vizualizing robots will require voxcraft-viz (multiple Voxcraft tutorials can be found [here](https://voxcraft.github.io/design) )

> The full list of params is in util/config.py