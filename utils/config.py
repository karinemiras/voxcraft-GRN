import argparse


class Config():

    def _get_params(self):
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--out_path",
            required=False,
            default="/working_data",
            type=str,
            help="path for results files"
        )

        parser.add_argument(
            "--docker_path",
            required=False,
            default="/workspace",
            type=str,
            help="path for docker"
        )

        parser.add_argument(
            "--study_name",
            required=False,
            default="defaultstudy",
            type=str,
            help="",
        )

        parser.add_argument(
            "--algorithm",
            required=False,
            default="basic_EA",
            type=str,
            help="",
        )

        parser.add_argument(
            "--population_size",
            required=False,
            default=50, #4
            type=int,
        )

        parser.add_argument(
            "--offspring_size",
            required=False,
            default=50, #4
            type=int,
        )

        parser.add_argument(
            "--num_generations",
            required=False,
            default=50, #2
            type=int,
        )

        parser.add_argument(
            "--simulation_time",
            required=False,
            default=4.4,
            type=int,
        )

        parser.add_argument(
            "--tournament_k",
            required=False,
            default=4,
            type=int,
        )

        parser.add_argument(
            "--max_voxels",
            required=False,
            default=15,
            type=int,
            help="",
        )

        parser.add_argument(
            "--tfs",
            required=False,
            default="reg2m3",
            type=str,
            help="list of tfs config",
        )

        parser.add_argument(
            "--cube_face_size",
            required=False,
            default=3,
            type=int,
            help="",
        )

        parser.add_argument(
            "--plastic",
            required=False,
            default=0,
            type=int,
            help="0 is not plastic, 1 is plastic",
        )

        parser.add_argument(
            "--env_conditions",
            required=False,
            default='',
            type=str,
            help="params that define environmental conditions and/or task",
        )

        parser.add_argument(
            "--crossover_prob",
            required=False,
            default=1,
            type=float,
        )

        parser.add_argument(
            "--mutation_prob",
            required=False,
            default=0.9,
            type=float,
        )

        parser.add_argument(
            "--fitness_metric",
            required=False,
            default="fitness",
            type=str,
        )

        parser.add_argument(
            "--generations",
            required=False,
            default="",
            type=str,
            help="list of generations of be analyzed",
        )

        parser.add_argument(
            "--final_gen",
            required=False,
            default="",
            type=str,
            help="last generation to be analyzed"
        )

        parser.add_argument(
            "--experiment_name",
            required=False,
            default="defaultexperiment",
            type=str,
            help="Name of the experiment.",
        )

        parser.add_argument(
            "--experiments",
            required=False,
            default="",
            type=str,
            help="list of experiment_name",
        )

        parser.add_argument(
            "--run",
            required=False,
            default=1,
            type=int,
            help="",
        )

        parser.add_argument(
            "--runs",
            required=False,
            default="",
            type=str,
            help="list of all runs",
        )

        parser.add_argument(
            "--run_simulation",
            required=False,
            default=1,
            type=int,
            help="If 0, runs optimizer without simulating robots, so behavioral measures are none."
        )

        args = parser.parse_args()

        return args

