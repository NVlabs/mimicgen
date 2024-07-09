# Launching Several Data Generation Runs

MimicGen inherits the [Config](https://robomimic.github.io/docs/tutorials/configs.html) system from robomimic. Configs are specified as json files, and support both dictionary and "dot" syntax (e.g. `config.experiment.name` and `config["experiment"]["name"]`). 

MimicGen also uses the `ConfigGenerator` class from robomimic, and can use it to generate several config jsons efficiently. For a tutorial on how this generator works, please see the [tutorial from robomimic](https://robomimic.github.io/docs/tutorials/hyperparam_scan.html). In this repository, we grouped several related settings together into helper functions in `utils/config_utils.py` that operate over `ConfigGenerator` objects. 

We furthermore provide examples of how to create and use multiple `ConfigGenerator` objects in scripts such as `scripts/generate_core_configs.py` and `scripts/generate_core_training_configs.py`. These scripts support using multiple base configs (e.g. one per task, since the task spec for each task will be different), and a user can specify different parameter settings per base config (see the `make_generators` function in these files). There are additional settings that are specifed as global variables at the top of the files. These scripts print the file paths for all generated configs, and the commands to launch runs for each config.

Users can easily modify the following files to generate large amounts of data generation and policy learning configs efficiently.

Data Generation:
- `scripts/generate_core_configs.py`
- `scripts/generate_robot_transfer_configs.py`


Policy Learning:
- `scripts/generate_core_training_configs.py`
- `scripts/generate_robot_transfer_training_configs.py`
