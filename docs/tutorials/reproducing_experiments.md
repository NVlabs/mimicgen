# Reproducing Published Experiments and Results

There are two options for reproducing the set of results in the MimicGen paper.

## Option 1: Download datasets and run policy learning

You can directly download the datasets we generated and used in the MimicGen paper, and then subsequently run policy learning on the downloaded data. See the [Datasets](https://mimicgen.github.io/docs/datasets/mimicgen_corl_2023.html) page for exact instructions on how to do this.

## Option 2: Run data generation and then policy learning

In this section, we show how to run MimicGen on the source demonstrations we released to generate datasets equivalent to the ones we produced in our paper, and then subsequently train policies on the generated data.

<div class="admonition note">
<p class="admonition-title">Note</p>

We recommend going through the [Getting Started](https://mimicgen.github.io/docs/tutorials/getting_started.html) tutorial first, so that you are familiar with the way data generation works.

</div>

The steps are very similar to the steps taken for the quick data generation run in the [Getting Started](https://mimicgen.github.io/docs/tutorials/getting_started.html) tutorial. We provide a brief outline of the steps and important changes below.

Before starting, make sure you are at the base repo path:
```sh
$ cd {/path/to/mimicgen}
```

### Step 1: Prepare source human datasets

Download all source demonstrations of interest. You can download all of them with the command below (optionally provide the `--download_dir` argument to set the download path):
```sh
$ python mimicgen/scripts/download_datasets.py --dataset_type source --tasks all
```

We need to prepare each one for data generation. The bash script `scripts/prepare_all_src_datasets.sh` outline the commands for each source dataset. We provide the command for Coffee below:
```sh
$ python mimicgen/scripts/prepare_src_dataset.py --dataset datasets/source/coffee.hdf5 --env_interface MG_Coffee --env_interface_type robosuite
```

### Step 2: Prepare data generation configs

Open `scripts/generate_core_configs.py` and set `NUM_TRAJ = 1000` and `GUARANTEE = True` -- this means we will keep generating data until we generate 1000 successful trajectories. You can set additional parameters at the top of the file as well, e.g. in case you would like to change where data is generated. 

Next, run the script:
```sh
$ python mimicgen/scripts/generate_core_configs.py
```

The generated configs correspond to the **core** dataset type described on the [Datasets](https://mimicgen.github.io/docs/datasets/mimicgen_corl_2023.html) page, and the **object** dataset type as well (Mug Cleanup O1 and O2).

If you would also like to generate configs for the **robot** dataset type (robot transfer experiments), you can follow the same steps above for `scripts/generate_robot_transfer_configs.py`.

### Step 3: Run data generation

The scripts above print lines that correspond to data generation runs for each config. You can pick and choose which ones you would like to run and then launch them with `scripts/generate_dataset.py`.

### Step 4: Run policy training

Finally, you can run policy training on the generated data -- to reproduce the paper results, you can run BC-RNN. To make this easy, we provide `scripts/generate_core_training_configs.py` and `scripts/generate_robot_transfer_training_configs.py`. As before, you can configure some settings in global variables at the top of the file, such as where to store policy training results. These scripts generate robomimic training configs that can be run with `scripts/train.py` in the robomimic repository.
