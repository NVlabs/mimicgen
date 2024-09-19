# MimicGen (CoRL 2023)

In this section, we provide an overview of the datasets we released with the MimicGen paper. These are the same exact datasets used in our paper (but postprocessed to support a higher version of robosuite). We also show how to easily reproduce policy learning results on this data.

<div class="admonition note">
<p class="admonition-title">Note</p>

If you would like to reproduce our data generation results in addition to the policy learning results, please see the [Reproducing Experiments](https://mimicgen.github.io/docs/tutorials/reproducing_experiments.html) tutorial for a comprehensive guide.

</div>

## Downloading and Using Datasets

### Dataset Types

As described in the paper, each task has a default reset distribution (D_0). Source human demonstrations (usually 10 demos) were collected on this distribution and MimicGen was subsequently used to generate large datasets (usually 1000 demos) across different task reset distributions (e.g. D_0, D_1, D_2), objects, and robots.

The datasets are split into different types:

- **source**: source human datasets used to generate all data -- this generally consists of 10 human demonstrations collected on the D_0 variant for each task.
- **core**: datasets generated with MimicGen for different task reset distributions. These correspond to the core set of results in Figure 4 of the paper.
- **object**: datasets generated with MimicGen for different objects. These correspond to the results in Appendix G of the paper.
- **robot**: datasets generated with MimicGen for different robots. These correspond to the results in Appendix F of the paper.
- **large_interpolation**: datasets generated with MimicGen using much larger interpolation segments. These correspond to the results in Appendix H in the paper.

<div class="admonition note">
<p class="admonition-title">Note</p>

We found that the large_interpolation datasets pose a significant challenge for imitation learning, and have substantial room for improvement.

</div>

### Dataset Statistics

The datasets contain over 48,000 task demonstrations across 12 tasks.

We provide more information on the amount of demonstrations for each dataset type:
- **source**: 120 human demonstrations across 12 tasks (10 per task) used to automatically generate the other datasets
- **core**: 26,000 task demonstrations across 12 tasks (26 task variants)
- **object**: 2000 task demonstrations on the Mug Cleanup task with different mugs
- **robot**: 16,000 task demonstrations across 4 different robot arms on 2 tasks (4 task variants)
- **large_interpolation**: 6000 task demonstrations across 6 tasks that pose significant challenges for modern imitation learning methods

### Dataset Download

#### Method 1: Using `download_datasets.py` (Recommended)

`download_datasets.py` (located at `mimicgen/scripts`) is a python script that provides a programmatic way of downloading the datasets. This is the preferred method, because this script also sets up a directory structure for the datasets that works out of the box with the code for reproducing policy learning results.

A few examples of using this script are provided below:

```sh
# default behavior - just download core square_d0 dataset
python download_datasets.py

# download core datasets for square D0, D1, D2 and coffee D0, D1, D2
python download_datasets.py --dataset_type core --tasks square_d0 square_d1 square_d2 coffee_d0 coffee_d1 coffee_d2

# download all core datasets, but do a dry run first to see what will be downloaded and where
python download_datasets.py --dataset_type core --tasks all --dry_run

# download all source human datasets
python download_datasets.py --dataset_type source --tasks all
```

#### Method 2: Using Hugging Face

You can download the datasets through Hugging Face by using direct download links from the HF dataset page, or by using their API.

**Hugging Face dataset repository:** [link](https://huggingface.co/datasets/amandlek/mimicgen_datasets)


## Reproducing Policy Learning Results

After downloading the appropriate datasets you’re interested in using by running the `download_datasets.py` script, the `generate_training_configs_for_public_datasets.py` script (located at `mimicgen/scripts`) can be used to generate all training config json files necessary to reproduce the experiments in the paper. A few examples are below.

```sh
# Assume datasets already exist in mimicgen/../datasets folder. Configs will be generated under mimicgen/exps/paper, and training results will be at mimicgen/../training_results after launching training runs.
python generate_training_configs_for_public_datasets.py

# Alternatively, specify where datasets exist, and specify where configs should be generated.
python generate_training_configs_for_public_datasets.py --config_dir /tmp/configs --dataset_dir /tmp/datasets --output_dir /tmp/experiment_results
```

Then, to reproduce a specific set of training runs for different experiment groups (see [Dataset Types](#dataset-types)), we can simply navigate to the generated config directory, and copy training commands from the generated shell script there. As an example, we can reproduce the image training results on the Coffee D0 dataset, by looking for the correct set of commands in `mimicgen/exps/paper/core.sh` and running them. The relevant section of the shell script is reproduced below.

```sh
#  task: coffee_d0
#    obs modality: image
python /path/to/robomimic/scripts/train.py --config /path/to/mimicgen/exps/paper/core/coffee_d0/image/bc_rnn.json
```

<div class="admonition note">
<p class="admonition-title">Note</p>

In the MimicGen paper, we generated our datasets on versions of environments built on robosuite `v1.2`. Since then, we changed the environments and datasets (through postprocessing) to be based on robosuite `v1.4`. However, `v1.4` has some visual and dynamics differences from `v1.2`, so the learning results may not exactly match up with the ones we reported in the paper. In our testing on these released datasets, we were able to reproduce nearly all of our results, but within 10% of the performance reported in the paper.

</div>
