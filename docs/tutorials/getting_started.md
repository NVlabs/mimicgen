# Getting Started and Pipeline Overview

<div class="admonition note">
<p class="admonition-title">Note</p>

This section helps users get started with data generation. If you would just like to download our existing datasets and use them with policy learning methods please see the [Reproducing Experiments](https://mimicgen.github.io/docs/tutorials/reproducing_experiments.html) tutorial for a guide, or the [Datasets](https://mimicgen.github.io/docs/datasets/overview.html) page to get details on the datasets.

</div>


## Quick Data Generation Run

Let's run a quick data generation example.

Befor starting, make sure you are at the base repo path:
```sh
$ cd {/path/to/mimicgen}
```

### Step 1: Prepare source human dataset.

MimicGen requires a handful of human demonstrations to get started.

TODO: download square
TODO: note that you could collect your own as well, using teleoperation (e.g. link to robosuite / robomimic) - must be in robomimic hdf5 format

TODO: postprocess square (env interface - where to get information needed during data generation, link to env interface module)

### Step 2: Prepare data generation config.

TODO: get config from template (lots of options for us to configure, but for now, 10 attempts)
TODO: note that config follows RoboMimic config style (link to it)


### Step 3: View data generation outputs.

TODO: compatibility with robomimic, can get info (link to robomimic)
TODO: see dataset successes and failures, statistics in json

## Overview of Typical Data Generation Pipeline

pipeline overview (collect demo, postprocess demo, optionally annotate demo subtask terminations, run data generation, then run policy training)