# Installation

## Requirements

- Mac OS X or Linux machine
- Python >= 3.6 (recommended 3.8.0)
- [conda](https://www.anaconda.com/products/individual) 
  - [virtualenv](https://virtualenv.pypa.io/en/latest/) is also an acceptable alternative, but we assume you have conda installed in our examples below

## Install MimicGen

We recommend installing the repo into a new conda environment (it is called `mimicgen` in the example below):

```sh
conda create -n mimicgen python=3.8
conda activate mimicgen
```

You can install most of the dependencies by cloning the repository and then installing from source:

```sh
cd <PATH_TO_YOUR_INSTALL_DIRECTORY>
git clone https://github.com/NVlabs/mimicgen.git
cd mimicgen
pip install -e .
```

However, there are some additional dependencies that we list below. These are best installed from source:

- [robosuite](https://robosuite.ai/)
    - **Note**: This is optional and only needed if running the examples provided with this repository. The MimicGen source code does not rely on robosuite and can be used with other simulation frameworks.
    - **Installation**
      ```sh
      cd <PATH_TO_YOUR_INSTALL_DIRECTORY>
      git clone https://github.com/ARISE-Initiative/robosuite.git
      git checkout b9d8d3de5e3dfd1724f4a0e6555246c460407daa
      cd robosuite
      pip install -e .
      ```
    - **Note**: the git checkout command corresponds to the commit we used for testing our policy learning results. In general the `master` branch (`v1.4+`) should be fine.
    - For more detailed instructions, see [here](https://robosuite.ai/docs/installation.html)
- [robomimic](https://robomimic.github.io/)
    - **Installation**
      ```sh
      cd <PATH_TO_YOUR_INSTALL_DIRECTORY>
      git clone https://github.com/ARISE-Initiative/robomimic.git
      git checkout d0b37cf214bd24fb590d182edb6384333f67b661
      cd robomimic
      pip install -e .
      ```
    - **Note**: the git checkout command corresponds to the commit we used for testing our policy learning results. In general the `master` branch (`v0.3+`) should be fine.
    - For more detailed instructions, see [here](https://robomimic.github.io/docs/introduction/installation.html)
- [robosuite_task_zoo](https://github.com/ARISE-Initiative/robosuite-task-zoo)
    - **Note**: This is optional and only needed for the Kitchen and Hammer Cleanup environments / datasets.
    - **Installation**
      ```sh
      cd <PATH_TO_YOUR_INSTALL_DIRECTORY>
      git clone https://github.com/ARISE-Initiative/robosuite-task-zoo
      git checkout 74eab7f88214c21ca1ae8617c2b2f8d19718a9ed
      cd robosuite_task_zoo
      pip install -e .
      ```

Lastly, if using robosuite, **please downgrade MuJoCo to 2.3.2**:
```sh
pip install mujoco==2.3.2
```

<div class="admonition warning">
<p class="admonition-title">MuJoCo Version</p>

This MuJoCo version (`2.3.2`) can be important -- in our testing, we found that other versions of MuJoCo could be problematic, especially for the Sawyer arm datasets (e.g. `2.3.5` causes problems with rendering and `2.3.7` changes the dynamics of the robot arm significantly from the collected datasets).
</div>

## Test Your Installation

The following script can be used to try random actions in one of our custom robosuite tasks.
```sh
cd mimicgen/scripts
python demo_random_action.py
```

<div class="admonition note">
<p class="admonition-title">Testing Data Generation</p>

To test data generation please move on to the [Getting Started](https://mimicgen.github.io/docs/tutorials/getting_started.html) tutorial.

</div>
