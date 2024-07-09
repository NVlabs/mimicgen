# Troubleshooting and Known Issues

If you run into an error not documented below, please search through the [GitHub issues](https://github.com/NVlabs/mimicgen/issues), and create a new one if you cannot find a fix.

## Installation

- In our testing on M1 macbook we ran into the following error: `RuntimeError: No ffmpeg exe could be found. Install ffmpeg on your system, or set the IMAGEIO_FFMPEG_EXE environment variable.` Using `conda install ffmpeg` fixed this issue on our end.
- If you run into trouble with installing [egl_probe](https://github.com/StanfordVL/egl_probe) during robomimic installation (e.g. `ERROR: Failed building wheel for egl_probe`) you may need to make sure `cmake` is installed. A simple `pip install cmake` should work.
- If you run into other strange installation issues, one potential fix is to launch a new terminal, activate your conda environment, and try the install commands that are failing once again. One clue that the current terminal state is corrupt and this fix will help is if you see installations going into a different conda environment than the one you have active.

## Policy Learning

- If your robomimic training seems to be proceeding slowly (especially for image-based agents), it might be a problem with robomimic and more modern versions of PyTorch. We recommend PyTorch 1.12.1 (on Ubuntu, we used `conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch`). It is also a good idea to verify that the GPU is being utilized during training.
- If you run into rendering issues with the Sawyer robot arm, or have trouble reproducing our results, your MuJoCo version might be the issue. As noted in the [Installation](https://mimicgen.github.io/docs/introduction/installation.html) section, please use MuJoCo 2.3.2 (`pip install mujoco==2.3.2`).
