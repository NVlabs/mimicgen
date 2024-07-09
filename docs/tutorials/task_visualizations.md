# Task Visualizations

We provide a convenience script to write videos for each task's reset distribution at `scripts/get_reset_videos.py` for the robosuite tasks we provide in this repository. Set the `OUTPUT_FOLDER` global variable to the folder where you want to write the videos, and set `DATASET_INFOS` appropriately if you would like to limit the environments visualized. Then run the script.

The environments are also readily compatible with robosuite visualization scripts such as the [demo_random_action.py](https://github.com/ARISE-Initiative/robosuite/blob/b9d8d3de5e3dfd1724f4a0e6555246c460407daa/robosuite/demos/demo_random_action.py) script and the [make_reset_video.py](https://github.com/ARISE-Initiative/robosuite/blob/b9d8d3de5e3dfd1724f4a0e6555246c460407daa/robosuite/scripts/make_reset_video.py) script, but you will need to modify these files to add a `import mimicgen` line to make sure that `robosuite` can find these environments.


<div class="admonition note">
<p class="admonition-title">Note</p>

You can find task reset visualizations on the [website](https://mimicgen.github.io), but they may look a little different as they were generated with robosuite v1.2.

</div>