# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

__version__ = "1.0.0"

# try to import all environment interfaces here
try:
    from mimicgen.env_interfaces.robosuite import *
except ImportError as e:
    print("WARNING: robosuite environment interfaces not imported...")
    print("Got error: {}".format(e))

# import tasks to make sure they are added to robosuite task registry
try:
    from mimicgen.envs.robosuite.threading import *
    from mimicgen.envs.robosuite.coffee import *
    from mimicgen.envs.robosuite.three_piece_assembly import *
    from mimicgen.envs.robosuite.mug_cleanup import *
    from mimicgen.envs.robosuite.stack import *
    from mimicgen.envs.robosuite.nut_assembly import *
    from mimicgen.envs.robosuite.pick_place import *
except ImportError as e:
    print("WARNING: robosuite environments not imported...")
    print("Got error: {}".format(e))

try:
    from mimicgen.envs.robosuite.hammer_cleanup import *
    from mimicgen.envs.robosuite.kitchen import *
except ImportError as e:
    print("WARNING: robosuite task zoo environments not imported, possibly because robosuite_task_zoo is not installed...")
    print("Got error: {}".format(e))

# stores released dataset links and rollout horizons in global dictionary.
# Structure is given below for each type of dataset:

# robosuite
# {
#   task:
#       url: link
#       horizon: value
#   ...
# }
DATASET_REGISTRY = {}


def register_dataset_link(dataset_type, task, link, horizon):
    """
    Helper function to register dataset link in global dictionary.
    Also takes a @horizon parameter - this corresponds to the evaluation
    rollout horizon that should be used during training.

    Args:
        dataset_type (str): identifies the type of dataset (e.g. source human data, 
            core experiment data, object transfer data)
        task (str): name of task for this dataset
        link (str): download link for the dataset
        horizon (int): evaluation rollout horizon that should be used with this dataset
    """
    if dataset_type not in DATASET_REGISTRY:
        DATASET_REGISTRY[dataset_type] = dict()
    DATASET_REGISTRY[dataset_type][task] = dict(url=link, horizon=horizon)


def register_all_links():
    """
    Record all dataset links in this function.
    """

    ### source human datasets used to generate all data ###
    dataset_type = "source"

    # info for each dataset (name, evaluation horizon, link)
    dataset_infos = [
        ("hammer_cleanup", 500, "https://drive.google.com/file/d/15EENNeAjm0nhaA2DxszUfvKBbm7tMKP8/view?usp=drive_link"),
        ("kitchen", 800, "https://drive.google.com/file/d/15OSYVQBKWjA_0Qb7vgSJxg0ePk5tPGHO/view?usp=drive_link"),
        ("coffee", 400, "https://drive.google.com/file/d/15LLftHGAzKw-t--KmA4Q9esNxSef8lmI/view?usp=drive_link"),
        ("coffee_preparation", 800, "https://drive.google.com/file/d/15KlgnIurTeHsUakHvWixVXtA7Bh7A6Gt/view?usp=drive_link"),
        ("nut_assembly", 500, "https://drive.google.com/file/d/150oTa-yEHxSsOduiiai0CpQ1pfPY14PF/view?usp=drive_link"),
        ("mug_cleanup", 500, "https://drive.google.com/file/d/15JHCOZabMN6XBHj_cXsS0QPPsJeQiPAN/view?usp=drive_link"),
        ("pick_place", 1000, "https://drive.google.com/file/d/15U2_Qm9y8CQ3HF6c-HbVJMtyEdeEccZv/view?usp=drive_link"),
        ("square", 400, "https://drive.google.com/file/d/15CCPUGukZqJmFoFRDYDadu7lIVot_2hC/view?usp=drive_link"),
        ("stack", 400, "https://drive.google.com/file/d/1519sVqkLD6PlI2pir8yjCpyogX1PfjUP/view?usp=drive_link"),
        ("stack_three", 400, "https://drive.google.com/file/d/151ur_DIhO2Nlp3ipnKuQlcVK_IkE2Ago/view?usp=drive_link"),
        ("threading", 400, "https://drive.google.com/file/d/15CzLAf_tAjwWnAFIaWsiyoPNYb3m84IK/view?usp=drive_link"),
        ("three_piece_assembly", 500, "https://drive.google.com/file/d/159aWGouuiKOsf8YblSR5Lkfq9d1aCVCV/view?usp=drive_link"),
    ]
    for task, horizon, link in dataset_infos:
        register_dataset_link(
            dataset_type=dataset_type,
            task=task,
            horizon=horizon,
            link=link,
        )

    ### core generated datasets ###
    dataset_type = "core"
    dataset_infos = [
        ("hammer_cleanup_d0", 500, "https://drive.google.com/file/d/1uLQSFqTiRquUbe3NprHSCVhLyOqjjrVR/view?usp=drive_link"),
        ("hammer_cleanup_d1", 500, "https://drive.google.com/file/d/1YL-cSs9dC3lsA3LxQVZ0w98ijTnSdbSH/view?usp=drive_link"),
        ("kitchen_d0", 800, "https://drive.google.com/file/d/1RPu6xTx8SFL5k9XpYUoR8Y5eZsbhqojj/view?usp=drive_link"),
        ("kitchen_d1", 800, "https://drive.google.com/file/d/12X7p60JpDkyD4Ia8gjn0qn6VNy7RFuLX/view?usp=drive_link"),
        ("coffee_d0", 400, "https://drive.google.com/file/d/1-0gQILd2jkhiOqTuh_bpP8wnidHrZXr2/view?usp=drive_link"),
        ("coffee_d1", 400, "https://drive.google.com/file/d/1rsOhOzlJnimXxGM9Oi7S9by1UpgSrdQK/view?usp=drive_link"),
        ("coffee_d2", 400, "https://drive.google.com/file/d/11X2d6WsRq1rQZzxTD9Gd23562VQ0k7OW/view?usp=drive_link"),
        ("coffee_preparation_d0", 800, "https://drive.google.com/file/d/1OsEvnTHDQDzsfjkkt6IFU3QDFOT1OGGd/view?usp=drive_link"),
        ("coffee_preparation_d1", 800, "https://drive.google.com/file/d/1trJlVyq9xTRARHBOMi8TOcxLDi804AN3/view?usp=drive_link"),
        ("nut_assembly_d0", 500, "https://drive.google.com/file/d/1N3Q2NJwn-Wt4OBS8Q04mit92uqrOxqBV/view?usp=drive_link"),
        ("mug_cleanup_d0", 500, "https://drive.google.com/file/d/1VV2PkvlTT0fGmc6MwwJR8bAtop0hpjIJ/view?usp=drive_link"),
        ("mug_cleanup_d1", 500, "https://drive.google.com/file/d/1bxJyN2c2yZsgn2FOGWFJsxlgHXWD3eiI/view?usp=drive_link"),
        ("pick_place_d0", 1000, "https://drive.google.com/file/d/1usOS0sbtmD0wB0L8KhxSjxo6uUmKNT60/view?usp=drive_link"),
        ("square_d0", 400, "https://drive.google.com/file/d/1FFMWZPzliM4QoiBxbuU69DGfZ4rmt9LL/view?usp=drive_link"),
        ("square_d1", 400, "https://drive.google.com/file/d/1LJfdITKFQTfPmETVVUjj9YTwiFcmDleZ/view?usp=drive_link"),
        ("square_d2", 400, "https://drive.google.com/file/d/1X8KCL1eSLT0aieIbFFWOMDb3H2z_czv5/view?usp=drive_link"),
        ("stack_d0", 400, "https://drive.google.com/file/d/1ZhPBfglfashd8yVwtb4HpDjcxHj_8oAX/view?usp=drive_link"),
        ("stack_d1", 400, "https://drive.google.com/file/d/1yw9XvvRm4WIsxsFVSR0MOuhM_VgvVdPw/view?usp=drive_link"),
        ("stack_three_d0", 400, "https://drive.google.com/file/d/1AzuUPtC8K5ZKiuvKAJ3UJ-by1UJqWeLX/view?usp=drive_link"),
        ("stack_three_d1", 400, "https://drive.google.com/file/d/1PawNzhGCroHdU-4Rl3ZoC7N-6Fj_fErS/view?usp=drive_link"),
        ("threading_d0", 400, "https://drive.google.com/file/d/1JYIIwRE31ulUYDV0BqrvnzBWzLiVqcKb/view?usp=drive_link"),
        ("threading_d1", 400, "https://drive.google.com/file/d/1t2Aduv9yic23RlKXg2vryV9jCLwaoFqu/view?usp=drive_link"),
        ("threading_d2", 400, "https://drive.google.com/file/d/1FUKnUN746m9C7-ReA-o2s58Y8eRyL0oY/view?usp=drive_link"),
        ("three_piece_assembly_d0", 500, "https://drive.google.com/file/d/1xyTJcrNagEk57Wdoq1YMxnja7ljZC9JW/view?usp=drive_link"),
        ("three_piece_assembly_d1", 500, "https://drive.google.com/file/d/1HLz9RstJvwkzxUphK2SC9_k_ijv6qFMj/view?usp=drive_link"),
        ("three_piece_assembly_d2", 500, "https://drive.google.com/file/d/1v59INEmTaMdyiivD3n37J-Oe81xBtQrQ/view?usp=drive_link"),
    ]
    for task, horizon, link in dataset_infos:
        register_dataset_link(
            dataset_type=dataset_type,
            task=task,
            horizon=horizon,
            link=link,
        )

    ### object transfer datasets ###
    dataset_type = "object"
    dataset_infos = [
        ("mug_cleanup_o1", 500, "https://drive.google.com/file/d/14llT-UQNjQrDQRdvlq9luvmDzfgdMKOi/view?usp=drive_link"),
        ("mug_cleanup_o2", 500, "https://drive.google.com/file/d/10M-XTciDT7qQfZTfhL_WEwicH7QIg8QJ/view?usp=drive_link"),
    ]
    for task, horizon, link in dataset_infos:
        register_dataset_link(
            dataset_type=dataset_type,
            task=task,
            horizon=horizon,
            link=link,
        )

    ### robot transfer datasets ###
    dataset_type = "robot"
    dataset_infos = [
        ("square_d0_panda", 400, "https://drive.google.com/file/d/1-Ez7dG2cUgiLveKWdZnLyq8k_FWZMsG_/view?usp=drive_link"),
        ("square_d0_sawyer", 400, "https://drive.google.com/file/d/1U3J39kc9k49T1RxcWCk_ua_dOAgqO_Rz/view?usp=drive_link"),
        ("square_d0_iiwa", 400, "https://drive.google.com/file/d/1bk5m9-hUQAHGkeJWzml-k7O30A7VF0Ab/view?usp=drive_link"),
        ("square_d0_ur5e", 400, "https://drive.google.com/file/d/14NFNlAN4aCwjieDDMbmKFbDxCxp4Frvf/view?usp=drive_link"),
        ("square_d1_panda", 400, "https://drive.google.com/file/d/1yXk6G0RqTrdSy6AppGcOfBKlh4DbbRgg/view?usp=drive_link"),
        ("square_d1_sawyer", 400, "https://drive.google.com/file/d/1V-F6PK4Tqy0aQ0iHwEfVop-9vLOAaySc/view?usp=drive_link"),
        ("square_d1_iiwa", 400, "https://drive.google.com/file/d/1oRFRev_PpIyNItV187h6gvabEQFIpXgP/view?usp=drive_link"),
        ("square_d1_ur5e", 400, "https://drive.google.com/file/d/1DSLG_2S6fgJRr2JZ_EBU0CdoRQ0sYuK7/view?usp=drive_link"),
        ("threading_d0_panda", 400, "https://drive.google.com/file/d/17MM_OPgtht2KbnZjG_J-K37wPD3XAEOd/view?usp=drive_link"),
        ("threading_d0_sawyer", 400, "https://drive.google.com/file/d/1v3fw7EjXe6_DP62rs2Q2zpMfOP6mHIp2/view?usp=drive_link"),
        ("threading_d0_iiwa", 400, "https://drive.google.com/file/d/13mqHSSYvTx4KiUCytkOHocPDj5xFw2sP/view?usp=drive_link"),
        ("threading_d0_ur5e", 400, "https://drive.google.com/file/d/13tNcKtel5nuC5_lA7u8B6K4tC6oGfMwq/view?usp=drive_link"),
        ("threading_d1_panda", 400, "https://drive.google.com/file/d/1g21kbOnFNOfC4F8PZYpk0vngYoPM4duQ/view?usp=drive_link"),
        ("threading_d1_sawyer", 400, "https://drive.google.com/file/d/1HttzNikV5uVn5KfzEXVR0nu4xEkH51Xv/view?usp=drive_link"),
        ("threading_d1_iiwa", 400, "https://drive.google.com/file/d/1MUZh8XqARR7Ei0RR5FoQdq1cq7L6kxmR/view?usp=drive_link"),
        ("threading_d1_ur5e", 400, "https://drive.google.com/file/d/1SecpwDmaIwFFe01I5FVLS2kPvhzSrFeE/view?usp=drive_link"),
    ]
    for task, horizon, link in dataset_infos:
        register_dataset_link(
            dataset_type=dataset_type,
            task=task,
            horizon=horizon,
            link=link,
        )

    ### large_interpolation datasets (larger eval horizons due to longer interpolation segments) ###
    dataset_type = "large_interpolation"
    dataset_infos = [
        ("coffee_d1", 550, "https://drive.google.com/file/d/1nslJUf5bV1wMNSZWsdeMWg3b9iBv9g5F/view?usp=drive_link"),
        ("pick_place_d0", 1400, "https://drive.google.com/file/d/1LylKLOzZfv9L47qZl5nOgvTjA2HQysqI/view?usp=drive_link"),
        ("square_d1", 500, "https://drive.google.com/file/d/1BUynvXIJxFhMXu4zup-Yo6vw8Ki93u5J/view?usp=drive_link"),
        ("stack_d1", 500, "https://drive.google.com/file/d/18OXuDBbbRN0_hPHbskN2R0ZnRMpM22IW/view?usp=drive_link"),
        ("threading_d1", 500, "https://drive.google.com/file/d/1TOEfBiHNxNIGmyBRcimq2_g-VGMdfrHy/view?usp=drive_link"),
        ("three_piece_assembly_d1", 700, "https://drive.google.com/file/d/1MoUBC05iXQD5_y65oFwq3gCvke6JcE5-/view?usp=drive_link"),
    ]
    for task, horizon, link in dataset_infos:
        register_dataset_link(
            dataset_type=dataset_type,
            task=task,
            horizon=horizon,
            link=link,
        )

register_all_links()
