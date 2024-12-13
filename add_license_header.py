# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Adds license header to new files that don't have the license header.
"""

import os

LICENSE_TEXT = """# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].
"""

def check_license_exists(filename):
    with open(filename, 'r') as f:
        lines_to_check = LICENSE_TEXT.strip().split('\n')
        for line in lines_to_check:
            file_line = f.readline().strip()
            if file_line != line:
                return False
        return True

def add_license_header(filename):
    if not check_license_exists(filename):
        with open(filename, 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write(LICENSE_TEXT + '\n' + content)

if __name__ == "__main__":
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                add_license_header(os.path.join(root, file))