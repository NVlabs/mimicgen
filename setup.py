from setuptools import setup, find_packages
import os

# read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    lines = f.readlines()

# remove images from README
lines = [x for x in lines if (('.png' not in x) and ('.gif' not in x))]
long_description = ''.join(lines)

def package_files(directory):
    paths = []
    for (path, _, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

extra_files = package_files('mimicgen/exps')
extra_files.extend(package_files('mimicgen/models'))

setup(
    name="mimicgen",
    packages=[
        package for package in find_packages() if package.startswith("mimicgen")
    ],
    install_requires=[
        "numpy>=1.13.3,<2",
        "h5py",
        "tqdm",
        "imageio",
        "imageio-ffmpeg",
        "gdown",
        "chardet",
    ],
    eager_resources=['*'],
    include_package_data=True,
    package_data={
        'mimicgen': extra_files
    },
    python_requires='>=3',
    description="MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations",
    author="Ajay Mandlekar",
    url="https://github.com/NVlabs/mimicgen",
    author_email="amandlekar@nvidia.com",
    version="1.0.0",
    long_description=long_description,
    long_description_content_type='text/markdown'
)
