#!/bin/bash

# Store commands used to prepare source datasets for each environment.


# coffee
python prepare_src_dataset.py \
    --dataset ../../datasets/source/coffee.hdf5 \
    --env_interface MG_Coffee \
    --env_interface_type robosuite

# coffee_preparation
python prepare_src_dataset.py \
    --dataset ../../datasets/source/coffee_preparation.hdf5 \
    --env_interface MG_CoffeePreparation \
    --env_interface_type robosuite

# hammer_cleanup
python prepare_src_dataset.py \
    --dataset ../../datasets/source/hammer_cleanup.hdf5 \
    --env_interface MG_HammerCleanup \
    --env_interface_type robosuite

# kitchen
python prepare_src_dataset.py \
    --dataset ../../datasets/source/kitchen.hdf5 \
    --env_interface MG_Kitchen \
    --env_interface_type robosuite

# mug_cleanup
python prepare_src_dataset.py \
    --dataset ../../datasets/source/mug_cleanup.hdf5 \
    --env_interface MG_MugCleanup \
    --env_interface_type robosuite

# nut_assembly
python prepare_src_dataset.py \
    --dataset ../../datasets/source/nut_assembly.hdf5 \
    --env_interface MG_NutAssembly \
    --env_interface_type robosuite

# pick_place
python prepare_src_dataset.py \
    --dataset ../../datasets/source/pick_place.hdf5 \
    --env_interface MG_PickPlace \
    --env_interface_type robosuite

# square
python prepare_src_dataset.py \
    --dataset ../../datasets/source/square.hdf5 \
    --env_interface MG_Square \
    --env_interface_type robosuite

# stack
python prepare_src_dataset.py \
    --dataset ../../datasets/source/stack.hdf5 \
    --env_interface MG_Stack \
    --env_interface_type robosuite

# stack_three
python prepare_src_dataset.py \
    --dataset ../../datasets/source/stack_three.hdf5 \
    --env_interface MG_StackThree \
    --env_interface_type robosuite

# threading
python prepare_src_dataset.py \
    --dataset ../../datasets/source/threading.hdf5 \
    --env_interface MG_Threading \
    --env_interface_type robosuite

# three_piece_assembly
python prepare_src_dataset.py \
    --dataset ../../datasets/source/three_piece_assembly.hdf5 \
    --env_interface MG_ThreePieceAssembly \
    --env_interface_type robosuite
