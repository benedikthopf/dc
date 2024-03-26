#!/bin/bash

#SBATCH --gres=gpu::1
#SBATCH --time=72:00:00
#SBATCH --output=dc-%j.out


singularity run --nv -B $(pwd),$(pwd)/.. dc.simg python train.py
