#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=dc-%j.out

# cd $WORK2/extclip

singularity run --nv -B $(pwd),$(pwd)/.. dc.simg python train.py --comment="dogs" -p ../datasets/few-shot-images/moongate/img -t jpg