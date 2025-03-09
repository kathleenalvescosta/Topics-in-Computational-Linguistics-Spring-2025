#!/bin/bash

###OCELOTE


#SBATCH --job-name=fastpt
#SBATCH --output=%x-%j.out
#SBATCH --account=rhenderson
#SBATCH --partition=gpu_standard
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
 
IMAGE=/groups/hammond/mikehome/u6/kathleencosta/exercise/mh2.sif

echo "starting"
SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 singularity exec \
	--nv $IMAGE python3 /home/u6/kathleencosta/TTS/fastpt.py
echo "finished"

