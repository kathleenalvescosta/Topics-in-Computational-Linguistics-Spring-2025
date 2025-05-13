#!/bin/bash

#ocelote
 
#SBATCH --job-name=virino
#SBATCH --output=%x-%j.out
#SBATCH --account=rhenderson
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=14
#SBATCH --gres=gpu:1
#SBATCH --time=50:00:00
 
echo "starting"
module load python/3.8/3.8.2
python3.8 eodeepspeech.py
echo "finished"

