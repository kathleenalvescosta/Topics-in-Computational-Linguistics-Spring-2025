#!/bin/bash
 
#SBATCH --job-name=mkwav
#SBATCH --output=%x-%j.out
#SBATCH --account=rhenderson
#SBATCH --partition=standard
#SBATCH --ntasks=10
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1gb
#SBATCH --time=01:00:00
 
echo "starting"
module load python/3.8/3.8.2
module load ffmpeg
python3 mkwav.py
echo "finished"

