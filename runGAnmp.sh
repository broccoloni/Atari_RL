#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --account=def-aali
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
source ~/lgraha/bin/activate
cd ~/scratch/CPSC532J/A2
python Atari_genetic.py 
