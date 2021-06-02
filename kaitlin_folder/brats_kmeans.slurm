#!/bin/bash

#SBATCH -J brats2020_data
#SBATCH -p debug
#SBATCH -o brats2020_data_%j.txt
#SBATCH -e brats2020_data_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kpet@iu.edu
#SBATCH --nodes=1
#SBATCH --mem=60gb
#SBATCH --ntasks-per-node=8
#SBATCH --time=01:00:00


cd /N/project/laidel_el_mcv/LAIDEL-MRI-tumor-id/scripts
module load anaconda
module load dipy
python3 kmeans_batch.py

