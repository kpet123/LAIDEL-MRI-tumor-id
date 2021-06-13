#!/bin/bash

#SBATCH -J brats2020_data
#SBATCH -p general
#SBATCH -o /N/project/laidel_el_mcv/LAIDEL-MRI-tumor-id/slurm_output/_%j.txt
#SBATCH -e /N/project/laidel_el_mcv/LAIDEL-MRI-tumor-id/slurm_output/_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kpet@iu.edu
#SBATCH --nodes=1
#SBATCH --mem=60gb
#SBATCH --ntasks-per-node=24
#SBATCH --time=02:00:00

#NOTE: THIS SCRIPT IS FAST. For stuff like this keep in debug queue and gets on queue faster
module load anaconda
module load dipy
python3 /N/project/laidel_el_mcv/LAIDEL-MRI-tumor-id/kaitlin_folder/scripts/batch_slic.py
