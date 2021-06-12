#!/bin/bash

#SBATCH -J brats2020_data
#SBATCH -p debug
#SBATCH -o /N/project/laidel_el_mcv/LAIDEL-MRI-tumor-id/slurm_output/_%j.txt
#SBATCH -e /N/project/laidel_el_mcv/LAIDEL-MRI-tumor-id/slurm_output/_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kpet@iu.edu
#SBATCH --nodes=1
#SBATCH --mem=60gb
#SBATCH --ntasks-per-node=16
#SBATCH --time=01:00:00

#NOTE: THIS SCRIPT IS FAST. For stuff like this keep in debug queue and gets on queue faster
# to run: sbatch metrics_name.slurm
# to check status: squeue -u USERNAME
# to cancel : scancel JOBID

module load anaconda
module load dipy
python3 /N/project/laidel_el_mcv/LAIDEL-MRI-tumor-id/general_scripts/determine_metrics.py
