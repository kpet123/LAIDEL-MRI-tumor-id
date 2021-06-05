import matplotlib.pyplot as plt
import numpy as np
from dipy.io.image import load_nifti
import pandas as pd
import sklearn.cluster
import pathlib
import sys
import os
from multiprocessing import Pool


#Path to directory containing Brats dataset
path_to_lib = "/N/project/laidel_el_mcv/LAIDEL-MRI-tumor-id/lib/MICCAI_BraTS2020_TrainingData" 

#Path to results folder to store tumor and nontumor images
path_to_results_folder ="/N/project/laidel_el_mcv/LAIDEL-MRI-tumor-id/seg_images/"



def loop_body(folder_path):

    if folder_path.is_file()==False:
        print(str(folder_path))
        data_id = folder_path.parts[len(folder_path.parts)-1]



        #load data
        data_flair, affine, im = load_nifti(str(folder_path)+"/"+data_id+"_flair.nii", return_img=True)
        data_seg, affine, im = load_nifti(str(folder_path)+"/"+data_id+"_seg.nii", return_img=True)

        #set non-tumor tissue to zero
        segmented_brain = data_flair.copy()
        segmented_brain[data_seg !=0]=0

        #set tumor tissue to 0
        segmented_tumor=data_flair.copy()
        segmented_tumor[data_seg ==0]=0

        #write data output
        np.save(path_to_results_folder+data_id+"_tumor.npy", segmented_tumor, allow_pickle=True, fix_imports=True)
        np.save(path_to_results_folder+data_id+"_clean.npy", segmented_brain, allow_pickle=True, fix_imports=True)
 

#Iterate through each MRI folder, segment image and report performance

#mutlithreading - make sure 'processes' <= number of total processes in SLURM
with Pool(processes=16) as pool:
        pool.map(loop_body, pathlib.Path(path_to_lib).iterdir())           

#too slow when writing
#[loop_body(x) for x in pathlib.Path(path_to_lib).iterdir()]

