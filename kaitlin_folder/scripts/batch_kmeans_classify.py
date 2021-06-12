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
#path_to_lib = "/N/project/laidel_el_mcv/BraTS2020/100-sample-dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/"
#path_to_lib = "/N/project/laidel_el_mcv/BraTS2020/100-sample-dataset/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/"

#Path to segmented kmeans images
path_to_kmeans_folder ="/N/project/laidel_el_mcv/LAIDEL-MRI-tumor-id/results/kmeans-preprocessed/"

#Path to classified image (tumor v no tumor) 
path_to_classified_folder ="/N/project/laidel_el_mcv/LAIDEL-MRI-tumor-id/results/kmeans_classified/"

#Path to results folder to store metatdata
path_to_results_folder ="/N/project/laidel_el_mcv/LAIDEL-MRI-tumor-id/results/"



#total clusters in FLAIR image (background, WM , GM, tumor)
clusters = 4

#total size of data
flat_length=240*240*155


#Function for finding mean and varieance of each cluster

def calc_mean_var(flair_2D, flair_kmeans, clusters):
    kmeans_arr=np.zeros((clusters, 2))
    for i in range(0, clusters):
        section=flair_2D[flair_kmeans==i]
        kmeans_arr[i]=np.array([np.mean(section), np.std(section)])
    return kmeans_arr


def loop_body(lib_folder_path):

  #folder_path of brats multimodal data - we are looking for .seg
  if lib_folder_path.is_file() == False:
    #extract ID of data file, used for naming
    data_id = lib_folder_path.parts[len(lib_folder_path.parts)-1]
            
    #load original data file and create derivatives
    data_flair_path = str(lib_folder_path)+"/"+str(data_id)+"_flair.nii"
    #print(data_flair_path)
    data_flair, affine, im = load_nifti(data_flair_path, return_img=True)
    data_flair_flattened = data_flair.flatten()

   #load seg file and create derivatives
    data_seg_path = str(lib_folder_path)+"/"+str(data_id)+"_seg.nii"
    data_seg, affine, im = load_nifti(data_seg_path, return_img=True)
    data_seg_flattened = data_seg.flatten()

        #load corresponding kmeans file and create derivatives  
    kmeans_path = path_to_kmeans_folder+str(data_id)+".npy"
    data_kmeans = np.load(kmeans_path)
    data_shape=data_kmeans.shape
    data_kmeans_flattened = data_kmeans.flatten()

        #Get kmeans metrics (mean and variance of each blob)
    kmeans_metrics = calc_mean_var(data_flair_flattened, data_kmeans_flattened, clusters)
   #print(kmeans_metrics, file=sys.stderr) #print to stderr so it shows up in error file   
               
    #extract putative tumor from original image and save - MODIFY THIS SECTION FOR POSTPROCESSING NEEDS
    #print("unique labels are", np.unique(data_kmeans_flattened))
    tumor_label = np.argmax(kmeans_metrics[:,0])
    print("tumor label for ", data_id, "is ", tumor_label)
    #maybe use skimage.segmentation.mark_boundary
    data_kmeans[data_kmeans != tumor_label]=0     
    #write segmented image to file
    np.save(path_to_classified_folder+data_id+"_classified.npy", data_kmeans, allow_pickle=True, fix_imports=True)
  

#Iterate through each MRI folder, segment image and report performance

#mutlithreading - make sure 'processes' <= number of total processes in SLURM
with Pool(processes=16) as pool:
        pool.map(loop_body, pathlib.Path(path_to_lib).iterdir())           

#too slow when writing
#[loop_body(x) for x in pathlib.Path(path_to_lib).iterdir()]

