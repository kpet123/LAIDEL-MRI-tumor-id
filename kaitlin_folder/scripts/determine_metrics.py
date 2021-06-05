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
path_to_kmeans_folder ="/N/project/laidel_el_mcv/LAIDEL-MRI-tumor-id/results/kmeans-kaitlin/"

#Path to classified image (tumor v no tumor) 
path_to_classified_folder ="/N/project/laidel_el_mcv/LAIDEL-MRI-tumor-id/results/kmeans_classified/"

#Path to results folder to store metatdata
path_to_results_folder ="/N/project/laidel_el_mcv/LAIDEL-MRI-tumor-id/results/"



#total clusters in FLAIR image (background, WM , GM, tumor)
clusters = 4

#total size of data
flat_length=240*240*155

#holds names of MRI data identifiers in the order they were accessed
data_order_ref = []

#holds cluster means and variances
mean_var_list = []

#how did classifier do on each image?
performance_eval_lst=[]


#Function for finding mean and varieance of each cluster

def calc_mean_var(flair_2D, flair_kmeans, clusters):
    kmeans_arr=np.zeros((clusters, 2))
    for i in range(0, clusters):
        section=flair_2D[flair_kmeans==i]
        kmeans_arr[i]=np.array([np.mean(section), np.std(section)])
    return kmeans_arr


#Evaluate segmentation

def evaluate_segmentation(flair_kmeans,data_seg_flattened, tumor_label):
    #coordinates of predicted and actual tumor
    y_coors = np.argwhere(data_seg_flattened!=0)
    flair_yhat_coors = np.argwhere(flair_kmeans ==tumor_label)
    
    #size of tumor and non-tumor (number of voxels)
    tumor_size=len(y_coors)
    non_tumor_size = len(data_seg_flattened)-tumor_size

    #true positive
    correct_id_tumor = np.intersect1d(y_coors, flair_yhat_coors, assume_unique=True)
    percent_tumor_correct = len(correct_id_tumor)/tumor_size

    #false positive
    false_positive_points = np.setdiff1d(flair_yhat_coors, y_coors, assume_unique=True)
    false_positive = len(false_positive_points)/non_tumor_size

    #false negative
    false_neg_points = np.setdiff1d(y_coors, flair_yhat_coors, assume_unique=True)
    false_neg = len(false_neg_points)/tumor_size

    
    #true negative
    true_negative=(len(data_seg_flattened)-(tumor_size+len(false_positive_points)))/non_tumor_size

    #Total error (misidentification
    total_incorrect_points = np.concatenate((false_neg_points, false_positive_points))
    total_error = len(total_incorrect_points)/len(data_seg_flattened)
    
    return np.array([percent_tumor_correct, false_neg, true_negative, false_positive, total_error])


 #Operation to be parellelized. Creates segmented tumor image and records metrics

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
    mean_var_list.append(kmeans_metrics)                
    #print(kmeans_metrics, file=sys.stderr) #print to stderr so it shows up in error file   
        #find label of tumor
    tumor_label = np.argmax(kmeans_metrics[:,0])
                
      #calculate confusion matrix-MODIFY THIS SECTION FOR DIFFERENT LOSS FUCNTIONS
    performance_metrics = evaluate_segmentation(data_kmeans_flattened, data_seg_flattened, tumor_label)
    performance_eval_lst.append(performance_metrics) 

    #Keep list of visited images
    data_order_ref.append(data_id)


#Iterate through each MRI folder, segment image and report performance

#mutlithreading - make sure 'processes' <= number of total processes in SLURM
#with Pool(processes=16) as pool:
#        pool.map(loop_body, pathlib.Path(path_to_lib).iterdir())           

[loop_body(x) for x in pathlib.Path(path_to_lib).iterdir()]

print(data_order_ref, file=sys.stderr)
#Save metadata as file
dct = {"names": np.asarray(data_order_ref), "cluster_metrics": np.asarray(mean_var_list), "performance": np.asarray(performance_eval_lst)}
np.savez(path_to_results_folder+"/kmeans_aggregate.npz", **dct)

