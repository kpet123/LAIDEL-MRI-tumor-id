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

#Path to classified MRI image in .npy format
path_to_yhat_folder ="/N/project/laidel_el_mcv/LAIDEL-MRI-tumor-id/results/kmeans_classified/"

#Path to results folder to store metatdata. End with tag of specific data
path_to_results_folder ="/N/project/laidel_el_mcv/LAIDEL-MRI-tumor-id/results/kmeans_metrics"

#how did classifier do on each image? Holds image name and metrics
performance_eval_dict={}



#Evaluating metrics on a single MRI image. 
#Looped later via list comprehension (this is faster than for loop since we are handling so much data)

def loop_body(lib_folder_path):

  #folder_path of brats multimodal data - we are looking for .seg
  if lib_folder_path.is_file() == False:

    #extract ID of data file, used for naming
    data_id = lib_folder_path.parts[len(lib_folder_path.parts)-1]
            
    #load seg file (mask)  and create derivatives
    data_seg_path = str(lib_folder_path)+"/"+str(data_id)+"_seg.nii"
    data_seg, affine, im = load_nifti(data_seg_path, return_img=True)
    data_seg_flattened = data_seg.flatten()

    #load classifier result file 
    yhat_path = path_to_yhat_folder+str(data_id)+".npy"
    data_yhat = np.load(yhat_path)
    data_shape=data_yhat.shape
    data_yhat_flattened = data_yhat.flatten()

    #find label of tumor - all tumor should be 1 but confirm
    tumor_label = np.max(data_yhat_flattened)
                
      #calculate confusion matrix-MODIFY THIS SECTION FOR DIFFERENT LOSS FUCNTIONS
    performance_metrics = evaluate_segmentation(data_yhat_flattened, 
												data_seg_flattened, 
												tumor_label)
    performance_eval_dct[data_id]=performance_metrics


#Iterate through each MRI folder, segment image and report performance
[loop_body(x) for x in pathlib.Path(path_to_lib).iterdir()]

#make sure isn't blank, this will print to slurm .err file
print(performance_eval_dct, file=sys.stderr)

#Save metadata as pandas dataframe, then covert to csv

results = pd.DataFrame(dct, index = ["True Positive", 
									"False negative", 
									"True Negative", 
                                    "False positive", 
									"Specificity", 
									"Combined TP and Specificity",
                                    "Dice Loss", 
									"Jaccard Loss", 
									"Total Error"]).transpose()
results.to_csv(output_path+".csv")
