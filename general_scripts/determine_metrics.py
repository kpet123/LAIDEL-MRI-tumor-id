#local library functions
import general_functions

#python library functions
#import matplotlib.pyplot as plt
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
path_to_yhat_folder ="/N/project/laidel_el_mcv/LAIDEL-MRI-tumor-id/results/PATH_TO_CLASSIFIED_IMAGES" +"/"

#Path to results folder to store metatdata. TODO: CHANGE TH
output_path ="/N/project/laidel_el_mcv/LAIDEL-MRI-tumor-id/results/metrics_csvs/YOUR_CSV_NAME_HERE"

#how did classifier do on each image? Holds image name and metrics
performance_eval_dct={}



#Evaluating metrics on a single MRI image. 
#Looped later via list comprehension (this is faster than for loop since we are handling so much data)

def loop_body(lib_folder_path):

  #folder_path of brats multimodal data - we are looking for .seg
  if lib_folder_path.is_file() == False:

    #extract ID of data file, used for naming
    data_id = lib_folder_path.parts[len(lib_folder_path.parts)-1]
    #print(data_id, file=sys.stderr)

    print("processing ", data_id)            
    #load seg file (mask)  and create derivatives
    data_seg_path = str(lib_folder_path)+"/"+str(data_id)+"_seg.nii"
    data_seg, affine, im = load_nifti(data_seg_path, return_img=True)
    data_seg_flattened = data_seg.flatten()

    #load classifier result file TODO MAKE SURE THIS MATCHES WHATEVER NAMING CONVENTION YOU USE
    yhat_path = path_to_yhat_folder+str(data_id)+"_classified.npy"
    data_yhat = np.load(yhat_path)
    print("values in classified image:", np.unique(data_yhat))
    data_shape=data_yhat.shape
    data_yhat_flattened = data_yhat.flatten()

    #find label of tumor - all tumor should be 1 but confirm
    tumor_label = np.max(data_yhat_flattened)
    #print("tumor label is ", tumor_label)            
      #calculate confusion matrix-MODIFY THIS SECTION FOR DIFFERENT LOSS FUCNTIONS
    performance_metrics = general_functions.evaluate_segmentation(data_yhat_flattened, 
							data_seg_flattened, 
							tumor_label)
    #performance_eval_dct[data_id]=performance_metrics
    #print(performance_metrics)    
    return {data_id: performance_metrics}

#Iterate through each MRI folder, segment image and report performance
result = [loop_body(x) for x in pathlib.Path(path_to_lib).iterdir()]

#make sure isn't blank, this will print to slurm .err file
#print(result, file=sys.stderr)
print("results are", result)
dct = {}
for smalldct in result:
    #print("smalldct is ", smalldct)
    if( smalldct is None): 
        print("nonetype result? look back into record")
    else:
        dct.update(smalldct)
#Save metadata as pandas dataframe, then covert to csv

output = pd.DataFrame(dct, index = ["True Positive", 
									"False negative", 
									"True Negative", 
                                    "False positive", 
									"Specificity", 
									"Combined TP and Specificity",
                                    "Dice Loss", 
									"Jaccard Loss", 
									"Total Error"]).transpose()
output.to_csv(output_path+".csv")
