import matplotlib.pyplot as plt
import numpy as np
from dipy.io.image import load_nifti
import pandas as pd
import sklearn.cluster
import skimage.segmentation
#from fast_slic import Slic
from skimage.segmentation import mark_boundaries
from skimage.color import gray2rgb
from skimage.filters import threshold_otsu
from skimage.filters import median
from skimage.color import rgb2gray
from scipy.stats import ttest_ind, norm
import pathlib
import sys
import os
from multiprocessing import Pool

'''
Path to directory containing Brats data (probably LGG or HGG)
'''

#path_to_lib = "../../lib/MICCAI_BraTS2020_TrainingData"
path_to_lib ="/N/project/laidel_el_mcv/LAIDEL-MRI-tumor-id/seg_images"
#first command line argument
#path_to_lib = sys.argv[1]

'''
path to results
'''
#second command line argument
#path_to_results=sys.argv[2]
path_to_results_folder ="/N/project/laidel_el_mcv/LAIDEL-MRI-tumor-id/results/control_slic_classified/"


'''
Body of list comprehension - hopefully
'''
def loop_body(path):

       if path.is_file() and ("clean" in str(path)):
            #extract ID of data file, used for naming
            data_id = path.parts[len(path.parts)-1]
            data_id =data_id[0:-4]
            
            print(data_id)
            data_flair = np.load(path)
            #NORMALIZE
            minval = np.min(data_flair)
            maxval = np.max(data_flair)
            flair_norm = (data_flair-minval)/(maxval-minval)
            #median filter (noise reduction)- UNCOMMENT IF VERSION OF SKIMAGE SUPPORTS 3D FILTER
            #flair_norm  = median(flair_norm )
            #convert to rGB color space 
            #flair_rgb = gray2rgb(flair_norm )
            #tolerance - bottom x percent viewed as background
            tolerance = 0.1
            #empty array for storing segmented image
            output = np.zeros(data_flair.shape)
            #number of superpixels per image
            n_segments_2d = 300
 
            #iterate through top-down slices (z direction)
            for i in range(0, data_flair.shape[2]):
                slice_2d = flair_norm[:,:,i]

                #do median filter here if skimage version doesn't allow 3d
                slice_2d=median(slice_2d)
                #also need to move rgb conversion here
                slice_2d=gray2rgb(slice_2d)
                maxval = np.max(slice_2d)
                # create mask - same number label if part of same cluster
                slic_mask_2d = skimage.segmentation.slic(slice_2d, n_segments=n_segments_2d, convert2lab=True)
                #convert back to grayscale
                slice_2d = rgb2gray(slice_2d)
               
                #set of unique labels (should be length n_segments_2d)
                labels2D = np.unique(slic_mask_2d)

                #replace each superpixel cluster with average value of cluster
                for label in labels2D:
                    slice_2d[slic_mask_2d ==label]=np.average(slice_2d[slic_mask_2d ==label])
                #assign mean clustered slice to output
                output[:,:,i]=slice_2d


            # take out bottom section so background will not affect otsu thresholding
            cluster_3d_masked = output[output > tolerance]
            #perfrom otsu thresholding
            thresh_3d = threshold_otsu(cluster_3d_masked.flatten())
            #zero out anything not tumor hat
            output[output< thresh_3d]=0
            #assign all tumor superpixels to value of 1
            tumor_otsu_label = 1
            output[output>0]=tumor_otsu_label                      
            #write data output
            np.save(path_to_results_folder+data_id+".npy", output, allow_pickle=True, fix_imports=True)
            #mean_var_list.append(calc_mean_var(flair_2D, flair_kmeans, clusters))
            #data_order_ref.append(data_id)


'''
Iterate through each MRI folder, extract FLAIR image, and convert to kmeans
'''

#mutlithreading
with Pool(processes=12) as pool:
        pool.map(loop_body, pathlib.Path(path_to_lib).iterdir())           



