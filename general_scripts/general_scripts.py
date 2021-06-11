#imports 
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


# functions
def evaluate_segmentation(seg_hat_flat, seg_flat, tumor_label):
    #coordinates of predicted and actual tumor
    y_coors = np.argwhere(seg_flat!=0) #mask 

    yhat_coors = np.argwhere(seg_hat_flat ==tumor_label) #prediction 

    #size of tumor and non-tumor (number of voxels)
    tumor_size=len(y_coors)
    total_size = len(seg_flat)
    non_tumor_size = total_size-tumor_size


    #true positive
    correct_id_tumor = np.intersect1d(y_coors, yhat_coors)

    
    
    percent_tumor_correct = len(correct_id_tumor)/tumor_size

    #false positive
    false_positive_points = np.setdiff1d(yhat_coors, y_coors)
    false_positive = len(false_positive_points)/non_tumor_size

    #false negative
    false_neg_points = np.setdiff1d(y_coors, yhat_coors)
    false_neg = len(false_neg_points)/tumor_size

    
    #true negative
    true_negative=(total_size-(tumor_size+len(false_positive_points)))/non_tumor_size

    #Total error (misidentification
    total_incorrect_points = np.concatenate((false_neg_points, false_positive_points))
    total_error = len(total_incorrect_points)/total_size
    
    #percentage of yhat that is correct
    yhat_size = len(yhat_coors)
    correct_id_tumor_hat = np.intersect1d(yhat_coors, y_coors)
    specificity = len(correct_id_tumor_hat)/yhat_size
    
    #New metric, TP + specificity
    metric = (percent_tumor_correct + specificity)/2
    
    
    #Jaccard Loss - intersection over union
    intersect =np.intersect1d(yhat_coors, y_coors)
    union = np.union1d(yhat_coors, y_coors)
    jaccard_loss = len(intersect)/len(union)
    
    #Dice loss    
    dice_loss = 2*len(intersect) / (tumor_size + yhat_size)

    
    
    
    
    
    print("True Positive: ", percent_tumor_correct)
    print("False negative: ", false_neg)
    print("True Negative: ", true_negative)
    print("False Positive: ", false_positive)
    print("'specificity':", specificity)
    print("Combined TP and Specificity ", metric)
    print("Dice loss is ", dice_loss)
    print("Jaccard loss is ", jaccard_loss)
    print("************")
    print("Total Error: ", total_error)


def testfun(a):
	return a+1
