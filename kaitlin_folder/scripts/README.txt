Kmeans files

batch_kmeans.py  --> PREPROCESSING : takes original FLAIR file and creates kmeans segmented files
batch_kmeans_classify.py --> CREATE YHAT --> make a ".seg "-like files for each MRI input  that gives our tumor prediction
classify.py --> DETERMINE PERFORMANCE --> evaluates closeness between real and predicted .seg files
