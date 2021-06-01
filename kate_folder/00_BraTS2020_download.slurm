#!/bin/bash

#SBATCH -J brats2020_data
#SBATCH -p general
#SBATCH -o brats2020_data_%j.txt
#SBATCH -e brats2020_data_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kmorten@iu.edu
#SBATCH --nodes=1
#SBATCH --mem=60gb
#SBATCH --ntasks-per-node=6
#SBATCH --time=05:00:00


cd /N/project/gorillaPAR/LAIDEL/

curl --header 'Host: storage.googleapis.com' --user-agent 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:88.0) Gecko/20100101 Firefox/88.0' --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' --header 'Accept-Language: en-US,en;q=0.5' --referer 'https://www.kaggle.com/' --header 'Upgrade-Insecure-Requests: 1' --header 'Sec-GPC: 1' 'https://storage.googleapis.com/kaggle-data-sets/751906/1299795/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20210526%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210526T211148Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=a2c4a55bf5acf4d29de3027c16b2242a36d94fc07fb80bfba6a2446ab27cd6ca0f44fd18e175e07f6494e9cf052eb800f59018dcfba61b2f236a54b057a0a01a107d1d2ac736e25ed998332ff3fa48a7c52871611ad5334311e6cc499ceda0149830ae39171a4e3735bb961191139426ada1e4576a46962cd71f3a41b131a150ba89e7a524409a4f294934b584fb79807b2778ca2111b8c8b008f3ed4f117880ad2a2795ad62f3e16aa7ce66482c7b3b337f388f28ec9a43002065fcac78c1b739ad16317f13d573b1d454f9650c61778bf1bb37ebf4af14abb1e0ed9da9c8e4433af6a65048e3bf04e491cde434ea49043257726fa335ac3ceb8f7575f1a2ec' --output 'archive.zip'

mv archive.zip BraTS2020.zip

unzip BraTS2020.zip


