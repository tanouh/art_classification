## Data split 
J'ai divisé les images abstraites et figuratives aléatoirement selon la répartition suivante : 
80% pour l'entrainement
10% pour la validation 
10% pour le test

Les données ne sont pas stockées sur le git mais sont copiés sur le cluster avec scp et synchroniser avec rsync.

``scp -r data tmahandry-23@gpu-gw:/home/ids/tmahandry-23/data``
``rsync -avz data tmahandry-23@gpu-gw:/home/ids/tmahandry-23/data`` (à faire  à chaque update)


conda env export > enironment.yml
conda env update --file environment.yml  --prune

conda create -n projenv python=3.10 -y
conda install pytorch torchvision torchaudio -c pytorch -c nvidia
conda install -c conda-forge tensorboard
conda install matplotlib scikit-learn


Etapes : 
1. Classification abstrait/figuratif
2. Extraction des principales caractéristiques des images abstraites
3. Clustering non supervisé des images abs pour faire apparaître les groupes naturels

Traitement des données: 
- Transformation :
- - Letterboxing ie ajout de padding pour avoir une dimension proportionnelle à 224*224 : miroir , puis redimensionnement à 224\*224
  - Data augmentation :
  -   - Rotation verticale/horizontale aléatoire de 90° (p=0.5)
  - Normalisation selon les données de ImageNet
 
Architecture : 
VGG16 - classifier = Sequential()





