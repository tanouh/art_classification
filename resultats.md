# Entraînement sur modèle VGG16
| JOB | Epoch | Starting lr | Criterion loss | Optimizer | Accuracy  (validation data) | Loss (validation data) | Accuracy (test data) | Loss (test data) |
|---|-------|----|----------------|-----------|-----------------------------|------------------------|----------------------|------------------|
| | 1 | 0.0005 | CrossEntropyLoss | Adam | 92.93% | 0.2256 | 0 | 0 |
| | 20 EarlyStop at 5th | 0.001 | CrossEntropyLoss | Adam | 93.32% | 0.1639 | 93.20%  | 0.1522 |
| | 5 | 0.001 | CrossEntropyLoss | Adam | 93.32% | 0.1830 | 92.94% | 0.1558 | 
| 325884 | 5 | 0.001 | CrossEntropyLoss | SGD | 94.24% | 0.1218 not the best train loss | 93.99% | 0.1504  |
| 325984 | 10 | 0.0005 | CrossEntropyLoss | SGD | 93.46% | 0.1197 best train loss on last epoch with same accuracy | 93.99%  | 0.1428 | 
| 325899 | 10 | 0.0005 | CrossEntropyLoss | Adam | 93.59% | 0.1387 | 93.73% | 0.1482 |


On a plutôt intérêt à choisir le modèle fine-tuner avec SGD, epoch=10, lr=0.0005

# Embeddings 
## Extraction task
326017 : Extracting embeddings from abstract images with vgg16_optiSGD
- 4096 features extracted from 1241 images 
## Visualisation 
1. Try to explicit the clusters 
- Test on several parameters 




