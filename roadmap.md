# Classification raisonnée d'images abstraites
## Etape 1 : Classification abstrait VS figuratif
**Objectif :** Rendre mon modèle robuste sur la classification abstrait vs figuratif

### Données 
Jeu de données d'images abstraites/figuratives, séparé en trois (80/10/10) : train, validation, test. 

### Entrainement
_Actions :_
1. Prétraitement des données
- Ajout des transformation afin d'adapter au modèle de VGG16 notamment :
  - Redimensionnement en format 224x224 
  - Normalisation aux données d'entrainement d'origine
2. Configuration du modèle
- Gel des couches convolutionnelles 
  - Motivation : réduction du coût de calcul, évite le suraprentissage,volume de données pas trop importante
- Modification du classifieur final pour une classification binaire
3. Entraînement
- Critère de perte : **crossEntropyLoss**, adapté aux tâches de classification multi-classe (ici 2)
- Optimiseur : **Adam**/**SGD**

Phase d'entrainement suivi d'une phase de validation et sauvegarde du meilleur modèle suivant la meilleure précision à chaque epoch. 

### Test
Phase de test sur les données non rencontrées en entrainement. 

## Etape 2 : Apprendre les features visuelles
**Objectif :**  Extraire les features visuelles des images abstraites seulement

_Actions :_
- Extraction des embeddings du modèle (utilisation des activations avant la dernière couche de classification)
- Visualisation des embeddings (avec UMAP et t-SNE)
  - Constat sur la formation naturelle de regroupements 

## Etape 3 : Classification raisonnée par caractéristiques
**Objectif :** Classification des images abstraites par *familles visuelles*.

_Actions :_
1. Création d'un dataset enrichi : WikiArt
- Pour chaque image, ajouter un ou plusieurs labels supplémentaires : 
  - Texture : lisse, granuleuse, rugueuse, etc...
  - Forme dominente : courbe, angulaire, fractale, etc...
  - Palette de couleurs : froide, chaude, contrastée, monochrome, etc...
2. Modification du modèle 
- Passer à un modèle multi-label
- Critère de perte : **nn.BCEWithLogitsLoss**