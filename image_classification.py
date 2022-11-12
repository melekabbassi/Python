# Dans un script Python (.py), vous allez utiliser le modèle obtenu dans ce notebook pour décider pour une image si elle correpond au chiffre 0 , 1 ou 2
# knn.pkl est le modèle obtenu dans ce notebook
# image.jpg est l'image à tester

import pickle
import numpy as np
from PIL import Image

# Chargement du modèle
knn = pickle.load(open('knn.pkl', 'rb'))

# test sur une image de 0
image = Image.open('image.jpg')

# Conversion de l'image en tableau numpy
image = np.array(image)

# On redimensionne l'image pour qu'elle corresponde à la taille des images d'entrainement
image = image.reshape(1, -1)

# On prédit la classe de l'image
prediction = knn.predict(image)

# On affiche la prédiction
print(prediction)

# On affiche la probabilité de la prédiction
print(knn.predict_proba(image))