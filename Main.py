import cv2
import numpy as np
import matplotlib.pyplot as plt
import Fonctions

img = cv2.imread('image.jpg',cv2.IMREAD_GRAYSCALE)

if img is None:
    print('image vide')
else :
    msgBinary = Fonctions.dec_bin(img) #je lui donne l'image recu elle me donne le msg en binaire avec decodage (inversement de la valeur)
    msgText = Fonctions.dec_ASCII(msgBinary)
    print(" end ",msgText)



