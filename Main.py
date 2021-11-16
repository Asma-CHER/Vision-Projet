import cv2
import numpy as np
import matplotlib.pyplot as plt
import Fonctions

img = cv2.imread('image.jpg',cv2.IMREAD_GRAYSCALE)

if img is None:
    print('image vide')
else :


    message = "hello usthb"
    Fonctions.create_image(message, 'image.jpg', 'secret-image.png')
    imgsec = cv2.imread('secret-image.png', cv2.IMREAD_GRAYSCALE)
    msgBinary = Fonctions.dec_bin(imgsec) #je lui donne l'image recu elle me donne le msg en binaire avec decodage (inversement de la valeur)
    msgText = Fonctions.dec_ASCII(msgBinary)
    print(" end ",msgText)
message = "hell"
imgsec = Fonctions.code_image(message, 'image.jpg','imgsec.jpg')
msgBinary = Fonctions.dec_bin(imgsec) #je lui donne l'image recu elle me donne le msg en binaire avec decodage (inversement de la valeur)
msgText = Fonctions.dec_ASCII(msgBinary)
print(" end ",msgText)

    img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)


    cv2.imshow(' img before ', img)
    cv2.imshow(' img code ', imgsec)




