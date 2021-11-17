import cv2
import numpy as np
import matplotlib.pyplot as plt
import Fonctions

'''
img = cv2.imread('image.png',cv2.IMREAD_GRAYSCALE)

if img is None:
    print('image vide')
else :


    message = "hello usthb"
    Fonctions.create_image(message, 'image.png', 'secret-image.png')
    imgsec = cv2.imread('secret-image.png', cv2.IMREAD_GRAYSCALE)
    msgBinary = Fonctions.dec_bin(imgsec) #je lui donne l'image recu elle me donne le msg en binaire avec decodage (inversement de la valeur)
    msgText = Fonctions.dec_ASCII(msgBinary)
    print(" end ",msgText)
    
    
    cv2.imshow(' img before ', img)
    cv2.imshow(' img code ', imgsec)
'''

message = "hello usthb racha asma vision artificielle blaaa blaaaaaaaa before after script de programme java python bla blaa         mmmm plus moins rcr2 tp1 tp2 projet data mining pour jeudi"
imgsec = Fonctions.code_image(message, 'image.png','imgsec.png')

i = cv2.imread('imgsec.png', cv2.IMREAD_GRAYSCALE)

print("heere1",i[0,7])

msgBinary = Fonctions.dec_bin(imgsec) #je lui donne l'image recu elle me donne le msg en binaire avec decodage (inversement de la valeur)
msgText = Fonctions.dec_ASCII(msgBinary)
print(" end ",msgText)





