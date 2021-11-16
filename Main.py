
import Fonctions

message = "hell"
imgsec = Fonctions.code_image(message, 'image.jpg','imgsec.jpg')
msgBinary = Fonctions.dec_bin(imgsec) #je lui donne l'image recu elle me donne le msg en binaire avec decodage (inversement de la valeur)
msgText = Fonctions.dec_ASCII(msgBinary)
print(" end ",msgText)



