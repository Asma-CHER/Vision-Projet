import cv2
from textwrap import wrap

def dec_bin(table): # elle retourne le msg en binaire de l'image
    msgBinaire = ""
    h, w = table.shape
    for y in range(h):
        for x in range(w):
            t = (int)(bin(table[y, x])[2:])
            val = t % 10
            if (val == 0):
                msgBinaire += str(1)
            else:
                msgBinaire += str(0)
    return msgBinaire

# fonction de decodage du binaire vers du text (code ascii)
def dec_ASCII(msgBinary):
    msgText = ""
    k = wrap(msgBinary, 8)
    for b in k:
        n = int(b, 2)
        #msgText += chr(n)
        if (n<=127) or (n>=192 and n<=255) :
            msgText += chr(n)
        else:
            break
    return msgText


def decode(image):
    return dec_ASCII(dec_bin(image))



img = cv2.imread('imgsec.png', cv2.IMREAD_GRAYSCALE)
if img is None:
    print('image vide')
else:

    message = decode(img)
    print(message)
    cv2.imshow('image codee', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()