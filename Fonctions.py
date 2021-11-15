import cv2
from textwrap import wrap

''' Fonctions '''
def dec_bin(table):
    msgBinaire = ""
    h, w = table.shape
    for y in range(h):
        for x in range(w):
            t= (int)(bin(table[y,x])[2:])
            #print(t)
            val = t % 10
            #print(val)
            if(val==0):
                msgBinaire +=str(1)
            else:
                msgBinaire +=str(0)

    #print(msgBinaire)
    return msgBinaire



#fonction de decodage du binaire vers du text (code ascii)
def dec_ASCII(msgBinary):
    msgText = ""
    k = wrap(msgBinary, 8)

    for b in k:
        msgText += chr(int(b, 2))

    return msgText