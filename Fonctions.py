import os
import random

import cv2
from typing import Iterable
import numpy as np
from PIL import Image
from textwrap import wrap

''' Fonctions '''


def dec_bin(table):
    print("heere2", table[0, 7])

    msgBinaire = ""
    h, w = table.shape
    #print(h," ",w)
    for y in range(h):
        for x in range(w):
            t = (int)(bin(table[y, x])[2:])

            #print(t)
            val = t % 10
            #print(val)
            if (val == 0):
                msgBinaire += str(1)
            else:
                msgBinaire += str(0)

    #print(msgBinaire)
    return msgBinaire


# fonction de decodage du binaire vers du text (code ascii)
def dec_ASCII(msgBinary):
    msgText = ""
    k = wrap(msgBinary, 8)
    #print(k)

    for b in k:
        #print(int(b, 2))
        n = int(b, 2)
        if n<=172 :
            msgText += chr(n)
        else:
            break

    return msgText


#def to_bit_generator(msg):
#    for c in (msg):
#        o = ord(c)  #unicode
#        for i in range(8):
#            yield (o & (1 << i)) >> i
def to_bit_generator(msg):
    result = []
    for c in msg:
        bits = bin(ord(c))[2:]
        bits = '00000000'[len(bits):] + bits
        #print(bits)
        result.extend([int(b) for b in bits])
        #print(result)
    return result


def code_image (message, img, output :str) :
    hidden_message = to_bit_generator(message)
    #print(hidden_message)


    image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape

    count =0
    for y in range (h):
        for x in range (w):
            r = (bin (image [y,x] ))[2:]
            list = wrap(r, 1)
            #print (k)
            #print(r)
            if count < len(hidden_message):
                k = hidden_message[count]
                count+=1
            else:
                break
            #print(k)
            if int(k) == 0:
                #print("k=0",k)
                list [-1] = '1'
            elif int(k) == 1:
                #print("k=1",k)
                list[-1] = '0'
            m = ''.join(list)
            #print(m)


            l = int(m,2)
            #print(l) #decimale

            image[y,x] = l
            #print(image[y,x])
            #count =+1


    # Write out the image with hidden message
    cv2.imwrite(output, image)



    return image
