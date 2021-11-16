import os
import random

import cv2
from typing import Iterable
import numpy as np
from PIL import Image
from textwrap import wrap

''' Fonctions '''


def dec_bin(table):
    msgBinaire = ""
    h, w = table.shape
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

    # print(msgBinaire)
    return msgBinaire


# fonction de decodage du binaire vers du text (code ascii)
def dec_ASCII(msgBinary):
    msgText = ""
    k = wrap(msgBinary, 8)

    for b in k:
        msgText += chr(int(b, 2))

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
        result.extend([int(b) for b in bits])
    return result


def code_image (message, img, output :str) :
    hidden_message = to_bit_generator(message)
    #print (hidden_message)
    # Read the original image
    image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape
    #length = len (message) *8
    #key = []
    count =0
    for y in range (h):
        for x in range (w):
            r = (bin (image [y,x] ))[2:]
            list = wrap(r, 1)
                #print (k)
                #print(r)
            k = hidden_message[count]
            if k == '0':
                list [-1] = '1'
            else:
                list[-1] = '0'
            m = ''.join(list)
            l = int(m,2)  #decimale
            image[y,x] = l
            count =+1
        #position = (-1,-1)
        #while position not in key:
        #    position = (random.randint(0, h), random.randint(0,w))
        #key.append (position)

    # Write out the image with hidden message
    cv2.imwrite(output, image)
    return image
