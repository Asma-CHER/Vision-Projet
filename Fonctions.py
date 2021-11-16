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


def bits_provider(message) -> Iterable[int]:
    for char in message:
        ascii_value = ord(char)
        for bit_position in range(8):
            power = 7 - bit_position
            yield 1 if ascii_value & (1 << power) else 0

def create_image(message: str, input, output: str) -> None:
    img = Image.open(input)
    pixels = np.array(img)
    img.close()
    clear_low_order_bits(pixels)
    for i, bit in enumerate(bits_provider(message)):
        row = i // pixels.shape[1]
        col = i % pixels.shape[1]
        pixels[row, col,0] |= bit
    out_img = Image.fromarray(pixels)
    out_img.save(output)
    out_img.close()


def clear_low_order_bits(pixels) -> None:
    for row in range(pixels.shape[0]):
        for col in range(pixels.shape[1]):
            pixels[row, col,0] &= ~1