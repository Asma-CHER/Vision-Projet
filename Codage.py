import cv2
from textwrap import wrap

def to_bit_generator(msg):
    result = []
    for c in msg:
        bits = bin(ord(c))[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result


def code(message, image) :
    hidden_message = to_bit_generator(message) #transformation du message en binaire code
    h, w = image.shape
    count =0
    for y in range (h):
        for x in range (w):
            r = (bin (image [y,x] ))[2:]
            list = wrap(r, 1)
            if count < len(hidden_message):
                k = hidden_message[count]
                count+=1
            else:
                break
            if int(k) == 0:
                list [-1] = '1'
            elif int(k) == 1:
                list[-1] = '0'
            m = ''.join(list)
            l = int(m,2)
            image[y,x] = l

    cv2.imwrite('imgsec.png', image)

    return image


image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
#message = "Un texte est une série orale ou écrite de mots perçus comme constituant un ensemble cohérent, porteur de sens et utilisant les structures propres à une langue (conjugaisons, construction et association des phrases). ... L'étude formelle des textes s'appuie sur la linguistique, qui est l'approche scientifique du langage."

message =input('Donnez le message à coder\n')
if image is None:
    print('image vide')
else:
    imageCode = code(message,image)

    cv2.imshow('image', image)
    cv2.imshow('image codee', imageCode)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
