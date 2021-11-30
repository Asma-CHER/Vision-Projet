import math

import cv2
import numpy as np

''' 
fonction load_lightSources qui return une matrice (Nx3) qui représente les positions des sources lumineuses 
(chaque ligne représente une position x,y,z) '''
def load_lightSources():
    #vec3D = np.zeros((i, j))

    with open('datasetVision/light_directions.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()
        #print(len(lines))
        cpt = 0

        vecDirec = [];
        for line in lines:
            # print(cpt," : ",line)
            val = []
            valDouble = []
            val = line.strip().split(" ")
            for v in val:
                valDouble.append(float(v))

            # print(cpt," : ",val)
            vecDirec.append(valDouble)
            a = np.array(vecDirec)
        #print(a)

    return a

'''
Créer une fonction load_intensSources qui return une matrice (N*3) qui représente les intensités des sources lumineuses 
 Chaque ligne represente l'intensite d'un pixel R,G,B)'''
def load_intensSources():
    with open('datasetVision/light_intensities.txt', 'r', encoding='utf-8') as file2:
        lines = file2.readlines()
        #print(len(lines))
        cpt = 0

        vecIntes = [];
        for line in lines:
            # print(cpt," : ",line)
            val = []
            valDouble = []
            val = line.strip().split(" ")
            for v in val:
                valDouble.append(float(v))

            # print(cpt," : ",val)
            vecIntes.append(valDouble)
            a = np.array(vecIntes)
        #print(a)

    return a


'''
Créer une fonction load_objMask qui retourne une matrice (image) binaire tel que :
1 represente un pixel de l'objet et 0 : un pixel du fond'''
def load_objMask():
    path = 'datasetVision/mask.png'
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print('image vide')
    else:

        h, w = image.shape
        mat = []
        for y in range(h):  # ligne par ligne
            val = []
            for x in range(w):  # colonne par colonne
                if image[y, x] == 0:
                    # print(" here ",x)
                    val.append(0)
                else:
                    # print(" ********************************** ",x)
                    val.append(1)
            mat.append(val)
        mask = np.array(mat)
        #print(image.shape)
        #print(len(mat))
        #print(mat)
    return mask


'''
Créer une fonction load_images qui permet de charger les N images (les images sont sauvegardées sur 16bits non signés).
On doit utiliser le fichier filenames.txt'''
def load_N_images():
    path_file = "datasetVision/filenames.txt"
    table = []
    with open(path_file, 'r', encoding='utf-8') as file:
        lines = file.read().splitlines()
        for line in lines:
            image = cv2.imread(str("datasetVision/" + str(line)), -1)  # read with uint16
            if image is None:
                print('datasetVision/' + str(line), 'vide')
            else:
                # changer l'intervalle des valuers de uint16 [0 , 216-1] à float32 [0 , 1] (ON DEVISE chaque valeur sur 2**16-1)
                #image = image.astype(np.float32)
                val_16 = math.pow(2,16)-1
                image_Float = image/val_16
                #print("*********************************",image_Float)
                table.append(image_Float)
        print(len(table))
    return table


# Diviser chaque pixel sur l'intensite de la source (B/intB, G/intG, R/intR) #chaque image sur une ligne
def div_image_intens(image, vecInt):
    h, w ,c = image.shape
    imageDivIntens = np.zeros(image.shape, np.float32)

    for y in range(h):  # ligne par ligne
        for x in range(w):
            imageDivIntens[y, x][0] = image[y, x][0] / vecInt[2]
            imageDivIntens[y, x][1] = image[y, x][1] / vecInt[1]
            imageDivIntens[y, x][2] = image[y, x][2] / vecInt[0]
    return imageDivIntens

# Convertir les images en niveau de gris (NVG = 0.3 * R + 0.59 * G + 0.11 * B)
def image_in_gray(image):
    h, w ,c = image.shape
    imageGray = np.zeros((h,w), np.float32)
    for y in range(h):  # ligne par ligne
        for x in range(w):
            imageGray[y,x]= image[y,x][0]*0.11+ image[y,x][1]*0.59+ image[y,x][2]*0.3
    return imageGray

def load_images():

    intens = load_intensSources() #fonction qui charge la matrice des intensites
    table = load_N_images() #fonction qui charge les N images du fichier filenames.txt
    tableDivIntens = []
    tableGray = []
    listAll =  []


    cpt=0
    for image in table:

        # Diviser chaque pixel sur l'intensite de la source (B/intB, G/intG, R/intR) #chaque image sur une ligne
        imgIntens = div_image_intens(image,intens[cpt])
        cpt += 1
        tableDivIntens.append(imgIntens)

        # Convertir les images en niveau de gris (NVG = 0.3 * R + 0.59 * G + 0.11 * B)
        imgGray = image_in_gray(imgIntens)
        tableGray.append(imgGray)

        # redimensionner l'image telle que chaque image est represnetee dans une seule ligne.
        imageReshape = imgGray.reshape(1,-1)
        #print(imageReshape.shape)

        # Ajouter les images dans un tableau (pour former une matrice de N lignes et (h*w) colonnes où chaque ligne représente une image).
        listAll.append(imageReshape)
        if cpt == 1: #si c'est la premiere image -> we put it directly
            matriceAll = imageReshape
        else: # else we concatenate the previous image vectors with the new one
            matriceAll = np.concatenate((matriceAll,imageReshape),axis=0)
    #print(len(listAll))
    #matriceAll = np.array(listAll)
    print(matriceAll.shape)
    print(matriceAll)

    # Retourner la matrice des images.
    return matriceAll

