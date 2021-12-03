import math

import Functions_Etape1 as f1
import numpy as np
from numpy import array
from numpy.linalg import pinv

'''
Créer une fonction calcul_needle_map qui calcule le champ de normales (le vecteur normal pour chaque pixel).
'''
def calcul_needle_map():
    #Charger les N images dans une variable obj_images(à l'aide de la fonction: load_images)
    print(" LOAD IMAGES  ************************************************************************")
    obj_images = f1.load_images()

    #charger les positions des sources lumineuses dans une variable light_sources(à l'aide de la fonction: load_lightSources
    print(" LOAD Light_SOURCES   ************************************************************************")
    lights = f1.load_lightSources()

    #charger le masque de l’objet dans une variable obj_masques(à l’aide de la fonction: load_objMask)
    print(" LOAD MASK   ************************************************************************")
    obj_masques = f1.load_objMask()


def Normale():

    image = f1.load_images()
    a = f1.load_lightSources()
    sInv = pinv(a)
    print("here shape image ", image.shape, " here shape sInv ", sInv.shape)

    mult = np.dot(sInv, image)
    print("here shape mult ", mult.shape)
    print("here mult ", mult)


    normal = np.ndarray((512, 612,3))

    for y in range(512):  # ligne par ligne
        for x in range(612):
            #normal[y,x][0] = mult[0,y*x]    #Nx
            #normal[y,x][1] = mult[1,y*x]    #Ny
            #normal[y,x][2] = mult[2,y*x]    #Nz

            # on calcule la norme (longeur)
            norme = math.sqrt(pow(mult[0,y*x],2)+pow(mult[1,y*x],2)+pow(mult[2,y*x],2))

            # on normalise les donnees
            normal[y,x][0] = mult[0,y*x]/norme    #Nx normalisee
            normal[y,x][1] = mult[1,y*x]/norme    #Ny normalisee
            normal[y,x][2] = mult[2,y*x]/norme    #Nznormalisee





    return normal
