import math

import Functions_Etape1 as f1
import numpy as np
from numpy.linalg import pinv
import pickle

'''
Créer une fonction calcul_needle_map qui calcule le champ de normales (le vecteur normal pour chaque pixel).
'''
def calcul_needle_map():
    #Charger les N images dans une variable obj_images(à l'aide de la fonction: load_images)
    obj_images = f1.load_images()
    #obj_images = read_file("normal.pkl")

    #charger les positions des sources lumineuses dans une variable light_sources(à l'aide de la fonction: load_lightSources
    lights = f1.load_lightSources()

    #charger le masque de l’objet dans une variable obj_masques(à l’aide de la fonction: load_objMask)
    obj_masques = f1.load_objMask()

    #Calculer les normales de l'objet(une matrice de h lignes, w colonnes et trois composantes x, y, z
    normal =  Normale(obj_images,lights,obj_masques)

    return showImage(obj_masques,normal)





def save_file(x):
    with open("normal.pkl", 'wb') as f:
        p = pickle.Pickler(f)
        p.dump(x)

def read_file(fname):
    with open(fname,'rb') as f:
        x = pickle.load(f)
    return x

def Normale(obj_images,lights,obj_masques):


    #image = f1.load_images()

    #save_file(image)

    #print(" MIN ",min(set(image.flatten())))

    #image = read_file("normal.pkl")

    #mask = f1.load_objMask()

    #a = f1.load_lightSources()

    sInv = pinv(lights)


    #print("here shape image ", obj_images.shape, " here shape sInv ", sInv.shape)
    #print("here sInv \n ", sInv)


    mult = np.dot(sInv, obj_images)
    #print("here shape mult ", mult.shape)
    #print("here mult ", mult)


    h,w = obj_masques.shape
    normal = np.zeros((h, w, 3))

    for y in range(h):
        for x in range(w):
            if obj_masques[y,x] == 1 :
                # on calcule la norme (longeur)
                norme = math.sqrt(pow(mult[0,y*612+x],2)+pow(mult[1,y*612+x],2)+pow(mult[2,y*612+x],2))

                # on normalise les donnees
                normal[y,x][0] = mult[0,y*612+x]/norme    #Nx normalisee
                normal[y,x][1] = mult[1,y*612+x]/norme    #Ny normalisee
                normal[y,x][2] = mult[2,y*612+x]/norme    #Nz normalisee
            else:
                normal[y, x] =0

    return normal

def showImage(mask, normal):

    h, w, c = normal.shape
    for y in range(h):
        for x in range(w):
            if mask[y,x] == 1 :
                normal[y,x] = (normal[y,x] + 1) / 2 * 255

    normal = normal.astype(np.uint8())
    return normal
