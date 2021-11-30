import cv2
import numpy as np
import Functions_Etape1 as f1



# Etape 1. Préparation de données
print(" LOAD LIGHT_SOURCES  ************************************************************************")
lights = f1.load_lightSources()
print(" HERE SHAPE OF LIGHTS ",lights.shape)

print(" LOAD INTENS_SOURCES   ************************************************************************")
intenses = f1.load_intensSources()
print(" HERE SHAPE OF intenses",intenses.shape)

print(" LOAD MASK   ************************************************************************")
mask = f1.load_objMask()
print(" HERE SHAPE OF mask",mask.shape)

print(" LOAD IMAGES   ************************************************************************")
#image = f1.load_images(lights)
#f1.load_N_images()
f1.load_images()



# Etape 2. Calcul des normales

