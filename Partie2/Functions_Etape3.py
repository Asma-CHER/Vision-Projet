import math

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

import Functions_Etape1 as f1
import numpy as np




def z(image):
    imageG = f1.image_in_gray(image)
    imageG = imageG/255
    '''h,w,c = imageG
    X = np.zeros(w)
    Y = np.zeros(h)
    Z = np.zeros((h,1))
    '''
    '''for y in range(h):
        for x in range(w):


    X, Y = np.mgrid[-10:10:100j, -10:10:100j]
    Z = (X ** 2 + Y ** 2) / 10  # definition of f
    T = np.sin(X * Y * Z)

    #norm = mpl.colors.Normalize(vmin=np.amin(T), vmax=np.amax(T))
    T = mpl.cm.hot(T)  # change T to colors
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, facecolors=T, linewidth=0,
                           cstride=1, rstride=1)
    plt.show()

'''
    X, Y = np.mgrid[0:512:512j, 0:612:612j]
    Z = imageG
    T = np.sin(X * Y * Z)
    norm = mpl.colors.Normalize(vmin=np.amin(T), vmax=np.amax(T))
    T = mpl.cm.hot(T)  # change T to colors
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X, Y, Z, facecolors=T, linewidth=0,
                           cstride=1, rstride=1)
    plt.show()

    return imageG


