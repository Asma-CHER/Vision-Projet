import math
import pickle

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
# Axes3D import has side effects, it enables using projection='3d' in add_subplot
import random

from scipy.sparse import csr_matrix, lil_matrix
from numpy.linalg import pinv

import Functions_Etape1 as f1
import numpy as np


def fun(image , x, y):
    return image[y,x]

def save_file(x):
    with open("z.pkl", 'wb') as f:
        p = pickle.Pickler(f)
        p.dump(x)


def depth_map_generation(obj_mask, normals_mat):
    """Generate the depth map of the object

    Keyword arguments:
    obj_mask -- the mask of the object
    normals_mat -- the normals of the object
    """

    # don't know what type of data to use

    h, w = obj_mask.shape

    index = np.zeros((h, w), np.uint8)

    object_pixel_rows = []
    object_pixel_cols = []

    # check if there is a function in np to do this
    for i in range(h):
        for j in range(w):
            if obj_mask[i, j] == 1:
                object_pixel_rows.append(i)
                object_pixel_cols.append(j)
    object_pixels = np.vstack((object_pixel_rows, object_pixel_cols))


    #print("after vstack : object_pixels ",object_pixels)

    nb_pixels = object_pixels.shape[1]

    #print(" here Total number of Pixels within the mask " , nb_pixels)

    #print(" here index before modif " , index)
    for d in range(nb_pixels):
        index[object_pixels[0, d], object_pixels[1, d]] = d

    #print(" here index after modif " , index)
    #print(" here index [74,374]" , index[74,374])
    #print(" here index [74,375]" , index[74,375])


    #print(" nb_pixels " , nb_pixels)

    m = lil_matrix((2 * nb_pixels, nb_pixels), dtype=np.float32) #ils ont utilise une matrice creuse
    b = np.zeros((2 * nb_pixels, 1), np.float32)
    # ça sert à quelque chose ?
    n = np.zeros((2 * nb_pixels, 1), np.float32) #pour sauvgarder les vals de z

    for d in range(nb_pixels):
        p_row = object_pixels[0, d]
        p_col = object_pixels[1, d]
        nx = normals_mat[p_row, p_col, 0]
        ny = normals_mat[p_row, p_col, 1]
        nz = normals_mat[p_row, p_col, 2]

        if index[p_row, p_col + 1] > 0 and index[p_row - 1, p_col] > 0:
            m[2 * d - 1, index[p_row, p_col]] = 1
            m[2 * d - 1, index[p_row, p_col + 1]] = -1
            b[2 * d - 1, 0] = nx / nz

            m[2 * d, index[p_row, p_col]] = 1
            m[2 * d, index[p_row - 1, p_col]] = -1
            b[2 * d, 0] = ny / nz

        elif index[p_row - 1, p_col] > 0:
            f = -1
            if index[p_row, p_col + 1] > 0:
                m[2 * d - 1, index[p_row, p_col]] = 1
                m[2 * d - 1, index[p_row, p_col + f]] = -1
                b[2 * d - 1, 0] = f * nx / nz

            m[2 * d, index[p_row, p_col]] = 1
            m[2 * d, index[p_row - 1, p_col]] = -1
            b[2 * d, 0] = ny / nz

        elif index[p_row, p_col + 1] > 0:
            f = -1
            if index[p_row - f, p_col] > 0:
                m[2 * d, index[p_row, p_col]] = 1
                m[2 * d, index[p_row - f, p_col]] = -1
                n[2 * d, 0] = f * ny / nz

            m[2 * d - 1, index[p_row, p_col]] = 1
            m[2 * d - 1, index[p_row, p_col + 1]] = -1
            n[2 * d - 1, 0] = nx / nz

        else:
            f = -1
            if index[p_row, p_col + f] > 0:
                m[2 * d - 1, index[p_row, p_col]] = 1
                m[2 * d - 1, index[p_row, p_col + f]] = -1
                n[2 * d - 1, 0] = f * nx / nz

            f = -1
            if index[p_row - f, p_col] > 0:
                m[2 * d, index[p_row, p_col]] = 1
                m[2 * d, index[p_row - f, p_col]] = -1
                b[2 * d, 0] = f * ny / nz

    # the problem is here
    #print("here before prlm we have : \n shape(m)=",m.shape,"\n shape(b)=",b.shape," b=",b)
    #x = np.linalg.lstsq(b, m.toarray())
    #Binv = pinv(b)
    #print("here we have : \n shape(m)=", m.shape,"\n shape(bInv)=", Binv.shape)
    #x = Binv.dot(m)
    #x = m/b
    r, c = m.nonzero()
    #rD_sp = csr_matrix(((1.0 / b)[r], (r, c)), shape=(m.shape))
    #val = np.repeat(Binv, m.getnnz(axis=1))
    val = np.repeat(1.0 / b, m.getnnz(axis=1))
    rD_sp = csr_matrix((val, (r, c)), shape=(m.shape))
    x = m.multiply(rD_sp)

    #print("after div ",x.dtype," x= ",x)
    #x = np.subtract(x , min(x))
    #print("after x-min(x) ",x)

    temp_shape = np.zeros((h, w), np.float32)
    for d in range(nb_pixels):
        p_row = object_pixels[0, d]
        p_col = object_pixels[1, d]
        temp_shape[p_row, p_col] = x[0, d]

    z = np.zeros((h, w), np.float32)
    for i in range(h):
        for j in range(w):
            z[i, j] = temp_shape[i, j]

    print(" here Z= ",z.shape,z)
    print(" here Z= ",z[74,374])
    #z = (z+1)/2*255

    #save_file(z)
    return z

def z(mask , z):
    '''
    yy = []
    xx = []
    h, w = mask.shape
    for y1 in range(h):
        for x1 in range(w):
            if(mask[y1,x1]==1):
                yy.append(y1)
                xx.append(x1)

    fig = plt.figure()
    ax = fig.add_subplot( projection='3d')
    yAxe  = np.arange(0, 512, 1)
    xAxe  = np.arange(0, 612, 1)

    x = np.array(xx)
    #print("here x ", x)
    y = np.array(yy)

    X, Y = np.meshgrid(xAxe, yAxe)


    zs = np.array(fun(image, np.ravel(X), np.ravel(Y)))
    #zs = np.array(fun(imageG, x, y))
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z, cmap="summer") #linewidth=0,
                           #cstride=1, rstride=1)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
    '''

    ax = plt.axes(projection="3d")
    x, y = np.mgrid[0:512:512j, 0:612:612j]
    ax.plot_surface(x, y, z,cmap="summer")
    plt.show()


    #return image


def calcul_3D(mask, image):

    imageG = f1.image_in_gray(image)
    #imageG = (imageG+1)/2*255
    p = np.gradient(imageG, 1, axis=1)
    q = np.gradient(imageG, 1, axis=0)


    z =np.zeros(imageG.shape,np.float32)
    h, w =imageG.shape

    for y in range(1,h):
        z[y,0] = z[y-1,0] - q[y,0]

    for y in range(h):
        for x in range(1,w):
            if mask[y,x] == 1 :
                z[y,x] = z[y, x-1] - p[y,x]





    return z
