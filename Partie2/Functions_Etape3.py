import math
import pickle


import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
# Axes3D import has side effects, it enables using projection='3d' in add_subplot
import random
import numpy.linalg as LA
from scipy.sparse import csr_matrix, lil_matrix
from numpy.linalg import pinv
#from skimage.transform import resize as imresize
#from skimage.filters import gaussian
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

def z(mask , image):

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
    yAxe  = np.arange(image.shape[0])
    xAxe  = np.arange(image.shape[1])



    X, Y = np.meshgrid(xAxe, yAxe)


    zs = np.array(fun(image, np.ravel(X), np.ravel(Y)))
    #zs = np.array(fun(imageG, x, y))
    Z = zs.reshape(X.shape)

    ax.set_zlim3d(-200, 200)
    ax.plot_surface(X, Y, Z, cmap="Greys") #linewidth=0,
                           #cstride=1, rstride=1)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    #ax.set_zlim3d(0,200)
    plt.show()
    '''

    ax = plt.axes(projection="3d")
    x, y = np.mgrid[0:512:512j, 0:612:612j]
    ax.plot_surface(x, y, z,cmap="summer" , linewidth=0, antialiased=False)
    plt.show()
'''

    #return image
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)
def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def calcul_3D(mask, image):

    #imageG = f1.image_in_gray(image)
    imageG = image
    '''
    imageG = image
    h, w, c = imageG.shape
    z = np.zeros((h, w), np.float32)

    for y in range(1, h):
        z[y,0] = 0

    vecX = np.array([0,1,0])
    for y in range(h):
        for x in range(1,w):
            if mask[y,x] == 1 :
                t_x_y = np.transpose(imageG[y,x])
                #t_x_y = np.rot90(imageG[y,x])
                #print("T ",t_x_y)
                #alpha2 = np.angle(imageG[y,x],vecX)
                v1_u = unit_vector(t_x_y)
                v2_u = unit_vector(vecX)
                rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
                #rad = angle(imageG[y,x],vecX)
                deg = np.rad2deg(rad)
                #deg = 90-alpha2Deg

                #deg = np.rad2deg(rad)
                print("here deg ",deg)

                #print(alpha.dtype)

                z[y,x]= z[y, x-1] + math.tan(deg)



    '''

    #p = np.gradient(imageG[:,:,1], 1, axis=1)
    #q = np.gradient(imageG[:,:,1], 1, axis=0)

    #print(p[300,300],q[300,300])

    h, w, c = imageG.shape
    z =np.zeros((h,w),np.float32)

    n_x, n_y, n_z = imageG[:,:,0], imageG[:,:,1], imageG[:,:,2]

    n_z[n_z==0] =1
    p = n_x/n_z
    q = n_y/n_z

    #zx = np.zeros((h,w),np.float32)
    #zy = np.zeros((h,w),np.float32)

   # for x in range(w-1):
    #    zx[:,x+1] = zx[:,x] - p[:,x]

    #for y in range(h - 1):
     #   zy[y + 1, :] = zy[y, :] - q[y, :]

    for x in range(1, w):
        z[0, x] = z[0, x - 1] - p[0, x]

    for x in range(w):
        for y in range(1, h):
            if mask[y, x] == 1:
                z[y, x] = z[y - 1, x] - q[y, x]
            else:
                z[y, x] = 0
    #z = (zx-zy)/2


    '''
    for y in range(1,h):
        q = imageG[y, 0, 1] / imageG[y, 0, 2]
        z[y,0] = z[y-1,0] - q

    for y in range(h):
        for x in range(1,w):
            if mask[y,x] == 1 :
                p = imageG[y,x,0]/imageG[y,x,2]
                z[y,x] = z[y, x-1] - p
            else:
                z[y,x]=0
    
    for x in range(1,w):
        z[0,x] = z[0,x-1] - p[0,x]

    for x in range(w):
        for y in range(1,h):
            if mask[y,x] == 1 :
                z[y,x] = z[y-1, x] - q[y,x]
            else:
                z[y,x]=0

    z = np.sqrt(gaussian(imresize(z, z.shape, order=2, mode='reflect'), 10))
    '''
    return z*mask
