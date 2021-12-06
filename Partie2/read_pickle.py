import pickle
import cv2


fname ="all_normals.pkl"
fname1 ="S_inverse.pkl"
with open(fname,'rb') as f:
    x = pickle.load(f)

print(x)

with open(fname1,'rb') as f2:
    x2 = pickle.load(f2)

print(x2)


cv2.imshow("image ",x)
cv2.waitKey(0)
cv2.destroyAllWindows()
