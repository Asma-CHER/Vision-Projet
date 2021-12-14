import cv2
import numpy as np
import Functions_Etape1 as f1
import Functions_Etape2 as f2
import Functions_Etape3 as f3

normal = f2.calcul_needle_map()

#normal = f2.Normale()

#n=normal
#image = f2.read_file("my_img.pkl")

#m = f2.showImage(n)

#****************************************
#mask = f1.load_objMask()
#z= f3.depth_map_generation(mask,normal)
#imageZ= f3.z(mask,z)
#****************************************
mask = f1.load_objMask()
z = f3.calcul_3D(mask,normal)
imageZ= f3.z(mask,z)



#print(imageZ)
#cv2.imshow("image nnn",normal)
#cv2.imshow("image z",imageZ)

#cv2.imwrite('imageNormale.png', normal)
cv2.waitKey(0)
cv2.destroyAllWindows()



