import cv2
import numpy as np
import Functions_Etape1 as f1
import Functions_Etape2 as f2


#m
normal = f2.calcul_needle_map()

#normal = f2.Normale()

#n=normal
#image = f2.read_file("my_img.pkl")

#m = f2.showImage(n)
cv2.imshow("image source",normal)
cv2.imwrite('imageNormale.png', normal)
cv2.waitKey(0)
cv2.destroyAllWindows()



