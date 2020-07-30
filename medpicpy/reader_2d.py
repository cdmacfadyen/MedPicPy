import SimpleITK as sitk
import cv2

# image = sitk.ReadImage("../mini-MIAS/mdb012.pgm")
image = cv2.imread("../mini-MIAS/mdb012.pgm", cv2.IMREAD_GRAYSCALE )

cv2.imshow('image', image)
cv2.waitKey(0)

resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)


cv2.imshow('resized', resized)
cv2.waitKey(0)

cv2.destroyAllWindows()
print(image.shape)

