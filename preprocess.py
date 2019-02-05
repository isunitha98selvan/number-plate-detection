import cv2
import numpy as np

def canny_edge(image):
	edges=cv2.Canny(image,170,200)
	cv2.imwrite('HPIM0942(2).png',edges)
	

def prewitt_edge(image):
	img_gaussian = cv2.GaussianBlur(image,(3,3),0)
	kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
	kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
	img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
	img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
	img_prewitt=img_prewittx+img_prewitty
	cv2.imwrite('HPIM0942(3).png',img_prewitt)

def sobel_edge(image):
	img_gaussian = cv2.GaussianBlur(image,(3,3),0)
	img_sobelx = cv2.Sobel(img_gaussian,cv2.CV_8U,1,0,ksize=5)
	img_sobely = cv2.Sobel(img_gaussian,cv2.CV_8U,0,1,ksize=5)
	img_sobel = img_sobelx + img_sobely
	cv2.imwrite('HPIM0942(4).png',img_sobel)


def main():
	#loads image in grayscale
	img=cv2.imread('car.jpg')
	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	
	cv2.imwrite('HPIM0942(1).png',img)
	img = cv2.bilateralFilter(img, 11, 17, 17)
	#various edge detection algorithms
	canny_edge(img)
	
	#Binarising the image
	img_bin=cv2.imread('HPIM0942(2).png',0)
	#OTSU Binarization
	ret3,binary_img = cv2.threshold(img_bin,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	cv2.imwrite('binarised_img.png',binary_img)
	#Finding contours
	(cnts, _) = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[0:2] 
	detected_plate_contour = None 
	

	# loop over our contours to find the best possible approximate contour of number plate
	count = 0
	for c in cnts:
		perimeter = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
		if len(approx) == 4:  # Select the contour with 4 corners
			detected_plate_contour = approx 
			break

	print(detected_plate_contour)

	# Drawing the selected contour on the original image
	cv2.drawContours(img,[detected_plate_contour], -1, (0,255,0), 3)
	cv2.imshow("Number plate", img)

	cv2.waitKey(0)

if __name__=='__main__':
	main()
