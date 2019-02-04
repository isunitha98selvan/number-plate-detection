import cv2
import numpy as np

def canny_edge(image):
	edges=cv2.Canny(image,100,200)
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
	img=cv2.imread('HPIM0942.JPG',0)
	cv2.imwrite('HPIM0942(1).png',img)
	#various edge detection algorithms
	canny_edge(img)
	prewitt_edge(img)
	sobel_edge(img)
	#Binarising the image
	cv2.imwrite('HPIM0942(2).png',img)
	#OTSU Binarization
	ret3,binary_img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	#th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	cv2.imwrite('binarised_img.png',binary_img)
	

if __name__=='__main__':
	main()
