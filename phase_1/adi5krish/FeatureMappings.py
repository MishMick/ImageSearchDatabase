import cv2
import numpy as np
from skimage import feature,data,exposure
import scipy

class Mapping:


	def __init__(self):
		pass


	@staticmethod
	def convert_yuv(image_path):   # This method reads the RGB image from  the image_path and converts it into YUV image
	    img = cv2.imread(image_path)
	    img_yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
	    return img_yuv

	@staticmethod
	def convert_rgb_gray(image_path):  # This method reads the RGB image from  the image_path and converts it into GRAY image
	    img = cv2.imread(image_path)
	    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	    return img_gray

	@staticmethod
	def divide_into_windows(image):  # Divides the input image into windows of size 100*100 on which features will be extracted and appending all the windows
										# into a list which gives us an 3d array
		image_windows = []

		for i in range(0,int(image.shape[0]/100)):
			for j in range(0,int(image.shape[1]/100)):
				image_block = image[i*100:(i+1)*100,j*100:(j+1)*100]
				image_windows.append(image_block)
		return image_windows    	

		
class BinaryPatterns(Mapping):


	@staticmethod
	def find_lbp(image_path):  # Extracts the local binary patterns from the image_path


		LBP = []
		grey_scale_image = Mapping.convert_rgb_gray(image_path)  # converting rgb image to gray scale using above implemented method
		image_windows = Mapping.divide_into_windows(grey_scale_image) # divides the gray scale image into windows
		LBP = np.array([BinaryPatterns.find_lbp_block(window) for window in image_windows]).ravel() # Extracting features on all the windows and flattening the
																									# array
		return LBP

	def find_lbp_block(image_block): # this method extracts the features given the image block

		eps = 1e-7
		radius = 2
		no_of_points = radius * 8
		lbp = feature.local_binary_pattern(image_block,no_of_points,radius,method = "uniform") 
		(hist,_) = np.histogram(lbp.ravel(), bins = np.arange(0,(no_of_points+1))) 
		hist = hist.astype('float')
		hist = hist/(hist.sum()+eps)  # Normalizing the data
		return hist    


class ColorMoments(Mapping):  

	
	@staticmethod
	def find_color_moments(image_path): # Extracts the color moments from the image_path

		CM = []
		yuv_image = Mapping.convert_yuv(image_path) # converting rbg to yuv image
		image_windows = Mapping.divide_into_windows(yuv_image) # dividing the image into windows
		CM = np.ravel([ColorMoments.calculate_color_moments(window) for window in image_windows])  # appending all the features
		return CM


	def calculate_color_moments(image_block):
		image_window_y, image_window_u, image_window_v = cv2.split(image_block)  # splitting the YUV image into three channels
		color_moments = np.ravel([[np.mean(channel), np.math.sqrt(np.var(channel)), scipy.stats.skew(channel.ravel())] for channel in 
			[image_window_y, image_window_u, image_window_v]])  # Extracting color moments for each channel
		return color_moments

