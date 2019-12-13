#import the libraries
import cv2
import numpy as np
from scipy import stats
from feature_conversions import Feature_Extraction

class ColorMoments(Feature_Extraction):

    def __init__(self):
        Feature_Extraction.__init__(self)

    
    #method that calculates and returns color moments for all three channels of a single window which is used above
    @staticmethod
    def calculate_color_moments_from_window(image_window_yuv):
        #splitting image in y,u and v
        image_window_y, image_window_u, image_window_v = cv2.split(image_window_yuv)
        #calculating the color moments for single window

        image_window_color_moments = np.ravel(
            [[np.mean(channel), np.math.sqrt(np.var(channel)), stats.skew(channel.ravel())] for channel in
             [image_window_y, image_window_u, image_window_v]])

        return image_window_color_moments

    #method that takes image path as input and calculate and return the color moments for all windows using second function
    def calculate_color_moments_from_image(self, image_path):
        #color_image=[]
        #convert to yuv
        yuv_image = Feature_Extraction.convert_to_yuv(image_path)
        #splitting the image in 100 * 100 windows
        image_windows_yuv = Feature_Extraction.split_image_into_windows(yuv_image)
        #calling the calculate_color_moments on image_windows to get the color moments
        color_image=np.ravel([self.calculate_color_moments_from_window(image_window_yuv) for image_window_yuv in image_windows_yuv])
        return color_image
