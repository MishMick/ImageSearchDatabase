#import the libraries
import cv2
from feature_conversions import Feature_Extraction


class Histogram(Feature_Extraction):
    
    def __init__(self):
        Feature_Extraction.__init__(self)

  
    #method to calculate the histogram
    @staticmethod
    def calculate_histogram(image_path):
        #downsampling image in 1:10
        image=Feature_Extraction.downsample_image(image_path)
        
        window_size = (64, 64)                                   #window size - align to block size and block stride
        block_size = (8, 8)                                      #block size - align to cell size
        cell_size = (2, 2)                                       #cell size
        block_stride = (8, 8)                                    #block stride - must be multiple of cell size
        no_of_bins = 9                      
        derivative_aperture = 1                                  #used for shading
        window_sigma = 4.                                        #Guassian smoothing parameter
        histogram_norm_type = 0                                  #L2-Norm assigned
        l2_hys_threshold = 2.0000000000000001e-01                #L2-Hys normalization method shrinkage
        gamma_correction = 0                                     #flag specifying if gamma preprocessing is required or not
        nlevels = 64                                             #default number of windows detected
        
        #create Histogram descriptor 
        hog = cv2.HOGDescriptor(window_size, block_size, block_stride, cell_size, no_of_bins, derivative_aperture, window_sigma,
                                histogram_norm_type, l2_hys_threshold, gamma_correction, nlevels)
        window_stride = (8, 8)                                   #must be multiple of block stride
        padding = (8, 8)                                         
        locations = ((10, 20),)                                  
        
        #compute Hog descriptors of the image
        histogram = hog.compute(image, window_stride, padding, locations)
        histogram = histogram.ravel()                            #return 1d array
        return histogram
    
