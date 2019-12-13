#import libraries
import cv2
import scipy

class Feature_Extraction:
    
    def __init__(self):
        pass
    
    #convert image from BGR to YUV
    def convert_to_yuv(image_path):
        image = cv2.imread(image_path)
        yuv_image = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
        return yuv_image
    
    #splitting the image into 100 * 100 windows
    def split_image_into_windows(image):
        windows = []
        for i in range(0, image.shape[0], 100): 
            for j in range(0, image.shape[1], 100): 
                windows.append(image[i:(i + 100), j:(j + 100)])

        return np.array(windows)

    #convert image from RGB to Grayscale
    def convert_to_greyscale(image_path):
        image=cv2.imread(image_path)
        grayscale_image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        return grayscale_image
    
   #downsampling the image in ratio 1:10
    def downsample_image(image_path):
        image = cv2.imread(image_path, 0)
        image = scipy.misc.imresize(image, 0.1)
        return image