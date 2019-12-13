import cv2
import numpy as np
import scipy
from skimage.feature import local_binary_pattern

# Image utilities
class ExtractFeature:
    def __init__(self):
        pass
    @staticmethod
    def get_yuv_image(_image_path):
        _image = cv2.imread(_image_path)
        _image_yuv = cv2.cvtColor(_image, cv2.COLOR_BGR2YUV)
        return _image_yuv
    @staticmethod
    def get_grayscale_for_image(_image_path):
        _image = cv2.imread(_image_path)
        _image_grayscale = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
        return _image_grayscale
    @staticmethod
    def get_image_windows(_image, nrows=100, ncols=100):
        #rows, cols = _image.shape
        #if rows % nrows !=0 || cols % ncols != 0:
        #    print("rows or cols can't be equally divided")

        _windows = []
        for i in range(0, _image.shape[0], nrows):  # for every pixel take 100 at a time
            for j in range(0, _image.shape[1], ncols):  # for every pixel take 100 at a time
                _windows.append(_image[i:(i + 100), j:(j + 100)])

        return np.array(_windows)
        

class ColorMomentExtraction(ExtractFeature):

    def __init__(self):
        ExtractFeature.__init__(self)

    @staticmethod
    def get_color_moments_from_image_window(_image_window_yuv):
        image_window_y, image_window_u, image_window_v = cv2.split(_image_window_yuv)

        image_window_color_moments = np.ravel(
            [[np.mean(channel), np.math.sqrt(np.var(channel)), scipy.stats.skew(channel.ravel())] for channel in
             [image_window_y, image_window_u, image_window_v]])

        return image_window_color_moments

    def get_color_moments_from_image(self, _image_path):
        _image_yuv = ExtractFeature.get_yuv_image(_image_path)
        image_windows_yuv = ExtractFeature.get_image_windows(_image_yuv)

        _cm_image = np.ravel(
            [self.get_color_moments_from_image_window(image_window_yuv) for image_window_yuv in image_windows_yuv]
        )

        return _cm_image




class LbpExtraction(ExtractFeature):
    
    def __init__(self):
        ExtractFeature.__init__(self)

    @staticmethod
    def get_lbp_from_image_window(_image_window):
        radius = 2
        no_points = 8 * radius
        _lbp = np.array(local_binary_pattern(_image_window, no_points, radius, method='uniform'))
        (hist, _) = np.histogram(_lbp.ravel(), bins=np.arange(0, no_points + 3), range=(0, no_points + 2))
        hist = hist.astype("float")
        hist = hist / (hist.sum() + 1e-7)
        return hist

    def get_lbp_from_image(self, _image_path):
        image_greyscale = ExtractFeature.get_grayscale_for_image(_image_path)
        image_windows_greyscale = ExtractFeature.get_image_windows(image_greyscale)
        _lbp_image = np.array([self.get_lbp_from_image_window(window) for window in image_windows_greyscale]).ravel()
        return _lbp_image

class HogExtraction(ExtractFeature):
    def __init__(self):
        ExtractFeature.__init__(self)

    @staticmethod
    def get_hog_for_image(_image_path):
        image = cv2.imread(_image_path, 0)
        image = scipy.misc.imresize(image, 0.1)
        win_size = (64, 64)
        block_size = (8, 8)
        block_stride = (8, 8)
        cell_size = (2, 2)
        no_bins = 9
        deriv_aperture = 1
        win_sigma = 4.
        histogram_norm_type = 0
        l2_hys_threshold = 2.0000000000000001e-01
        gamma_correction = 0
        nlevels = 64
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, no_bins, deriv_aperture, win_sigma,
                                histogram_norm_type, l2_hys_threshold, gamma_correction, nlevels)

        win_stride = (8, 8)
        padding = (8, 8)
        locations = ((10, 20),)
        hist = hog.compute(image, win_stride, padding, locations)
        hist = hist.ravel()

        return hist

class SiftVectorMapping(ExtractFeature):

    def __init__(self):
        ExtractFeature.__init__(self)

    def get_sift_from_image(self, _image_path):
        sift = cv2.xfeatures2d.SIFT_create()
        _image = self.get_grayscale_for_image(_image_path)
        kp, descriptors = sift.detectAndCompute(_image, None)
        descriptors = descriptors.ravel()
        return kp, descriptors

'''
sift_vector_mapping = SiftVectorMapping()

img1 = cv2.imread("/Users/sandeepkunichi/Work/ASU/2019_3/CSE515/DEV/cse515-mwdb-project/data/Hands/Hand_0000002.jpg", 0)
img2 = cv2.imread("/Users/sandeepkunichi/Work/ASU/2019_3/CSE515/DEV/cse515-mwdb-project/data/Hands/Hand_0000005.jpg", 0)

kp1, des1 = sift_vector_mapping.get_sift_from_image(
    "/Users/sandeepkunichi/Work/ASU/2019_3/CSE515/DEV/cse515-mwdb-project"
    "/data/Hands/Hand_0000002.jpg")

kp2, des2 = sift_vector_mapping.get_sift_from_image(
    "/Users/sandeepkunichi/Work/ASU/2019_3/CSE515/DEV/cse515-mwdb-project"
    "/data/Hands/Hand_0000004.jpg")

bf = cv2.BFMatcher()

matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
        a = len(good)
        percent = (a*100)/len(kp2)
        print("{} % similarity".format(percent))
        if percent >= 75.00:
            print('Match Found')
        if percent < 75.00:
            print('Match not Found')
'''