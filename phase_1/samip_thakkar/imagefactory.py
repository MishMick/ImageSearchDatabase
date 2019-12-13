#import glob to match path directory
import glob
#import regular expression
import re

class ImageFactory:


    def __init__(self, base_path):
        self.base_path = base_path
        self.image_type = ".jpg"
    ##glob for finding all pathnames matching the pattern 
    def get_image_path_list(self):
        image_paths = [image_path for image_path in glob.glob(self.base_path + "/*" + self.image_type)]
        return image_paths

    @staticmethod
    #setting regular expression to get the path of images in format Hands + id + .jpg returns series
    def get_image_id_from_path(image_path):
        return re.findall(r'\d+', str(image_path).split("/Hands/")[1].split(".jpg")[0])[0]