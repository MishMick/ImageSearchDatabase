import glob


# Image factory stuff
import re


class ImageFactory:

    def __init__(self, base_path):
        self.base_path = base_path
        self.image_type = ".jpg"

    def get_image_path_list(self):
    	# print(self.base_path)
    	_image_paths = [_image_path for _image_path in glob.glob(self.base_path + "\*" + self.image_type)] # Given a folder of images, this returns the path of 
                                                                                                            # each image
    	return _image_paths

    @staticmethod
    def get_image_id_from_path(_image_path):
    	return str(_image_path).split("\Hand_")[1].split(".jpg")[0]  # Returns the ID of the image from the given path
