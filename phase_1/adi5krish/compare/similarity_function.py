from tqdm import tqdm
import numpy as np
import math
import cv2

from skimage.measure import compare_ssim


class SimilarityFunction:

    def __init__(self, df, model_type):
        self.df = df
        self.model_type = model_type

    @staticmethod
    def extract_distance(json):
        try:
            return float(json["s"])
        except KeyError:
            return 0

    # def get_ssim_for_images(self, image_id_1, image_id_2):
    #     return compare_ssim(np.ravel(self.df[image_id_1]), np.ravel(self.df[image_id_2]))
    #     print(self.df[image_id_1])

    def get_image_distances(self, image_id):  # This method calculates similarity scores with respect to all other images given an image
                                                # and stores the distances in the descending order so that we can return first K distances
                                                # when asked for K similar images
        distances = []
        # image_id = '000'+str(image_id)
        # print(image_id)
        print("Calculating distances of images from image_id = " + str(image_id))

        for other_image_id in tqdm([x for x in self.df.keys() if x != image_id]):
            distances.append({"s": self.cosine_similarity(image_id, other_image_id),
                              "other_image_id": other_image_id})

        distances.sort(key=self.extract_distance, reverse=True)
        return distances

    def cosine_similarity(self,image_id_1,image_id_2):   # The cosine similarity function that returns the similarity scores for the two input images
         
        dot_product = np.dot(np.ravel(self.df[image_id_1]),np.ravel(self.df[image_id_2]))
        magnitude = np.linalg.norm((np.ravel(self.df[image_id_1]))*np.ravel(np.linalg.norm(self.df[image_id_2])))

        return dot_product/magnitude


