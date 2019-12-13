#import tqdm for progress bar
from tqdm import tqdm
#import compare_ssim to compute similarity distance
#from skimage.measure import compare_ssim
import numpy as np

class SimilarityFunction:

    def __init__(self, df, model_type):
        self.df = df
        self.model_type = model_type
    
    #extracting the distances
    @staticmethod
    def extract_distance(json):
        try:
            return float(json["s"])
        except KeyError:
            return 0
        
    #finding similarities by cosine method
    def get_similarity(self, image_id_1, image_id_2):
        #compute dot product of two images and returns a scalar
        dot_product = np.dot(self.df.loc[image_id_1, :], self.df.loc[image_id_2, :])
        #calculate the magnitude 
        magnitude = np.linalg.norm(self.df.loc[image_id_1, :]) * np.linalg.norm(self.df.loc[image_id_2, :])
        return (dot_product / magnitude)

    #get the similarity for all images and store in distances, then sort
    def get_image_distances(self, image_id):
        distances = []
        print("Calculating distances of images from image_id = " + str(image_id))
        #progression bar by tqdm 
        for other_image_id in tqdm([x for x in self.df.index.values if x != image_id]):
            distances.append({"s": self.get_similarity(image_id, other_image_id),
                              "other_image_id": other_image_id})
        #sorting the distances
        distances.sort(key=self.extract_distance, reverse=True)
        return distances