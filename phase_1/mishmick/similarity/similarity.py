import numpy as np
from tqdm import tqdm


class SimilarityFunction:

    def __init__(self, df):
        self.df = df

    @staticmethod
    def extract_distance(json):
        try:
            return float(json["s"])
        except KeyError:
            return 0

    def get_euclidean_sim(self, image_id_1, image_id_2):
        return np.linalg.norm(np.ravel(self.df[str(image_id_1).zfill(7)]) - np.ravel(self.df[str(image_id_2).zfill(7)]))

    def get_cosine_sim(self, image_id_1, image_id_2):
        dot_product = np.dot(np.ravel(self.df[str(image_id_1).zfill(7)]), np.ravel(self.df[str(image_id_2).zfill(7)]))
        magnitude = np.linalg.norm(
            (np.ravel(self.df[str(image_id_1).zfill(7)])) * np.ravel(np.linalg.norm(self.df[str(image_id_2).zfill(7)])))
        return dot_product / magnitude

    def get_cm_distances(self, image_id):
        distances = []
        print("Calculating cosine distances of images from image_id = " + str(image_id).zfill(7))
        for other_image_id in tqdm([x for x in self.df if x != str(image_id).zfill(7)]):
            distances.append({"s": self.get_cosine_sim(str(image_id).zfill(7), str(other_image_id).zfill(7)),
                              "other_image_id": other_image_id})
        distances.sort(key=self.extract_distance, reverse=True)
        return distances

    def get_sift_distances(self, image_id):
        distances = []
        print("Calculating euclidean distances of images from image_id = " + str(image_id).zfill(7))
        for other_image_id in tqdm([x for x in self.df if x != str(image_id).zfill(7)]):
            distances.append({"s": self.get_euclidean_sim(str(image_id).zfill(7), str(other_image_id).zfill(7)),
                              "other_image_id": other_image_id})
        distances.sort(key=self.extract_distance, reverse=False)
        return distances
