import numpy
from numpy.linalg import norm
from skimage.measure import compare_ssim
from tqdm import tqdm


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

    def get_similarity(self, image_id_1, image_id_2):
        pass

    def get_image_distances(self, image_id):
        distances = []
        for other_image_id in tqdm([x for x in self.df.index.values if x != image_id]):
            distances.append({"s": self.get_similarity(image_id, other_image_id),
                              "other_image_id": other_image_id})

        print(type(distances))
        distances.sort(key=self.extract_distance, reverse=True)
        return distances

class CosineSimilarityFunction(SimilarityFunction):

    def __init__(self, df, model_type):
        SimilarityFunction.__init__(self, df, model_type)

    def get_similarity(self, image_id_1, image_id_2):
        dot_prod = numpy.dot(self.df.loc[image_id_1, :], self.df.loc[image_id_2, :])
        norms = (norm(self.df.loc[image_id_1, :]) * norm(self.df.loc[image_id_2, :]))
        sim = dot_prod / norms
        #sim_val = sim.item()
        # print(sim_val)
        # print(type(sim_val))
        return sim
