import json
import os
import pickle
import sys

import pandas as pd
from enum import Enum


class ModelType(Enum):
    LBP = "lbp"
    COLOR_MOMENTS = "cm"


class FeatureStorage:

    def __init__(self, storage_path):
        self.storage_path = storage_path
        self.switcher_json = {
            ModelType.LBP: "_lbp.json",
            ModelType.COLOR_MOMENTS: "_color_moments.json",
        }
        self.switcher_pkl = {
            ModelType.LBP: "_lbp.pkl",
            ModelType.COLOR_MOMENTS: "_color_moments.pkl",
        }

        self.switcher_distances = {
            ModelType.LBP: "lbp.distances.json",
            ModelType.COLOR_MOMENTS: "color_moments.distances.json",
        }

    def store_feature_set_json(self, model_type, feature_vectors):  # Method that stores the features into a json file and calls the pickle function where 
                                                                    # the features are also stored as pickle file
        db_file = self.storage_path + self.switcher_json.get(model_type)  
        with open(db_file, 'w') as f:
            json.dump(feature_vectors, f)
        self.store_feature_set_pickle(model_type,feature_vectors)

    def load_to_df(self, model_type):   # Method that returns the features for a given model
        
        if(model_type == 'cm'):
            model_type = ModelType.COLOR_MOMENTS
        elif(model_type == 'lbp'):
            model_type = ModelType.LBP
        db_file = self.storage_path + self.switcher_pkl.get(model_type)
        df = pd.read_pickle(db_file)
        return df

    def store_feature_set_pickle(self, model_type,feature_vectors):  # Method to store features as pickle file
        pickle_file = self.storage_path + self.switcher_pkl.get(model_type)
        with open(pickle_file,'wb') as f:
            pickle.dump(feature_vectors,f)

    def store_distances(self, model_type, distances_json):  # Method that stores the distances for a given imageID, feature model and K
        db_file = self.storage_path + self.switcher_distances.get(model_type)

        with open(db_file, 'w') as f:
            json.dump(distances_json, f)
