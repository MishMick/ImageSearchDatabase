from tqdm import tqdm
import json

from arvind_hasti.feature_extraction.image_factory import ImageFactory
from arvind_hasti.feature_extraction.extract_features import ColorMomentExtraction, LbpExtraction, HogExtraction
from arvind_hasti.store.feature_store import FeatureStorage, ModelType
from arvind_hasti.plot.image_plot import ImagePlot
from arvind_hasti.compare.similarity_function import CosineSimilarityFunction
from arvind_hasti.config.config_provider import ConfigProvider

import multiprocessing
import os
from contextlib import contextmanager

config_path = "/home/arvind-hasti/Documents/MWDB/cse515-mwdb-project/phase_1/arvind_hasti/config.json"
config_provider = ConfigProvider(config_path)
image_base_path = config_provider.get_image_base_path()
storage_path = config_provider.get_storage_path()
output_path = config_provider.get_output_path()


image_factory = ImageFactory(image_base_path)
image_paths = image_factory.get_image_path_list()

cm_vector_mapping = ColorMomentExtraction()
lbp_vector_mapping = LbpExtraction()
hog_vector_mapping = HogExtraction()
feature_storage = FeatureStorage(storage_path)


def do_lbp_create(_image_factory, _lbp_vector_mapping, _feature_storage):
    # LBP
    print("Creating LBP features:")
    lbp_feature_vectors = {}
    for _image_path in tqdm(_image_factory.get_image_path_list()):
        _image_id = _image_factory.get_image_id_from_path(_image_path)
        lbp_from_image = _lbp_vector_mapping.get_lbp_from_image(_image_path)
        lbp_feature_vectors[_image_id] = lbp_from_image.tolist()

    _feature_storage.clear_model_storage(ModelType.LBP)
    _feature_storage.store_feature_set(ModelType.LBP, lbp_feature_vectors)
    print("Features stored in " + storage_path + "lbp.json")


def do_cm_create(_image_factory, _cm_vector_mapping, _feature_storage):
    # CM
    print("Creating Color Moments features:")
    cm_feature_vectors = {}
    for _image_path in tqdm(_image_factory.get_image_path_list()):
        _image_id = _image_factory.get_image_id_from_path(_image_path)
        cm_from_image = _cm_vector_mapping.get_color_moments_from_image(_image_path)
        cm_feature_vectors[_image_id] = cm_from_image.tolist()

    _feature_storage.clear_model_storage(ModelType.COLOR_MOMENTS)
    _feature_storage.store_feature_set(ModelType.COLOR_MOMENTS, cm_feature_vectors)
    print("Features stored in " + storage_path + "color_moments.json")


def do_hog_create(_image_factory, _hog_vector_mapping, _feature_storage):
    # HOG
    print("Creating HOG features:")
    hog_feature_vectors = {}
    for _image_path in tqdm(_image_factory.get_image_path_list()):
        _image_id = _image_factory.get_image_id_from_path(_image_path)
        hog_from_image = _hog_vector_mapping.get_hog_from_image(_image_path)
        hog_feature_vectors[_image_id] = hog_from_image.tolist()

    _feature_storage.clear_model_storage(ModelType.HOG)
    _feature_storage.store_feature_set(ModelType.HOG, hog_feature_vectors)
    print("Features stored in " + storage_path + "hog.json")


def do_df_load(model_type, _feature_storage):
    return _feature_storage.load_to_df(model_type)
    #return _feature_storage.load_to_df(ModelType.COLOR_MOMENTS)


def do_distances_get(model_type, image_id, k, _feature_storage):
    df = feature_storage.load_to_df(model_type)
    similarity_function = CosineSimilarityFunction(df, model_type)
    distances_json = similarity_function.get_image_distances(image_id)[:k]
    return distances_json


def do_plot(image_id, distances_json):
    image_plot = ImagePlot(image_id, image_base_path)
    other_image_ids = [node["other_image_id"] for node in distances_json]
    return image_plot.plot_comparison(other_image_ids)

def save_plot(plt, image_id, query_info):
    output_directory = output_path + "/query_image_" + str(image_id)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_image_path = output_directory + "/output.png"
    output_info_path = output_directory + "/query.json"
    plt.savefig(output_image_path)
    print(query_info)
    with open(output_info_path, 'w') as f:
        #json.dump(query_info, f)
        f.write(str(query_info))