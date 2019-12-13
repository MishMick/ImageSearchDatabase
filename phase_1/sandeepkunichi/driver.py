import json

from compare.similarity_function import *
from mapping.image_factory import ImageFactory
from mapping.vector_mapping import *
from config.config_provider import ConfigProvider
from plot.image_plot import ImagePlot
from store.feature_store import FeatureStorage, ModelType

import multiprocessing
import os
from contextlib import contextmanager

config_path = "/Users/sandeepkunichi/Work/ASU/2019_3/CSE515/DEV/cse515-mwdb-project/phase_1/sandeepkunichi/config.json"
config_provider = ConfigProvider(config_path)
image_base_path = config_provider.get_image_base_path()
storage_path = config_provider.get_storage_path()
output_path = config_provider.get_output_path()

image_factory = ImageFactory(image_base_path)
lbp_vector_mapping = LbpVectorMapping()
cm_vector_mapping = ColorMomentsVectorMapping()
hog_vector_mapping = HogVectorMapping()
sift_vector_mapping = SiftVectorMapping()
feature_storage = FeatureStorage(storage_path)
image_paths = image_factory.get_image_path_list()

distance_measure_switcher = {
    "cosine": CosineSimilarityFunction,
    "l2": EuclideanSimilarityFunction
}


@contextmanager
def pool_context(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def do_cm_task(_image_path):
    _image_id = image_factory.get_image_id_from_path(_image_path)
    cm_from_image = cm_vector_mapping.get_color_moments_from_image(_image_path)
    return _image_id, cm_from_image.tolist()


def do_lbp_task(_image_path):
    _image_id = image_factory.get_image_id_from_path(_image_path)
    lbp_from_image = lbp_vector_mapping.get_lbp_from_image(_image_path)
    return _image_id, lbp_from_image.tolist()


def do_hog_task(_image_path):
    _image_id = image_factory.get_image_id_from_path(_image_path)
    hog_from_image = hog_vector_mapping.get_hog_from_image(_image_path)
    return _image_id, hog_from_image.tolist()


def do_sift_task(_image_path):
    _image_id = image_factory.get_image_id_from_path(_image_path)
    sift_from_image = sift_vector_mapping.get_sift_from_image(_image_path)
    return _image_id, sift_from_image.tolist()


def do_parallel(task):
    feature_vectors = {}
    with pool_context(processes=10) as pool:
        _image_paths = image_factory.get_image_path_list()
        for _image_id, _image_feature_vector in list(tqdm(pool.imap(task, _image_paths), total=len(_image_paths))):
            feature_vectors[_image_id] = _image_feature_vector
    return feature_vectors


def do_lbp_create(_image_factory, _lbp_vector_mapping, _feature_storage):
    # LBP
    print "Creating LBP features:"
    _feature_storage.clear_model_storage(ModelType.LBP)
    _feature_storage.store_feature_set(ModelType.LBP, do_parallel(do_lbp_task))
    print "Features stored in " + storage_path + "lbp.json"


def do_cm_create(_image_factory, _cm_vector_mapping, _feature_storage):
    # CM
    print "Creating Color Moments features:"
    _feature_storage.clear_model_storage(ModelType.COLOR_MOMENTS)
    _feature_storage.store_feature_set(ModelType.COLOR_MOMENTS, do_parallel(do_cm_task))
    print "Features stored in " + storage_path + "color_moments.json"


def do_hog_create(_image_factory, _hog_vector_mapping, _feature_storage):
    # HOG
    print "Creating HOG features:"
    _feature_storage.clear_model_storage(ModelType.HOG)
    _feature_storage.store_feature_set(ModelType.HOG, do_parallel(do_hog_task))
    print "Features stored in " + storage_path + "hog.json"


def do_sift_create(_image_factory, _sift_vector_mapping, _feature_storage):
    # SIFT
    print "Creating SIFT features:"
    _feature_storage.clear_model_storage(ModelType.SIFT)
    _feature_storage.store_feature_set(ModelType.SIFT, do_parallel(do_sift_task))
    print "Features stored in " + storage_path + "sift.json"


def do_df_load(model_type, _feature_storage):
    return _feature_storage.load_to_df(model_type)


def do_distances_get(distance_measure, model_type, image_id, k, _feature_storage):
    df = feature_storage.load_to_df(model_type)
    similarity_function = distance_measure_switcher.get(distance_measure)(df, model_type)
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
    with open(output_info_path, 'w') as f:
        json.dump(query_info, f)
