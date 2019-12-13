from tqdm import tqdm
import numpy as np

from compare.similarity_function import SimilarityFunction
from mapping.image_factory import ImageFactory
from FeatureMappings import BinaryPatterns, ColorMoments
from plot.image_plot import ImagePlot
from store.feature_store import FeatureStorage, ModelType

# The set containing all the input images
image_base_path = "F:\Masters at ASU\Semester 1\Multimedia and Webdatases\CSE 515 Fall19 - Smaller Dataset\CSE 515 Fall19 - Smaller Dataset"
# The path where the extracted features would be stored
storage_path = "F:\Masters at ASU\Semester 1\Multimedia and Webdatases\cse515-mwdb-project-master\cse515-mwdb-project-master"

image_factory = ImageFactory(image_base_path)
lbp_vector_mapping = BinaryPatterns()
cm_vector_mapping = ColorMoments()
feature_storage = FeatureStorage(storage_path)
image_paths = image_factory.get_image_path_list()

# Method which calls the LBP function for extracting features and dumps the features in json file
def do_lbp_create(_image_factory, _lbp_vector_mapping, _feature_storage):
    # LBP
    print("Creating LBP features:")
    lbp_feature_vectors = {}
    for _image_path in tqdm(_image_factory.get_image_path_list()):
        _image_id = _image_factory.get_image_id_from_path(_image_path)
        lbp_from_image = _lbp_vector_mapping.find_lbp(_image_path)
        lbp_feature_vectors[_image_id] = lbp_from_image.tolist()
    _feature_storage.store_feature_set_json(ModelType.LBP, lbp_feature_vectors)

# Method which calls the Color Moments function for extracting features and dumps the features in json file
def do_cm_create(_image_factory, _cm_vector_mapping, _feature_storage):
    # CM
    print("Creating Color Moments features:")
    cm_feature_vectors = {}
    for _image_path in tqdm(_image_factory.get_image_path_list()):
        _image_id = _image_factory.get_image_id_from_path(_image_path)
        cm_from_image = _cm_vector_mapping.find_color_moments(_image_path)
        cm_feature_vectors[_image_id] = cm_from_image.tolist()
    _feature_storage.store_feature_set_json(ModelType.COLOR_MOMENTS, cm_feature_vectors)

# Method that loads the list, given a model type and image ID
def do_df_load(model_type, _feature_storage):
    return _feature_storage.load_to_df(model_type)

# Method which returns nearest K distances for the given image
def do_distances_get(model_type, image_id, k, _feature_storage):
    df = feature_storage.load_to_df(model_type)
    similarity_function = SimilarityFunction(df, model_type)
    distances_json = similarity_function.get_image_distances(image_id)[:k]
    return distances_json

# Method which plots the images corresponding to the distances 
def do_plot(image_id, distances_json):
    image_plot = ImagePlot(image_id, image_base_path)
    other_image_ids = [node["other_image_id"] for node in distances_json]
    image_plot.plot_comparison(other_image_ids)

