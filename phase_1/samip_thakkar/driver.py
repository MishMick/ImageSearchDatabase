#import libraries: tqdm for progress bar
from tqdm import tqdm
import numpy as np
import os
import json
#import all the files in project
from similarityfunction import SimilarityFunction
from imagefactory import ImageFactory
from colormoments import ColorMoments
from config_provider import ConfigProvider
from histogram import Histogram
from imageplot import ImagePlot
from featurestore import FeatureStorage, ModelType
from MyEncoder import MyEncoder
config_path = "C:/ASU/Sem 1/MWDB/Project/Phase1/config.json"

config_provider = ConfigProvider(config_path)
image_base_path = config_provider.get_image_base_path()
storage_path = config_provider.get_storage_path()
output_path = config_provider.get_output_path()

#creating the objects
image_factory = ImageFactory(image_base_path)
colormoments = ColorMoments()
histogram = Histogram()
feature_storage = FeatureStorage(storage_path)  

#setting the path of the image
image_path = image_factory.get_image_path_list()

def do_cm_task(image_path):
    image_id = image_factory.get_image_id_from_path(image_path)
    cm_from_image = colormoments.calculate_color_moments_from_image(image_path)
    return image_id, cm_from_image.tolist()


def do_hog_task(image_path):
    image_id = image_factory.get_image_id_from_path(image_path)
    hog_from_image = histogram.get_hog_from_image(image_path)
    return image_id, hog_from_image.tolist()

#creating the color moments features
def do_cm_create(image_factory, colormoments, feature_storage):
    # Color Moments
    print("Creating Color Moments features:")
    cm_feature_vectors = {}
    for image_path in tqdm(image_factory.get_image_path_list()):
        image_id = image_factory.get_image_id_from_path(image_path)
        cm_from_image = colormoments.calculate_color_moments_from_image(image_path)
        cm_feature_vectors[image_id] = cm_from_image.tolist()

    feature_storage.clear_model_storage(ModelType.COLOR_MOMENTS)
    feature_storage.store_feature_set(ModelType.COLOR_MOMENTS, cm_feature_vectors)
    print("Features stored in " + storage_path + "color_moments.json")

#creating the histogram features
def do_hog_create(image_factory, histogram, feature_storage):
    # Histogram
    print("Creating HOG features:")
    hog_feature_vectors = {}
    for image_path in tqdm(image_factory.get_image_path_list()):
        image_id = image_factory.get_image_id_from_path(image_path)
        hog_from_image = histogram.calculate_histogram(image_path)
        hog_feature_vectors[image_id] = hog_from_image.tolist()

    feature_storage.clear_model_storage(ModelType.HOG)
    feature_storage.store_feature_set(ModelType.HOG, hog_feature_vectors)
    print("Features stored in " + storage_path + "hog.json")

#loading to dataframe
def do_df_load(model_type, feature_storage):
    return feature_storage.load_to_df(model_type)

#calculating the distances 
def do_distances_get(model_type, image_id, k, feature_storage):
    df = feature_storage.load_to_df(model_type)
    print(np.shape(df))
    similarity_function = SimilarityFunction(df, model_type)
    distances_json = similarity_function.get_image_distances(image_id)[:k]
    return distances_json

#plotting the image
def do_plot(image_id, distances_json):
    image_plot = ImagePlot(image_id, image_base_path)
    other_image_ids = [node["other_image_id"] for node in distances_json]
    return image_plot.plot_comparison(other_image_ids)
    
def save_plot(plt, image_id, query_info):
    output_directory = output_path + "/query_image_" + str(image_id)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_image_path = output_directory + "/output.jpg"
    output_info_path = output_directory + "/query.json"
    plt.savefig(output_image_path)
    with open(output_info_path, 'w') as f:
        json.dump(query_info, f, cls=MyEncoder)