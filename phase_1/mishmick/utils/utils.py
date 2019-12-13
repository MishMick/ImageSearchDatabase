import cv2
import numpy as np
import re
import scipy.stats
import pickle
import pandas as pd
import glob
import os
import json
from mishmick.config.config_provider import ConfigProvider

config_path = "/Users/mish/ASU/Sem_1/MWDB/phase_1/phase_1/mishmick/config.json"
config_provider = ConfigProvider(config_path)
base_path = config_provider.get_image_base_path()
storage_path = config_provider.get_storage_path()
output_path = config_provider.get_output_path()
image_type = config_provider.get_image_type()

_image_paths = [_image_path for _image_path in glob.glob(base_path + "/*" + image_type)]

cm_feature_vectors = {}
sift_feature_vectors = {}


def get_image_id_from_path(_image_path):
    return re.findall(r'\d+', str(_image_path).split("/HandsT/")[1].split(".jpg")[0])[0]


def get_yuv_for_image(_image_path):
    _image = cv2.imread(_image_path)  # read image from path
    _image_yuv = cv2.cvtColor(_image, cv2.COLOR_BGR2YUV)  # get yuv channels for image
    return _image_yuv


def get_greyscale_for_image(_image_path):
    _image = cv2.imread(_image_path)  # read image from path
    _image_grayscale = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)  # get grayscale channels for image
    return _image_grayscale


def get_image_windows(_image):
    _windows = []
    for i in range(0, _image.shape[0], 100):
        for j in range(0, _image.shape[1], 100):
            _windows.append(_image[i:(i + 100), j:(j + 100)])
    return np.array(_windows)


# Calculate color moments for image windows


def get_color_moments_for_image_window(_image_window_yuv):
    image_window_y, image_window_u, image_window_v = cv2.split(_image_window_yuv)
    [[np.mean(channel), np.math.sqrt(np.var(channel)), scipy.stats.skew(channel.ravel())] for channel in
     [image_window_y, image_window_u, image_window_v]]
    image_window_color_moments = np.ravel(
        [[np.mean(channel), np.math.sqrt(np.var(channel)), scipy.stats.skew(channel.ravel())] for channel in
         [image_window_y, image_window_u, image_window_v]])
    return image_window_color_moments


# Calculate aggregate color moment for the image


def get_color_moment(_image_path):
    # Convert image to yuv
    _image_yuv = get_yuv_for_image(_image_path)
    image_windows_yuv = get_image_windows(_image_yuv)
    _cm_image = np.ravel(
        [get_color_moments_for_image_window(image_window_yuv) for image_window_yuv in image_windows_yuv]
    )
    return _cm_image


def print_features(_model_type, _image_id):
    # CM
    print("Extracting color moment features from image : ", _image_id)
    features = {}
    if _model_type == 'cm':
        features = read_pickle('color_moments.pck')
        print(features[str(_image_id).zfill(7)])
    elif _model_type == 'sift':
        features = read_pickle('sift_features.pck')
        print([f for f in features[str(_image_id).zfill(7)]])
    else:
        print('Invalid option specified')


def extract_color_moments():
    # CM
    for _image_path in _image_paths:
        print("Extracting color moment features from image : ", _image_path)
        _image_id = get_image_id_from_path(_image_path)
        cm_from_image = get_color_moment(_image_path)
        cm_feature_vectors[_image_id] = cm_from_image.tolist()
    return cm_feature_vectors


def store_color_moments(_features_file):
    cm_features = extract_color_moments()
    db_file = storage_path + _features_file
    with open(db_file, 'wb') as f:
        pickle.dump(cm_features, f)
    print("Color moment Features stored in " + db_file)


# Feature extractor
def extract_sift_features(_image_path, vector_size=32):
    print("Extracting SIFT features from image : ", _image_path)
    try:
        # Using SIFT
        sift = cv2.xfeatures2d.SIFT_create()
        # Convert RBG to greyscale
        _image_hand = get_greyscale_for_image(_image_path)
        # Finding image keypoints
        kps = sift.detect(_image_hand)
        # Getting first 32 of them.
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = sift.compute(_image_hand, kps)
        # Flatten all of them in one big vector - our feature vector
        dsc = dsc.flatten()
        # Making descriptor of same size - 128 is arbitrary value
        needed_size = (vector_size * 128)
        if dsc.size < needed_size:
            # if we have less the 128 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    except cv2.error as e:
        print('Error: ', e)
        return None

    return dsc


def store_sift_features(_features_file):
    # SIFT
    for _image_path in _image_paths:
        _image_id = get_image_id_from_path(_image_path)  # extract image id from given image path
        sift_feature_vectors[_image_id] = extract_sift_features(_image_path)

    print("Storing SIFT features")
    db_file = storage_path + _features_file
    with open(db_file, 'wb') as f:
        pickle.dump(sift_feature_vectors, f)
    print("SIFT Features stored in " + db_file)


def read_pickle(_features_file):
    db_file = storage_path + _features_file
    df = pd.read_pickle(db_file)
    return df


def save_plot(plt, image_id, query_info):
    print("here")
    output_directory = output_path + "/query_image_" + str(image_id)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_image_path = output_directory + "/output.png"
    output_info_path = output_directory + "/query.json"
    plt.savefig(output_image_path)
    with open(output_info_path, 'w') as f:
        json.dump(str(query_info), f)
