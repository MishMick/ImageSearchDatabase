from consolemenu import ConsoleMenu
from consolemenu.items import FunctionItem

from sandeepkunichi.driver import *

menu = ConsoleMenu("CSE515 Phase 1 Console")


def do_extract_cm():
    do_cm_create(image_factory, cm_vector_mapping, feature_storage)


def do_extract_lbp():
    do_lbp_create(image_factory, lbp_vector_mapping, feature_storage)


def do_extract_hog():
    do_hog_create(image_factory, hog_vector_mapping, feature_storage)


def do_extract_sift():
    do_sift_create(image_factory, sift_vector_mapping, feature_storage)


def do_load_df():
    model_type = raw_input("Select from [cm, lbp, hog, sift]: ")
    image_id = int(raw_input("Image ID: "))
    print "Selected: " + model_type
    df = do_df_load(model_type, feature_storage)
    features = [f for f in df.loc[image_id, :]]
    print "Feature shape: " + str(len(features))
    print features


def do_get_distances():
    distance_measure = raw_input("Select from [l2, cosine]: ")
    model_type = raw_input("Select from [cm, lbp, hog, sift]: ")
    image_id = int(raw_input("Image ID: "))
    k = int(raw_input("K: "))
    print "Selected: " + model_type
    similar_images = do_distances_get(distance_measure, model_type, image_id, k, feature_storage)
    print similar_images
    plt = do_plot(image_id, similar_images)
    save_plot(plt, image_id, {"distance_measure": distance_measure, "model_type": model_type, "query_image_id": image_id, "k": k, "similar_images": similar_images})
    plt.show()


extract_cm_item = FunctionItem("Extract Color Moments Features", do_extract_cm)
extract_lbp_item = FunctionItem("Extract LBP Features", do_extract_lbp)
extract_hog_item = FunctionItem("Extract HOG Features", do_extract_hog)
extract_sift_item = FunctionItem("Extract SIFT Features", do_extract_sift)
load_df_item = FunctionItem("Load DF", do_load_df)
get_distances_item = FunctionItem("Get similar images", do_get_distances)

menu.append_item(extract_cm_item)
menu.append_item(extract_lbp_item)
menu.append_item(extract_hog_item)
menu.append_item(extract_sift_item)
menu.append_item(load_df_item)
menu.append_item(get_distances_item)

menu.show()
