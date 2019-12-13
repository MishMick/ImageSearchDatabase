from consolemenu import ConsoleMenu
from consolemenu.items import FunctionItem

from arvind_hasti.driver import *

menu = ConsoleMenu("CSE515 Phase 1 Console")


def do_extract_cm():
    do_cm_create(image_factory, cm_vector_mapping, feature_storage)


def do_extract_lbp():
    do_lbp_create(image_factory, lbp_vector_mapping, feature_storage)


def do_extract_hog():
    do_hog_create(image_factory, hog_vector_mapping, feature_storage)


def do_load_df():
    model_type = int(input("Select from [1->cm, 2->lbp]"))
    image_id = int(input("Image ID: "))
    if model_type == 1:
        print("Selected: " + "CM")
        df = do_df_load(ModelType.COLOR_MOMENTS, feature_storage)
        features = [f for f in df.loc[image_id, :]]
        
    else:
        print("Selected: " + "LBP")
        df = do_df_load(ModelType.LBP, feature_storage)
        features = [f for f in df.loc[image_id, :]]
    print("Feature shape: " + str(len(features)))
    print(features)

    with open('./load_df.out', "w") as f:
        f.write("Feature shape: ")
        f.write(str(len(features)))
        f.write("\n")
        features_str = ' '.join(str(feature) for feature in features)
        f.write(features_str)

    
def do_get_distances():
    #distance_measure = input("Select from [l2, cosine]: ")
    model_type = int(input("Select from [1->cm, 2->lbp]: "))
    image_id = int(input("Image ID: "))
    k = int(input("K: "))
    
    #similar_images = do_distances_get(model_type, image_id, k, feature_storage)
    if model_type == 1:
        print("Selected: " + "CM")
        similar_images = do_distances_get(ModelType.COLOR_MOMENTS, image_id, k, feature_storage)
    else:
        print("Selected: " + "LBP")
        similar_images = do_distances_get(ModelType.LBP, image_id, k, feature_storage)

    print(similar_images)
    plt = do_plot(image_id, similar_images)
    save_plot(plt, image_id, {"distance_measure": "cosine", "model_type": model_type, "query_image_id": image_id, "k": k, "similar_images": similar_images})
    plt.show()


extract_cm_item = FunctionItem("Extract Color Moments Features", do_extract_cm)
extract_lbp_item = FunctionItem("Extract LBP Features", do_extract_lbp)
extract_hog_item = FunctionItem("Extract HOG Features", do_extract_hog)
load_df_item = FunctionItem("Load DF", do_load_df)
get_distances_item = FunctionItem("Get similar images", do_get_distances)

menu.append_item(extract_cm_item)
menu.append_item(extract_lbp_item)
#menu.append_item(extract_hog_item)
menu.append_item(load_df_item)
menu.append_item(get_distances_item)

menu.show()
