#import console menu
from consolemenu import ConsoleMenu
from consolemenu.items import FunctionItem

#import driver file
from driver import *

menu = ConsoleMenu("MWDB Project Phase 1 Console")

#extract colormoments
def do_extract_cm():
    do_cm_create(image_factory, colormoments, feature_storage)

#extract histogram
def do_extract_hog():
    do_hog_create(image_factory, histogram, feature_storage)

#load dataframe
def do_load_df():
    model_type = input("Select from [cm,  hog]")
    image_id = int(input("Image ID: "))
    print("Selected: " + model_type)
    df = do_df_load(model_type, feature_storage)
    features = [f for f in df.loc[image_id, :]]
    print("Feature shape: " + str(len(features)))
    print(features)
#get the distance
def do_get_distances():
    model_type = input("Select from [cm, hog]: ")
    image_id = int(input("Image ID: "))
    k = int(input("K: "))
    print ("Selected: " + model_type)
    similar_images = do_distances_get(model_type, image_id, k, feature_storage)
    print(similar_images)
    plt = do_plot(image_id, similar_images)
    save_plot(plt, image_id, {"model_type": model_type, "query_image_id": image_id, "k": k, "similar_images": similar_images})
    plt.show()

extract_cm_item = FunctionItem("Extract Color Moments Features", do_extract_cm)
extract_hog_item = FunctionItem("Extract HOG Features", do_extract_hog)
load_df_item = FunctionItem("View features", do_load_df)
get_distances_item = FunctionItem("Get similar images", do_get_distances)

menu.append_item(extract_cm_item)
menu.append_item(extract_hog_item)
menu.append_item(load_df_item)
menu.append_item(get_distances_item)

menu.show()
