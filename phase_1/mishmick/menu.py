from consolemenu import ConsoleMenu

from mishmick.driver import *

menu = ConsoleMenu("CSE515 Phase 1 Console")
ch = 0


def do_extract_cm():
    image_id = int(input("Image ID: "))
    k = int(input("K: "))
    do_cm(image_id, k)


def do_extract_sift():
    image_id = int(input("Image ID: "))
    k = int(input("K: "))
    do_sift(image_id, k)


print(" 1. Extract features \n", "2. Extract Features for a given Image ID \n", "3. Get similar images \n", "4. Exit \n")
ch = int(input("Enter your choice: "))

# This is the while loop that prints the options we have until we enter an invalid output
while ch < 5:
    if ch == 1:
        model_type = input("Select model from [cm, sift]: ")
        if model_type == 'cm':
            store_color_moments("color_moments.pck")
        elif model_type == 'sift':
            store_sift_features("sift_features.pck")
    if ch == 2:
        _image_id = input("Select image id ")
        model_type = input("Select model from [cm, sift]: ")
        print_features(model_type, _image_id)
    elif ch == 3:
        model_type = input("Select from [cm, sift]: ")
        if model_type == 'cm':
            do_extract_cm()
        elif model_type == 'sift':
            do_extract_sift()
    elif ch == 4:
        exit(0)
    else:
        break

    print(" 1. Extract features \n", "2. Extract Features for a given Image ID \n",
          "3. Get similar images \n",  "4. Exit \n")
    ch = int(input("Enter your choice: "))



