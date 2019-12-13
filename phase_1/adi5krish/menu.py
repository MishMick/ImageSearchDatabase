from consolemenu import ConsoleMenu
from consolemenu.items import FunctionItem

from driver import *

ch = 0

def do_extract_cm():  # this method calls the method for extracting color moments
    do_cm_create(image_factory, cm_vector_mapping, feature_storage)


def do_extract_lbp(): # this method calls the method for extracting local binary patterns
    do_lbp_create(image_factory, lbp_vector_mapping, feature_storage)

def do_load_df():
	model_type = input("Select from [cm, lbp]: ")  # This method takes inputs as imageID , feature model and returns the corresponding feature descriptor
	image_id = input("Image ID: ")
	print("The model type you selected: " + model_type)
	df = do_df_load(model_type, feature_storage)
	features = df[image_id]
	print("Shape of the Features: "+str(len(features)))
	print(features)  # Depends on the model and imageID, this prints the features of the image

def do_get_distances():
    model_type = input("Select from [cm, lbp]: ")  # Select model type 
    image_id = input("Image ID: ") # input the image ID for which the similar images should be extracted
    k = int(input("K: ")) # Input the number 'K' of similar images
    print("The model type you selected: " + model_type)
    similar_image_distances = do_distances_get(model_type, image_id, k, feature_storage) # Get nearest 'K' distances for the image
    print(similar_image_distances) # prints ID's of the images similar to the query image along with the similarity scores
    do_plot(image_id, similar_image_distances) # plotting similar images 

print(" 1. Extract features from a folder \n", "2. Retrieve Features for a given Image ID \n", "3. Get similar images \n")
ch = int(input("Enter your choice: "))

# This is the while loop that prints the options we have until we enter an invalid output
while(ch<4 and ch>0):
	
	if(ch == 1):
		model_type = input("Select from [cm, lbp]: ") # Select an option between cm and lbp
		if(model_type == 'cm'):
			do_extract_cm()
		elif(model_type == 'lbp'):
			do_extract_lbp()

	elif(ch == 2):
		
		do_load_df()

	elif(ch == 3):
		do_get_distances()
	else:
		break

	print(" 1. Extract features from a folder \n" , "2. Extract Features for a given Image ID \n", "3. Get similar images \n")
	ch = int(input("Enter your choice: "))
