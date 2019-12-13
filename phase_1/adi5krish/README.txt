In this project, i have implemented three tasks.

1. Extract Features for a given folder of images
2. Retrieve Features for a given imageID and feature model
3. Visualize K similar Images for a given imageID, feature model and K

The feature models i have implemented are 

1. Color Moments(Mean, Standard Deviation and Skewness)
2. Local Binary Patterns

Have run the model on the Sample Dataset provided by the Professor as part of the project.
Note : Due to the processing limitations of my laptop, extracting features for larger data sets takes more
time.

To run the code, you have to install some python libraries and the list is highlighted in the requirements
section of the project and also in the "requirements.txt" file in the code folder.

Python version (preferably 3.5 or higher)
    
opencv-python (For reading Images)
 
scikit-image (for computing local binary patterns)
   
json and pickle libraries (for reading and writing data to  a file)
Note : These comes in-built with python.
    
numpy, scipy, matplotlib (for manipulating lists, dicts and for displaying images)

The python version i am using is Python 3.6.8 and i recommend the users to have python versions 3.5 and higher
as some of the functionalities in scikit-learn and scikit-image have been deprecated.

Note that before running any of the commands install all the libaries and recommended python version and 
change the "image_base_path" in driver.py file to the path where the data set is stored in your local machine.
Also change the "storage_path" in driver.py file to the path where you want to store the created json and 
pickle files.

Now, open the command prompt and change the path to the code folder in your terminal
Run the command: python menu.py
Firstly, must enter the first option atleast once which does the feature extraction for all the images and stores them
as json and pickle files. Now can enter any of the options which retrieves, prints feature vectors and visualizes
k similar images in task 3.
The "menu.py" file contains the command line interface (CLI) code and provides you with three choices as mentioned above
in the tasks section. If you enter any choice other than the three, you will come out of the loop.

The code is properly commented for users to understand.





