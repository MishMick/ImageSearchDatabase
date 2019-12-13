#import matplotlib for ploting and converting image to numpy arrays
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class ImagePlot:

    def __init__(self, image_id, image_base_path):
        self.image_id = image_id
        self.image_base_path = image_base_path
    
    
    #comparing the plot and displaying the images
    def plot_comparison(self, other_image_ids):
        
        #length of other images ids
        l = len(other_image_ids)
        
        #creating the figure of [width, height] in inches
        fig = plt.figure(figsize=(l * 4, 4))
        columns = l + 1
        rows = 1

        image_ids = [self.image_id] + other_image_ids

        for i in range(1, columns * rows + 1):
            #zfill - pad the 0s in image file name 
            image_file_name = self.image_base_path + "Hand_" + str(image_ids[i - 1]).zfill(7) + ".jpg"
            #reading the image
            image = mpimg.imread(image_file_name)
            #adding the subplot
            fig.add_subplot(rows, columns, i)
            plt.imshow(image)
        
        return plt

