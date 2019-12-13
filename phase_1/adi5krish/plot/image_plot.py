import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class ImagePlot:

    def __init__(self, image_id, image_base_path):
        self.image_id = image_id
        self.image_base_path = image_base_path

    def plot_comparison(self, other_image_ids):  # Method that plots the similar images for given image and K
        k = len(other_image_ids)

        fig = plt.figure(figsize=(k * 2, 1))
        columns = k + 1
        rows = 1

        image_ids = [self.image_id] + other_image_ids

        for i in range(1, columns * rows + 1):
            image_file_name = self.image_base_path + "\Hand_" + str(image_ids[i - 1]).zfill(7) + ".jpg"  # From the imageIDs this gets the path of the image
            img = mpimg.imread(image_file_name) # Displays  the image
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)

        plt.show()

