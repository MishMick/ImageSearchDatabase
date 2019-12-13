from mishmick.similarity.similarity import SimilarityFunction
from mishmick.plot.image_plot import ImagePlot
from mishmick.utils.utils import *


def do_sift(_image_id, k, _feature_file="sift_features.pck"):
    # Store SIFT features in pickle
    store_sift_features(_feature_file)
    # Read SIFT features from pickle
    df = read_pickle(_feature_file)
    # Initialise matcher
    sf = SimilarityFunction(df)
    # Plot comparisons
    image_plot = ImagePlot(_image_id, base_path)
    distances_json = sf.get_sift_distances(_image_id)[:k]
    plt = image_plot.plot_comparison([node["other_image_id"] for node in distances_json])
    save_plot(plt, _image_id, {"query_image_id": _image_id, "k": k, "similar_images": distances_json})
    # Show plot
    plt.show()


def do_cm(_image_id, k, _feature_file="color_moments.pck"):
    # Store color moments in pickle
    store_color_moments(_feature_file)
    # Read color moments from pickle
    df = read_pickle(_feature_file)
    # Initialise matcher
    sf = SimilarityFunction(df)
    # Plot comparisons
    image_plot = ImagePlot(_image_id, base_path)
    distances_json = sf.get_cm_distances(_image_id)[:k]
    plt = image_plot.plot_comparison([node["other_image_id"] for node in distances_json])
    save_plot(plt, _image_id, {"query_image_id": _image_id, "k": k, "similar_images": distances_json})
    # Show plot
    plt.show()

