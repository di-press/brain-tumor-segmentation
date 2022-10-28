import numpy as np
from numpy.random import default_rng
import scipy.spatial.distance

def kmeans_segmentation(image, clusters: int=2, max_iter: int=100, tolerance: float=1e-3):
    x_indices, y_indices = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing='ij')
    image_data = np.concatenate((image.reshape(-1, 1), x_indices.reshape(-1, 1), y_indices.reshape(-1, 1)), axis=1)
    clustering = kmeans_clustering(image_data, clusters, max_iter, tolerance)[:, 0]
    return clustering.reshape(image.shape)


def kmeans_clustering(image_data, clusters: int=2, max_iter: int=100, tolerance: float=1e-3):
    # Initialize centroids with randomly selected clusters
    indices = np.array(range(image_data.shape[0]), dtype=np.int32)
    rng = default_rng(seed=42)
    rng.shuffle(indices)
    
    centroids = np.array(image_data[indices[:clusters]])
    distances = scipy.spatial.distance.pdist(centroids)

    # If there are too many pixels, the pdist will consume too much
    # RAM (e.g. more than 12 GB). If there are too many pixels, to avoid
    # such aggressive consume of memory, it's better to take a random sample
    # of the pixels.
    number_of_samples = min(image_data.shape[0], 20000)
    sampling_indices = np.random.choice(image_data.shape[0], number_of_samples, replace=False)
    data_distances = scipy.spatial.distance.pdist(image_data[sampling_indices, :])

    # If the initial centroids are too close to each other, resample
    centroid_distance_threshold = np.percentile(data_distances, 10)
    while np.any(distances < centroid_distance_threshold):
        rng.shuffle(indices)
        centroids = np.array(image_data[indices[:clusters]])
        distances = scipy.spatial.distance.pdist(centroids)
    
    cluster_classes = [[] for _ in range(clusters)]
    for _ in range(max_iter):
        # Find the indices of the closest centroids
        pairwise_distances = scipy.spatial.distance.cdist(image_data, centroids)
        cluster_indices = np.argsort(pairwise_distances)[:, 0]
        for sample, index in zip(range(image_data.shape[0]), cluster_indices):
            cluster_classes[index].append(sample)
                
        previous_centroids = centroids.copy()
        for cluster_class, cluster_samples in enumerate(cluster_classes):
            centroids[cluster_class] = np.mean(image_data[cluster_samples], axis=0)

        # If all the centroids stop moving (i.e. their distance falls bellow a tolerance threshold),
        # assume convergence and stop the algorithm.
        # This distance is computed as a relative measure.
        norms = np.linalg.norm(((centroids - previous_centroids) / previous_centroids) * 100.0, axis=1)
        if np.all(norms < tolerance):
            break
    
    clustering = np.zeros_like(image_data)
    for cluster_class, cluster_samples in enumerate(cluster_classes):
        clustering[cluster_samples] = cluster_class
    
    return clustering

if __name__ == "__main__":
    data = np.array([
        [1, 1, 1],
        [3, 3, 3],
        [7, 7, 7],
        [111, 111, 111],
        [117, 117, 117],
        [131, 131, 131],
        [283, 283, 283],
        [391, 391, 391],
        [317, 317, 317],
    ], dtype=np.float32)

    custom_result = kmeans_clustering(data, clusters=3)
    colors = []
    for val in custom_result[:, 0]:
        if val == 0:
            colors.append("red")
        elif val == 1:
            colors.append("green")
        else:
            colors.append("blue")
    
    import matplotlib.pyplot as plt
    plt.scatter(data[:, 0], data[:, 1], c=colors)
    plt.show()
    import matplotlib.pyplot as plt
    from skimage.data import binary_blobs
    from skimage.segmentation import slic
    import skimage
    from skimage import data, color
    from skimage.transform import resize
    image = color.rgb2gray(data.coffee())
    img_data = resize(image, (image.shape[0] // 4, image.shape[1] // 4),
                       anti_aliasing=True)
    result = kmeans_segmentation(img_data, clusters=10)
    segments = slic(img_data, n_segments=10, compactness=10)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3.2),
                                            sharex=True, sharey=True)
    ax1.imshow(img_data, cmap="gray")
    ax1.axis('off')
    ax1.set_title("Input")
    ax2.imshow(result, cmap="gray")
    ax2.axis("off")
    ax2.set_title("Implementation Result")
    ax3.imshow(segments, cmap="gray")
    ax3.axis("off")
    ax3.set_title("SLIC")

    fig.tight_layout()
    plt.show()