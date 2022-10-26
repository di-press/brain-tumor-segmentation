import numpy as np

def erosion(image, structuring_element_size: int=3):
    assert structuring_element_size % 2 != 0
    structuring_element_radius = structuring_element_size // 2
    padded_image = np.pad(image.copy(), (structuring_element_radius, structuring_element_radius))
    eroded_image = np.zeros_like(padded_image)
    rows, columns = eroded_image.shape
    for i in range(structuring_element_radius, rows - structuring_element_radius):
        for j in range(structuring_element_radius, columns - structuring_element_radius):
            row_range = (i - structuring_element_radius, i + structuring_element_radius + 1)
            column_range = (j - structuring_element_radius, j + structuring_element_radius + 1)
            eroded_image[i, j] = min(padded_image[row_range[0]:row_range[1], column_range[0]:column_range[1]].flatten())
    return eroded_image[structuring_element_radius:rows - structuring_element_radius, structuring_element_radius:columns - structuring_element_radius]

def dilation(image, structuring_element_size: int=3):
    assert structuring_element_size % 2 != 0
    structuring_element_radius = structuring_element_size // 2
    padded_image = np.pad(image, (structuring_element_radius, structuring_element_radius))
    eroded_image = np.zeros_like(padded_image)
    rows, columns = eroded_image.shape
    for i in range(structuring_element_radius, rows - structuring_element_radius):
        for j in range(structuring_element_radius, columns - structuring_element_radius):
            row_range = (i - structuring_element_radius, i + structuring_element_radius + 1)
            column_range = (j - structuring_element_radius, j + structuring_element_radius + 1)
            eroded_image[i, j] = max(padded_image[row_range[0]:row_range[1], column_range[0]:column_range[1]].flatten())
    return eroded_image[structuring_element_radius:rows - structuring_element_radius, structuring_element_radius:columns - structuring_element_radius]

def opening(image, structuring_element_size: int=3):
	return dilation(erosion(image, structuring_element_size), structuring_element_size)

def closing(image, structuring_element_size: int=3):
	return erosion(dilation(image, structuring_element_size), structuring_element_size)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    import skimage
    from skimage.data import binary_blobs
    from skimage.morphology import binary_dilation, binary_erosion, binary_opening, binary_closing, square

    functions = (
        (dilation, binary_dilation),
        (erosion, binary_erosion),
        (opening, binary_opening),
        (closing, binary_closing)
    )
    source_data = skimage.img_as_float(binary_blobs(length=256, seed=42))
    data = np.zeros_like(source_data, dtype=np.uint8)
    data[source_data > 0.5] = 1

    for custom_function, skimage_function in functions:
        skimage_result = skimage_function(data, footprint=square(15))
        result = custom_function(data, structuring_element_size=15)

        # Plot results
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3.2),
                                            sharex=True, sharey=True)
        fig.suptitle(f"{custom_function.__name__.capitalize()}")
        ax1.imshow(skimage_result, cmap="gray")
        ax1.axis('off')
        ax1.set_title("Skimage Result")
        ax2.imshow(result, cmap="gray")
        ax2.axis("off")
        ax2.set_title("Implementation Result")
        diff = np.abs(skimage_result - result)
        ax3.imshow(diff, cmap="gray")
        ax3.axis("off")
        ax3.set_title("Difference")

        fig.tight_layout()
        plt.show()