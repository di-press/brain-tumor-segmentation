import numpy as np

def bilateral_filter(image, kernel_size, spatial_sigma, intensity_sigma):
    kernel_radius = kernel_size // 2
    image = image.astype(np.float32)
    pad_image = np.pad(image, pad_width=kernel_radius, mode="constant")
    processed_image = np.zeros_like(pad_image)
    rows, columns = image.shape

    for i in range(kernel_radius, rows+kernel_radius):
        for j in range(kernel_radius, columns+kernel_radius):
            weighted_pixels = 0.0
            weights_sum = 0.0
            current_pixel = pad_image[i, j]

            for m in range(-kernel_radius, kernel_radius+1):
                for n in range(-kernel_radius, kernel_radius+1):
                    shifted_pixel = pad_image[i + m, j + n]
                    spatial_weight = np.exp(-0.5 * (m**2 + n**2) / (spatial_sigma**2))
                    intensity_weight = np.exp(-0.5 * (current_pixel - shifted_pixel)**2 / (intensity_sigma**2))
                    weight = spatial_weight * intensity_weight
                    weighted_pixels += weight * shifted_pixel
                    weights_sum += weight

            processed_image[i, j] = weighted_pixels / weights_sum
    return processed_image[kernel_radius:rows+kernel_radius, kernel_radius:columns+kernel_radius]

if __name__ == "__main__":
    from skimage import io
    import matplotlib.pyplot as plt
    from skimage.restoration import denoise_bilateral

    image = io.imread("dataset/index.jpeg", as_gray=True).astype(np.float32)
    answer = bilateral_filter(image, kernel_size=5, spatial_sigma=1, intensity_sigma=image.std())
    skimage_answer = denoise_bilateral(image, win_size=5, sigma_color=image.std())
    _, axes = plt.subplots(nrows=1, ncols=3)
    axes[0].imshow(image, cmap="gray")
    axes[1].imshow(skimage_answer, cmap="gray")
    axes[2].imshow(answer, cmap="gray")
    plt.show()
