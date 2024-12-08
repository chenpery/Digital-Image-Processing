import os
import cv2
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

import skimage


def create_folders():
    if not os.path.exists("trajectories_plots"):
        os.mkdir("trajectories_plots")
    if not os.path.exists("discrete_point_spread_functions"):
        os.mkdir("discrete_point_spread_functions")
    if not os.path.exists("blurred_frames"):
        os.mkdir("blurred_frames")
    if not os.path.exists("deblurred_images"):
        os.mkdir("deblurred_images")
    if not os.path.exists("statistics"):
        os.mkdir("statistics")


def generate_trajectories_plots(mat):
    for i, (x, y) in enumerate(zip(mat['X'], mat['Y'])):
        plt.scatter(x, y)
        plt.savefig(os.path.join("trajectories_plots", f"trajectory_{i}.png"))
        plt.close()


def generate_psf_plots(mat):
    psfs_collection = []
    size = 2 * round(max(mat['X'].max(), mat['Y'].max())) + 1
    for i, (x, y) in enumerate(zip(mat['X'], mat['Y'])):
        psf_array = np.zeros((size, size))
        center = (size - 1) / 2
        for cx, cy in zip((x + center).astype(int), (y + center).astype(int)):
            if cx < size and cy < size:
                psf_array[cx, cy] += 1 / 1000
        psfs_collection.append(psf_array)
        plt.imshow(psf_array, cmap='gray')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(os.path.join("discrete_point_spread_functions", f'psf_{i}.png'))
        plt.close()
    return psfs_collection


def generate_blurred_images(psfs_collection, image):
    blurred_images = []
    for i, psf in enumerate(psfs_collection):
        blurred = convolve2d(in1=image, in2=psf, mode='same')
        blurred_images.append(blurred)
        plt.imshow(blurred, cmap='gray')
        plt.savefig(f'blurred_frames/blurred_{i}.png')
        plt.close()
    return blurred_images


def deblur_single_image_using_k_frames(k_blurred_images):
    k_fouriered = [scipy.fftpack.fftn(img) for img in k_blurred_images]
    k_fouriered_abs = [np.abs(f) for f in k_fouriered]

    # calculate weight for each image depending on the Fourier spectrum magnitude.
    k_fouriered_sum = np.sum(k_fouriered_abs, axis=0)
    weighted_each_image = [fouriere / k_fouriered_sum for fouriere in k_fouriered_abs]

    omega = np.fft.fftfreq(n=k_fouriered_sum.size)
    sinc_res = np.sinc(omega).reshape(k_fouriered_sum.shape)

    fourier_approx = np.sum(np.array(k_fouriered / sinc_res) * weighted_each_image, axis=0)
    deblurr_img = np.abs(np.fft.ifftn(fourier_approx))
    plt.imshow(deblurr_img, cmap='gray')
    plt.savefig(f"deblurred_images/deblurred_using_{len(k_blurred_images)}_frames")
    plt.close()
    return deblurr_img


def debblur_handler(blurred_images):
    deblurr_images = []

    for k in range(len(blurred_images)):
        deblurr_images.append(deblur_single_image_using_k_frames(blurred_images[:k + 1]))

    return deblurr_images


def plot_statistics(deblurred_images, original_image):
    psnr_values = []
    for deblurred_image in deblurred_images:
        mse = ((original_image - deblurred_image) ** 2).mean()
        psnr_values.append(10 * np.log10(255 ** 2 / mse))

        # psnr_values.append(skimage.metrics.peak_signal_noise_ratio(deblurred_image, original_image))
    plt.title("k debluer quality comparison")
    plt.scatter(range(1, len(psnr_values) + 1), psnr_values)
    plt.plot(psnr_values)
    plt.xlabel('#frames - K - ')
    plt.ylabel('PSNR Value')
    plt.savefig("statistics/psnr_k-deblured_graph.png")
    plt.close()

    df = pd.DataFrame({'PSNR': psnr_values}, index=range(1, len(psnr_values) + 1))
    df.index.name = 'k'
    df.to_csv(f"statistics/K_{len(psnr_values)}_psnr_vals.csv")


def input_parser():
    # Get the current directory where the Python file is located
    current_directory = os.path.dirname(os.path.abspath(__file__))
    # Define the path to the resources folder
    resources_folder = os.path.join(current_directory, "resources")
    # Check if the resources folder exists
    if os.path.exists(resources_folder) and os.path.isdir(resources_folder):
        original_image = cv2.imread(os.path.join('resources', 'DIPSourceHW1.jpg'), cv2.IMREAD_GRAYSCALE)
        mat = scipy.io.loadmat(os.path.join('resources', '100_motion_paths.mat'))
    else:
        original_image = cv2.imread('DIPSourceHW1.jpg', cv2.IMREAD_GRAYSCALE)
        mat = scipy.io.loadmat('100_motion_paths.mat')

    return original_image, mat


if __name__ == '__main__':
    print("Part 1...")
    create_folders()
    original_image, mat = input_parser()
    generate_trajectories_plots(mat)
    print("trajectories plotted.. ")
    psfs_collection = generate_psf_plots(mat)
    print("psf plotted.. ")
    blurred_images = generate_blurred_images(psfs_collection, original_image)
    print("blurred images plotted.. ")
    print("Part 1 files have been successfully generated!")
    print("Part 2...")
    print("deblurring images.. ")
    deblurred_images = debblur_handler(blurred_images)
    print("calc statistics.. ")
    plot_statistics(deblurred_images, original_image)
    print("Finished!!!")
