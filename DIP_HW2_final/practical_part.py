import os
import cv2
import numpy as np
import scipy
from sklearn.neighbors import BallTree
from scipy.signal import convolve2d
from scipy import signal, fftpack
from tqdm import tqdm


def generate_gaussian_filter(filter_size):
    range = filter_size / 16
    z = np.linspace(-range, range, filter_size)
    x, y = np.meshgrid(z, z)
    d = np.sqrt(x * x + y * y)
    g = np.exp(-((d) ** 2 / (2.0)))
    g = g / g.sum()
    return g


def generate_sinc_filter(filter_size):
    range = filter_size / 4
    x = np.linspace(-range, range, filter_size)
    xx = np.outer(x, x)
    s = np.sinc(xx)
    s = s / s.sum()
    return s


def create_folders():
    if not os.path.exists("results"):
        os.mkdir("results")


def input_parser():
    # Get the current directory where the Python file is located
    current_directory = os.path.dirname(os.path.abspath(__file__))
    # Define the path to the resources folder
    resources_folder = os.path.join(current_directory, "resources")
    # Check if the resources folder exists
    if os.path.exists(resources_folder) and os.path.isdir(resources_folder):
        original_image = cv2.imread(os.path.join('resources', 'DIPSourceHW2.png'), cv2.IMREAD_GRAYSCALE)

    else:
        original_image = cv2.imread('DIPSourceHW2.jpg', cv2.IMREAD_GRAYSCALE)

    return original_image




def q1(img, alpha, filter_size):
    shape_0_res = img.shape[0] % alpha
    shape_1_res = img.shape[1] % alpha
    if shape_0_res != 0:
        img = img[:-shape_0_res, :-shape_1_res]
    filters = {"sinc_filter": generate_sinc_filter(filter_size),
               "gaussian_filter": generate_gaussian_filter(filter_size)}
    image_dict = {}
    for filter_key, filter_value in filters.items():
        img_with_filter = signal.convolve2d(img, filter_value, mode='same', boundary='wrap')
        img_with_filter_low_res = img_with_filter[::alpha, ::alpha]
        image_dict[filter_key + "_l"] = img_with_filter_low_res
        image_dict[filter_key + "_h"] = img_with_filter
        cv2.imwrite(os.path.join("results", f'img_{filter_key}_high.png'), img_with_filter)
        cv2.imwrite(os.path.join("results", f'img_{filter_key}_low.png'), img_with_filter_low_res)
    return image_dict


def generate_down_sample_cnv_mat(img, ratio):
    down_sampled_matrix = np.zeros((np.power(img.shape[0], 2) // np.power(ratio, 2), np.power(img.shape[0], 2)))
    new_img_size = img.shape[0] // ratio

    for i in range(new_img_size):
        for j in range(new_img_size):
            down_sampled_matrix[
                i * new_img_size + j,
                i * new_img_size * (np.power(ratio, 2)) + j * ratio
            ] = 1

    column_indices = np.arange(img.size)
    row_indices = np.arange(img.size)[:, np.newaxis]
    conv_mat = img.reshape(img.size)
    shift_indices = (row_indices + column_indices) % len(conv_mat)
    conv_cyclic = conv_mat[shift_indices]

    return down_sampled_matrix @ conv_cyclic


def generate_R(high_res_patches, alpha):
    R = list()
    for patch in high_res_patches:
        R_j = generate_down_sample_cnv_mat(patch, alpha)
        R.append(R_j)
    return R


def generate_q_vector(low_res_patches):
    q_vec = []
    for patch in low_res_patches:
        q_i = patch.reshape(patch.size)
        q_vec.append(q_i)
    return np.array(q_vec)


def initalize_k_with_delta_function():
    k = fftpack.fftshift(scipy.signal.unit_impulse((filter_size, filter_size)))
    k = k.reshape(k.size)
    return k


def down_sample_example_patches(R, k):
    r_alpha = list()
    for j in range(len(high_res_patches)):
        r_alpha.append(R[j] @ k)
    return np.array(r_alpha)


def find_NNs_and_compute_weights(r_alpha, q_vec, k_neighbors, sigma):
    tree = BallTree(r_alpha, leaf_size=2)
    w = np.zeros((len(q_vec), len(r_alpha)))
    for i, q_i in enumerate(q_vec):
        expand_q_i = np.expand_dims(q_i, 0)
        _, indices = tree.query(expand_q_i, k=k_neighbors)
        for j in indices:
            w[i, j] = np.exp(-0.5 * (pow(np.linalg.norm(q_i - r_alpha[j]), 2)) /
                             (pow(sigma, 2)))

    w_sum = np.sum(w, axis=1)
    for row in range(w.shape[0]):
        row_sum = w_sum[row]
        if row_sum:
            w[row] = w[row] / row_sum

    return w


def normalize_images(image_dict):
    normalized_dict = {}

    for key, image in image_dict.items():
        normalized_image = image / 255.0
        normalized_dict[key] = normalized_image

    return normalized_dict


def generate_C(size):
    a = -1.0
    b = 4.0

    mat_size = pow(size, 2)
    C = np.zeros((mat_size, mat_size))

    start_index_x = 0
    end_index_x = size
    start_index_y = size
    end_index_y = 2 * size

    diag_mat_2 = np.diag(np.full(size, a))
    for i in range(size - 1):
        C[start_index_x:end_index_x, start_index_y:end_index_y] = diag_mat_2
        C[start_index_y:end_index_y, start_index_x:end_index_x] = diag_mat_2

        start_index_x += size
        end_index_x += size
        start_index_y += size
        end_index_y += size

    diag_mat_1 = (
            np.diag(np.full(size, b)) +
            np.diag(np.full(size - 1, a), -1) +
            np.diag(np.full(size - 1, a), 1)
    )

    start_index = 0
    end_index = size

    for i in range(size):
        C[start_index:end_index, start_index:end_index] = diag_mat_1
        start_index += size
        end_index += size

    return C


def update_k(k, w, R, q_vec, sigma, filter_size):
    regular_weight = 0.5
    C = generate_C(filter_size)
    CT_C = C.T @ C
    v = np.zeros_like(k)
    mat = np.zeros((pow(filter_size, 2), pow(filter_size, 2)))

    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            if not w[i, j]:
                continue
            mat += w[i, j] * R[j].T @ R[j] + CT_C * regular_weight
            v += w[i, j] * R[j].T @ q_vec[i]

    mat = mat / (pow(sigma, 2))
    epsilon = 1e-12
    epsilon_mat = np.eye(k.shape[0]) * epsilon
    k = np.linalg.inv(mat + epsilon_mat) @ v
    return k


def estimate_kernel(high_res_patches, low_res_patches, filter_size, alpha):
    q_vec = generate_q_vector(low_res_patches)
    R = generate_R(high_res_patches, alpha)

    k = initalize_k_with_delta_function()

    sigma = 0.1
    k_neighbors = 5
    T = 6
    bar = tqdm(total=T)
    for _ in range(T):
        r_alpha = down_sample_example_patches(R, k)
        w = find_NNs_and_compute_weights(r_alpha, q_vec, k_neighbors, sigma)
        k = update_k(k, w, R, q_vec, sigma, filter_size)
        bar.update(1)

    k_reshaped = k.reshape((filter_size, filter_size))
    return k_reshaped


def create_patches(image, patch_size=15, step_size=1):
    patches = list()
    vertical_size = int(image.shape[0] - patch_size)
    horizontal_size = int(image.shape[1] - patch_size)

    for i in range(0, vertical_size, step_size):
        for j in range(0, horizontal_size, step_size):
            patch = image[i:i + patch_size, j:j + patch_size]
            patches.append(patch)

    return patches


def wiener(image, psf, K):
    if np.sum(psf):
        psf /= np.sum(psf)
    kernel = fftpack.fft2(psf, shape=image.shape)
    kernel = np.conj(kernel) / (np.power(np.abs(kernel), 2) + K)
    copied_image = np.copy(image)
    fft2_image = fftpack.fft2(copied_image)
    return np.abs(fftpack.ifft2(fft2_image * kernel))


def psnr(image1, image2):
    mse = np.mean(np.power(np.subtract(image1, image2), 2))
    PIXEL_MAX = 1.0
    return 20 * np.log10(PIXEL_MAX) - 10 * np.log10(mse)


if __name__ == '__main__':
    create_folders()
    alpha = 3
    filter_size = 15
    original_image = input_parser()
    print("Question 1...")

    img_dict = q1(original_image, alpha, filter_size)
    img_dict = normalize_images(img_dict)
    l_im_sinc_upsample = cv2.resize(img_dict["sinc_filter_l"],
                                    (int(img_dict["sinc_filter_l"].shape[1] * alpha),
                                     int(img_dict["sinc_filter_l"].shape[0] * alpha)))
    l_im_gaussian_upsample = cv2.resize(img_dict["gaussian_filter_l"],
                                        (int(img_dict["gaussian_filter_l"].shape[1] * alpha),
                                         int(img_dict["gaussian_filter_l"].shape[0] * alpha)))

    upsamples = {
        "sinc": l_im_sinc_upsample,
        "gaussian": l_im_gaussian_upsample
    }

    print("Question 2...")
    estimated_k_hat = {}
    kernel_types = ["gaussian", "sinc"]
    for kernel_type in kernel_types:
        print(f"USING kernel_type: {kernel_type}..")
        low_res_patches = create_patches(
            img_dict[f"{kernel_type}_filter_l"],
            patch_size=filter_size // alpha,
            step_size=1
        )
        high_res_patches = create_patches(
            img_dict[f"{kernel_type}_filter_l"],
            patch_size=filter_size,
            step_size=alpha
        )
        estimated_k_hat[kernel_type] = estimate_kernel(high_res_patches, low_res_patches, filter_size, alpha)

    print("Question 3 +4 ...")
    for i, kernel_type in enumerate(kernel_types):
        l_im_estimated_recon = wiener(upsamples[kernel_type], estimated_k_hat[kernel_type], 0.1)
        l_im_wrong_f_recon = wiener(upsamples[kernel_type], estimated_k_hat[kernel_types[np.abs(i - 1)]], 0.1)
        label_of_wrong= f"{kernel_type}_from_{kernel_types[np.abs(i - 1)]}"
        recon_estimated_f_psnr = psnr(l_im_estimated_recon, img_dict[f"{kernel_type}_filter_h"])
        recon_wrong_f_psnr = psnr(l_im_wrong_f_recon, img_dict[f"{kernel_type}_filter_h"])
        print(label_of_wrong, recon_wrong_f_psnr)
        print(f"{kernel_type} estimated from {kernel_type}:", recon_estimated_f_psnr)

        cv2.imwrite(os.path.join("results", f'{kernel_type}_estimated_psnr_{recon_estimated_f_psnr}.png'),
                    l_im_estimated_recon * 255)
        cv2.imwrite(os.path.join("results", f'{kernel_type}_wrong_psnr_{recon_wrong_f_psnr}.png'),
                    l_im_wrong_f_recon * 255)
