
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_image_ssim_multiscale_inputs():
    """
    Generates a list of valid inputs for tf.image.ssim_multiscale.
    The primary constraint is that the image dimensions at each scale must be
    greater than or equal to the filter size.
    `min(H, W) / (2^(len(power_factors) - 1)) >= filter_size`
    """
    list_of_inputs = []

    default_power_factors = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
    default_filter_size = 11
    default_filter_sigma = 1.5
    default_k1 = 0.01
    default_k2 = 0.03
    
    # For default settings (5 scales, filter_size=11), min image dim is 11 * 2^4 = 176
    min_dim_default = 176

    # Input 1: Basic case with large enough grayscale images
    img1_1 = np.random.rand(1, min_dim_default, min_dim_default, 1).astype(np.float32)
    img2_1 = np.random.rand(1, min_dim_default, min_dim_default, 1).astype(np.float32)
    input_dict_1 = {
        'img1': img1_1,
        'img2': img2_1,
        'max_val': 1.0,
        'power_factors': default_power_factors,
        'filter_size': default_filter_size,
        'filter_sigma': default_filter_sigma,
        'k1': default_k1,
        'k2': default_k2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Color images with 3 channels
    img1_2 = np.random.rand(2, 200, 180, 3).astype(np.float32)
    img2_2 = np.random.rand(2, 200, 180, 3).astype(np.float32)
    input_dict_2 = {
        'img1': img1_2,
        'img2': img2_2,
        'max_val': 1.0,
        'power_factors': default_power_factors,
        'filter_size': default_filter_size,
        'filter_sigma': default_filter_sigma,
        'k1': default_k1,
        'k2': default_k2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Images with dynamic range [0, 255]
    img1_3 = (np.random.rand(4, 176, 176, 1) * 255).astype(np.float32)
    img2_3 = (np.random.rand(4, 176, 176, 1) * 255).astype(np.float32)
    input_dict_3 = {
        'img1': img1_3,
        'img2': img2_3,
        'max_val': 255.0,
        'power_factors': default_power_factors,
        'filter_size': default_filter_size,
        'filter_sigma': default_filter_sigma,
        'k1': default_k1,
        'k2': default_k2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Fewer scales, allowing smaller images. min_dim = 11 * 2^2 = 44
    img1_4 = np.random.rand(1, 50, 50, 1).astype(np.float32)
    img2_4 = np.random.rand(1, 50, 50, 1).astype(np.float32)
    input_dict_4 = {
        'img1': img1_4,
        'img2': img2_4,
        'max_val': 1.0,
        'power_factors': (0.25, 0.5, 0.25),
        'filter_size': default_filter_size,
        'filter_sigma': default_filter_sigma,
        'k1': default_k1,
        'k2': default_k2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Smaller filter size. min_dim = 7 * 2^4 = 112
    img1_5 = np.random.rand(1, 128, 128, 3).astype(np.float32)
    img2_5 = np.random.rand(1, 128, 128, 3).astype(np.float32)
    input_dict_5 = {
        'img1': img1_5,
        'img2': img2_5,
        'max_val': 1.0,
        'power_factors': default_power_factors,
        'filter_size': 7,
        'filter_sigma': 1.0,
        'k1': default_k1,
        'k2': default_k2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Different k1 and k2 values
    img1_6 = np.random.rand(1, 180, 180, 1).astype(np.float32)
    img2_6 = np.random.rand(1, 180, 180, 1).astype(np.float32)
    input_dict_6 = {
        'img1': img1_6,
        'img2': img2_6,
        'max_val': 1.0,
        'power_factors': default_power_factors,
        'filter_size': default_filter_size,
        'filter_sigma': default_filter_sigma,
        'k1': 0.02,
        'k2': 0.05
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Identical images (should yield SSIM of 1)
    img1_7 = np.random.rand(2, 176, 176, 3).astype(np.float32)
    img2_7 = copy.deepcopy(img1_7)
    input_dict_7 = {
        'img1': img1_7,
        'img2': img2_7,
        'max_val': 1.0,
        'power_factors': default_power_factors,
        'filter_size': default_filter_size,
        'filter_sigma': default_filter_sigma,
        'k1': default_k1,
        'k2': default_k2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Single scale (equivalent to SSIM). min_dim = 11 * 2^0 = 11
    img1_8 = np.random.rand(1, 32, 32, 1).astype(np.float32)
    img2_8 = np.random.rand(1, 32, 32, 1).astype(np.float32)
    input_dict_8 = {
        'img1': img1_8,
        'img2': img2_8,
        'max_val': 1.0,
        'power_factors': (1.0,),
        'filter_size': default_filter_size,
        'filter_sigma': default_filter_sigma,
        'k1': default_k1,
        'k2': default_k2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: float64 dtype for images
    img1_9 = np.random.rand(1, 176, 176, 1).astype(np.float64)
    img2_9 = np.random.rand(1, 176, 176, 1).astype(np.float64)
    input_dict_9 = {
        'img1': img1_9,
        'img2': img2_9,
        'max_val': 1.0,
        'power_factors': default_power_factors,
        'filter_size': default_filter_size,
        'filter_sigma': default_filter_sigma,
        'k1': default_k1,
        'k2': default_k2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: Broadcastable batch dimensions. 4 scales -> min_dim = 11 * 2^3 = 88
    img1_10 = np.random.rand(2, 1, 100, 100, 1).astype(np.float32)
    img2_10 = np.random.rand(1, 3, 100, 100, 1).astype(np.float32)
    input_dict_10 = {
        'img1': img1_10,
        'img2': img2_10,
        'max_val': 1.0,
        'power_factors': (0.1, 0.2, 0.3, 0.4),
        'filter_size': default_filter_size,
        'filter_sigma': default_filter_sigma,
        'k1': default_k1,
        'k2': default_k2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.image.ssim_multiscale"] = tf_image_ssim_multiscale_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.ssim_multiscale' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.ssim_multiscale'.")

check_valid('tf.image.ssim_multiscale', generated_inputs['tf.image.ssim_multiscale'], lib="tf", suffix=0)
