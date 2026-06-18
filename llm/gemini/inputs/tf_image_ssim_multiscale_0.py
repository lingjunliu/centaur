
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def gen_ssim_multiscale_inputs():
    np.random.seed(42)
    list_of_inputs = []

    # Input 1: Standard float32 images, default parameters, max_val=255.0
    img1 = np.random.uniform(0.001, 255.0, (1, 256, 256, 3)).astype(np.float32)
    img2 = np.random.uniform(0.001, 255.0, (1, 256, 256, 3)).astype(np.float32)
    input_dict = {
        'img1': img1,
        'img2': img2,
        'max_val': 255.0,
        'power_factors': (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
        'filter_size': 11,
        'filter_sigma': 1.5,
        'k1': 0.01,
        'k2': 0.03
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Grayscale-like float32 images, max_val=1.0, batch size of 2
    img1 = np.random.uniform(0.001, 1.0, (2, 256, 256, 1)).astype(np.float32)
    img2 = np.random.uniform(0.001, 1.0, (2, 256, 256, 1)).astype(np.float32)
    input_dict = {
        'img1': img1,
        'img2': img2,
        'max_val': 1.0,
        'power_factors': (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
        'filter_size': 11,
        'filter_sigma': 1.5,
        'k1': 0.01,
        'k2': 0.03
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Smaller images with 2 scales (custom power factors)
    img1 = np.random.uniform(0.001, 255.0, (1, 64, 64, 3)).astype(np.float32)
    img2 = np.random.uniform(0.001, 255.0, (1, 64, 64, 3)).astype(np.float32)
    input_dict = {
        'img1': img1,
        'img2': img2,
        'max_val': 255.0,
        'power_factors': (0.5, 0.5),
        'filter_size': 11,
        'filter_sigma': 1.5,
        'k1': 0.01,
        'k2': 0.03
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different filter size and filter sigma
    img1 = np.random.uniform(0.001, 255.0, (1, 300, 300, 3)).astype(np.float32)
    img2 = np.random.uniform(0.001, 255.0, (1, 300, 300, 3)).astype(np.float32)
    input_dict = {
        'img1': img1,
        'img2': img2,
        'max_val': 255.0,
        'power_factors': (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
        'filter_size': 7,
        'filter_sigma': 1.1,
        'k1': 0.01,
        'k2': 0.03
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Custom k1 and k2 values
    img1 = np.random.uniform(0.001, 1.0, (1, 256, 256, 3)).astype(np.float32)
    img2 = np.random.uniform(0.001, 1.0, (1, 256, 256, 3)).astype(np.float32)
    input_dict = {
        'img1': img1,
        'img2': img2,
        'max_val': 1.0,
        'power_factors': (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
        'filter_size': 11,
        'filter_sigma': 1.5,
        'k1': 0.02,
        'k2': 0.04
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Higher batch dimensions (5D tensor shape)
    img1 = np.random.uniform(0.001, 1.0, (2, 3, 256, 256, 3)).astype(np.float32)
    img2 = np.random.uniform(0.001, 1.0, (2, 3, 256, 256, 3)).astype(np.float32)
    input_dict = {
        'img1': img1,
        'img2': img2,
        'max_val': 1.0,
        'power_factors': (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
        'filter_size': 11,
        'filter_sigma': 1.5,
        'k1': 0.01,
        'k2': 0.03
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Broadcasting batch dimensions (img1 has 1 batch, img2 has 5 batches)
    img1 = np.random.uniform(0.001, 255.0, (1, 256, 256, 3)).astype(np.float32)
    img2 = np.random.uniform(0.001, 255.0, (5, 256, 256, 3)).astype(np.float32)
    input_dict = {
        'img1': img1,
        'img2': img2,
        'max_val': 255.0,
        'power_factors': (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
        'filter_size': 11,
        'filter_sigma': 1.5,
        'k1': 0.01,
        'k2': 0.03
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger image size with 6 scales (custom power factors)
    img1 = np.random.uniform(0.001, 1.0, (1, 512, 512, 1)).astype(np.float32)
    img2 = np.random.uniform(0.001, 1.0, (1, 512, 512, 1)).astype(np.float32)
    input_dict = {
        'img1': img1,
        'img2': img2,
        'max_val': 1.0,
        'power_factors': (0.01, 0.09, 0.2, 0.3, 0.3, 0.1),
        'filter_size': 11,
        'filter_sigma': 1.5,
        'k1': 0.01,
        'k2': 0.03
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single-scale MS-SSIM
    img1 = np.random.uniform(0.001, 1.0, (1, 64, 64, 3)).astype(np.float32)
    img2 = np.random.uniform(0.001, 1.0, (1, 64, 64, 3)).astype(np.float32)
    input_dict = {
        'img1': img1,
        'img2': img2,
        'max_val': 1.0,
        'power_factors': (1.0,),
        'filter_size': 11,
        'filter_sigma': 1.5,
        'k1': 0.01,
        'k2': 0.03
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Float64 images, larger filter size, and larger image dimension
    img1 = np.random.uniform(0.001, 255.0, (1, 512, 512, 3)).astype(np.float64)
    img2 = np.random.uniform(0.001, 255.0, (1, 512, 512, 3)).astype(np.float64)
    input_dict = {
        'img1': img1,
        'img2': img2,
        'max_val': 255.0,
        'power_factors': (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
        'filter_size': 15,
        'filter_sigma': 2.0,
        'k1': 0.005,
        'k2': 0.015
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.image.ssim_multiscale"] = gen_ssim_multiscale_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.ssim_multiscale' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.ssim_multiscale'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.ssim_multiscale', generated_inputs['tf.image.ssim_multiscale'], lib="tf", suffix=0)
