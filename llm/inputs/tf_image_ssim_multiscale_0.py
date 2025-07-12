
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_ssim_multiscale_inputs():
    list_of_inputs = []

    # Input 1: Basic valid case
    img1 = np.random.rand(1, 256, 256, 3).astype(np.float32)
    img2 = np.random.rand(1, 256, 256, 3).astype(np.float32)
    img1 = np.abs(img1)
    img2 = np.abs(img2)
    max_val = 1.0
    power_factors = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
    filter_size = 11
    filter_sigma = 1.5
    k1 = 0.01
    k2 = 0.03

    input_dict = {
        "img1": img1,
        "img2": img2,
        "max_val": max_val,
        "power_factors": power_factors,
        "filter_size": filter_size,
        "filter_sigma": filter_sigma,
        "k1": k1,
        "k2": k2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different image size
    img1 = np.random.rand(1, 128, 128, 3).astype(np.float32)
    img2 = np.random.rand(1, 128, 128, 3).astype(np.float32)
    img1 = np.abs(img1)
    img2 = np.abs(img2)
    max_val = 1.0
    power_factors = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
    filter_size = 11
    filter_sigma = 1.5
    k1 = 0.01
    k2 = 0.03

    input_dict = {
        "img1": img1,
        "img2": img2,
        "max_val": max_val,
        "power_factors": power_factors,
        "filter_size": filter_size,
        "filter_sigma": filter_sigma,
        "k1": k1,
        "k2": k2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different batch size
    img1 = np.random.rand(4, 256, 256, 3).astype(np.float32)
    img2 = np.random.rand(4, 256, 256, 3).astype(np.float32)
    img1 = np.abs(img1)
    img2 = np.abs(img2)
    max_val = 1.0
    power_factors = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
    filter_size = 11
    filter_sigma = 1.5
    k1 = 0.01
    k2 = 0.03

    input_dict = {
        "img1": img1,
        "img2": img2,
        "max_val": max_val,
        "power_factors": power_factors,
        "filter_size": filter_size,
        "filter_sigma": filter_sigma,
        "k1": k1,
        "k2": k2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different max_val
    img1 = np.random.rand(1, 256, 256, 3).astype(np.float32)
    img2 = np.random.rand(1, 256, 256, 3).astype(np.float32)
    img1 = np.abs(img1)
    img2 = np.abs(img2)
    max_val = 255.0
    power_factors = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
    filter_size = 11
    filter_sigma = 1.5
    k1 = 0.01
    k2 = 0.03

    input_dict = {
        "img1": img1,
        "img2": img2,
        "max_val": max_val,
        "power_factors": power_factors,
        "filter_size": filter_size,
        "filter_sigma": filter_sigma,
        "k1": k1,
        "k2": k2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different power_factors
    img1 = np.random.rand(1, 256, 256, 3).astype(np.float32)
    img2 = np.random.rand(1, 256, 256, 3).astype(np.float32)
    img1 = np.abs(img1)
    img2 = np.abs(img2)
    max_val = 1.0
    power_factors = (0.1, 0.2, 0.3, 0.2, 0.2)
    filter_size = 11
    filter_sigma = 1.5
    k1 = 0.01
    k2 = 0.03

    input_dict = {
        "img1": img1,
        "img2": img2,
        "max_val": max_val,
        "power_factors": power_factors,
        "filter_size": filter_size,
        "filter_sigma": filter_sigma,
        "k1": k1,
        "k2": k2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6: Different filter_size
    img1 = np.random.rand(1, 256, 256, 3).astype(np.float32)
    img2 = np.random.rand(1, 256, 256, 3).astype(np.float32)
    img1 = np.abs(img1)
    img2 = np.abs(img2)
    max_val = 1.0
    power_factors = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
    filter_size = 7
    filter_sigma = 1.5
    k1 = 0.01
    k2 = 0.03

    input_dict = {
        "img1": img1,
        "img2": img2,
        "max_val": max_val,
        "power_factors": power_factors,
        "filter_size": filter_size,
        "filter_sigma": filter_sigma,
        "k1": k1,
        "k2": k2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different filter_sigma
    img1 = np.random.rand(1, 256, 256, 3).astype(np.float32)
    img2 = np.random.rand(1, 256, 256, 3).astype(np.float32)
    img1 = np.abs(img1)
    img2 = np.abs(img2)
    max_val = 1.0
    power_factors = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
    filter_size = 11
    filter_sigma = 2.0
    k1 = 0.01
    k2 = 0.03

    input_dict = {
        "img1": img1,
        "img2": img2,
        "max_val": max_val,
        "power_factors": power_factors,
        "filter_size": filter_size,
        "filter_sigma": filter_sigma,
        "k1": k1,
        "k2": k2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different k1
    img1 = np.random.rand(1, 256, 256, 3).astype(np.float32)
    img2 = np.random.rand(1, 256, 256, 3).astype(np.float32)
    img1 = np.abs(img1)
    img2 = np.abs(img2)
    max_val = 1.0
    power_factors = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
    filter_size = 11
    filter_sigma = 1.5
    k1 = 0.05
    k2 = 0.03

    input_dict = {
        "img1": img1,
        "img2": img2,
        "max_val": max_val,
        "power_factors": power_factors,
        "filter_size": filter_size,
        "filter_sigma": filter_sigma,
        "k1": k1,
        "k2": k2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different k2
    img1 = np.random.rand(1, 256, 256, 3).astype(np.float32)
    img2 = np.random.rand(1, 256, 256, 3).astype(np.float32)
    img1 = np.abs(img1)
    img2 = np.abs(img2)
    max_val = 1.0
    power_factors = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
    filter_size = 11
    filter_sigma = 1.5
    k1 = 0.01
    k2 = 0.1

    input_dict = {
        "img1": img1,
        "img2": img2,
        "max_val": max_val,
        "power_factors": power_factors,
        "filter_size": filter_size,
        "filter_sigma": filter_sigma,
        "k1": k1,
        "k2": k2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Grayscale images
    img1 = np.random.rand(1, 256, 256, 1).astype(np.float32)
    img2 = np.random.rand(1, 256, 256, 1).astype(np.float32)
    img1 = np.abs(img1)
    img2 = np.abs(img2)
    max_val = 1.0
    power_factors = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
    filter_size = 11
    filter_sigma = 1.5
    k1 = 0.01
    k2 = 0.03

    input_dict = {
        "img1": img1,
        "img2": img2,
        "max_val": max_val,
        "power_factors": power_factors,
        "filter_size": filter_size,
        "filter_sigma": filter_sigma,
        "k1": k1,
        "k2": k2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Smaller filter size and sigma
    img1 = np.random.rand(1, 64, 64, 3).astype(np.float32)
    img2 = np.random.rand(1, 64, 64, 3).astype(np.float32)
    img1 = np.abs(img1)
    img2 = np.abs(img2)
    max_val = 1.0
    power_factors = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
    filter_size = 3
    filter_sigma = 0.5
    k1 = 0.01
    k2 = 0.03

    input_dict = {
        "img1": img1,
        "img2": img2,
        "max_val": max_val,
        "power_factors": power_factors,
        "filter_size": filter_size,
        "filter_sigma": filter_sigma,
        "k1": k1,
        "k2": k2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Different image sizes with smaller filter
    img1 = np.random.rand(1, 32, 32, 3).astype(np.float32)
    img2 = np.random.rand(1, 32, 32, 3).astype(np.float32)
    img1 = np.abs(img1)
    img2 = np.abs(img2)
    max_val = 1.0
    power_factors = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
    filter_size = 3
    filter_sigma = 0.5
    k1 = 0.01
    k2 = 0.03
    input_dict = {
        "img1": img1,
        "img2": img2,
        "max_val": max_val,
        "power_factors": power_factors,
        "filter_size": filter_size,
        "filter_sigma": filter_sigma,
        "k1": k1,
        "k2": k2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13: Single channel images with values between 0 and 1
    img1 = np.random.rand(1, 64, 64, 1).astype(np.float32)
    img2 = np.random.rand(1, 64, 64, 1).astype(np.float32)
    img1 = np.clip(img1, 0.0, 1.0)
    img2 = np.clip(img2, 0.0, 1.0)
    max_val = 1.0
    power_factors = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
    filter_size = 3
    filter_sigma = 0.5
    k1 = 0.01
    k2 = 0.03
    input_dict = {
        "img1": img1,
        "img2": img2,
        "max_val": max_val,
        "power_factors": power_factors,
        "filter_size": filter_size,
        "filter_sigma": filter_sigma,
        "k1": k1,
        "k2": k2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.ssim_multiscale"] = tf_image_ssim_multiscale_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.ssim_multiscale' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.ssim_multiscale'.")

check_valid('tf.image.ssim_multiscale', generated_inputs['tf.image.ssim_multiscale'], lib="tf", suffix=0)
