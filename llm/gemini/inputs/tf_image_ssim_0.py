
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_ssim_inputs():
    list_of_inputs = []

    # Input 1: Basic grayscale images
    img1 = np.random.rand(1, 32, 32, 1).astype(np.float32)
    img2 = np.random.rand(1, 32, 32, 1).astype(np.float32)
    max_val = 1.0
    filter_size = 11
    filter_sigma = 1.5
    k1 = 0.01
    k2 = 0.03
    return_index_map = False
    input_dict = {'img1': img1, 'img2': img2, 'max_val': max_val, 'filter_size': filter_size,
                  'filter_sigma': filter_sigma, 'k1': k1, 'k2': k2, 'return_index_map': return_index_map}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Color images
    img1 = np.random.rand(1, 32, 32, 3).astype(np.float32)
    img2 = np.random.rand(1, 32, 32, 3).astype(np.float32)
    max_val = 1.0
    filter_size = 11
    filter_sigma = 1.5
    k1 = 0.01
    k2 = 0.03
    return_index_map = False
    input_dict = {'img1': img1, 'img2': img2, 'max_val': max_val, 'filter_size': filter_size,
                  'filter_sigma': filter_sigma, 'k1': k1, 'k2': k2, 'return_index_map': return_index_map}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different batch sizes
    img1 = np.random.rand(5, 32, 32, 3).astype(np.float32)
    img2 = np.random.rand(5, 32, 32, 3).astype(np.float32)
    max_val = 1.0
    filter_size = 11
    filter_sigma = 1.5
    k1 = 0.01
    k2 = 0.03
    return_index_map = False
    input_dict = {'img1': img1, 'img2': img2, 'max_val': max_val, 'filter_size': filter_size,
                  'filter_sigma': filter_sigma, 'k1': k1, 'k2': k2, 'return_index_map': return_index_map}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different image sizes
    img1 = np.random.rand(1, 64, 64, 3).astype(np.float32)
    img2 = np.random.rand(1, 64, 64, 3).astype(np.float32)
    max_val = 1.0
    filter_size = 11
    filter_sigma = 1.5
    k1 = 0.01
    k2 = 0.03
    return_index_map = False
    input_dict = {'img1': img1, 'img2': img2, 'max_val': max_val, 'filter_size': filter_size,
                  'filter_sigma': filter_sigma, 'k1': k1, 'k2': k2, 'return_index_map': return_index_map}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Integer images
    img1 = np.random.randint(0, 256, size=(1, 32, 32, 3)).astype(np.float32)
    img2 = np.random.randint(0, 256, size=(1, 32, 32, 3)).astype(np.float32)
    max_val = 255.0
    filter_size = 11
    filter_sigma = 1.5
    k1 = 0.01
    k2 = 0.03
    return_index_map = False
    input_dict = {'img1': img1, 'img2': img2, 'max_val': max_val, 'filter_size': filter_size,
                  'filter_sigma': filter_sigma, 'k1': k1, 'k2': k2, 'return_index_map': return_index_map}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6: return_index_map = True
    img1 = np.random.rand(1, 32, 32, 1).astype(np.float32)
    img2 = np.random.rand(1, 32, 32, 1).astype(np.float32)
    max_val = 1.0
    filter_size = 11
    filter_sigma = 1.5
    k1 = 0.01
    k2 = 0.03
    return_index_map = True
    input_dict = {'img1': img1, 'img2': img2, 'max_val': max_val, 'filter_size': filter_size,
                  'filter_sigma': filter_sigma, 'k1': k1, 'k2': k2, 'return_index_map': return_index_map}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Smaller filter size (still >= 11 is required, or an error is thrown)
    img1 = np.random.rand(1, 32, 32, 3).astype(np.float32)
    img2 = np.random.rand(1, 32, 32, 3).astype(np.float32)
    max_val = 1.0
    filter_size = 11
    filter_sigma = 1.5
    k1 = 0.01
    k2 = 0.03
    return_index_map = False
    input_dict = {'img1': img1, 'img2': img2, 'max_val': max_val, 'filter_size': filter_size,
                  'filter_sigma': filter_sigma, 'k1': k1, 'k2': k2, 'return_index_map': return_index_map}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger filter sigma
    img1 = np.random.rand(1, 32, 32, 3).astype(np.float32)
    img2 = np.random.rand(1, 32, 32, 3).astype(np.float32)
    max_val = 1.0
    filter_size = 11
    filter_sigma = 3.0
    k1 = 0.01
    k2 = 0.03
    return_index_map = False
    input_dict = {'img1': img1, 'img2': img2, 'max_val': max_val, 'filter_size': filter_size,
                  'filter_sigma': filter_sigma, 'k1': k1, 'k2': k2, 'return_index_map': return_index_map}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different K1 and K2 values
    img1 = np.random.rand(1, 32, 32, 3).astype(np.float32)
    img2 = np.random.rand(1, 32, 32, 3).astype(np.float32)
    max_val = 1.0
    filter_size = 11
    filter_sigma = 1.5
    k1 = 0.05
    k2 = 0.1
    return_index_map = False
    input_dict = {'img1': img1, 'img2': img2, 'max_val': max_val, 'filter_size': filter_size,
                  'filter_sigma': filter_sigma, 'k1': k1, 'k2': k2, 'return_index_map': return_index_map}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: uint8 images with max_val 255
    img1 = np.random.randint(0, 256, size=(1, 32, 32, 3)).astype(np.float32)
    img2 = np.random.randint(0, 256, size=(1, 32, 32, 3)).astype(np.float32)
    max_val = 255.0
    filter_size = 11
    filter_sigma = 1.5
    k1 = 0.01
    k2 = 0.03
    return_index_map = False
    input_dict = {'img1': img1, 'img2': img2, 'max_val': max_val, 'filter_size': filter_size,
                  'filter_sigma': filter_sigma, 'k1': k1, 'k2': k2, 'return_index_map': return_index_map}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.ssim"] = tf_image_ssim_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.ssim' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.ssim'.")

check_valid('tf.image.ssim', generated_inputs['tf.image.ssim'], lib="tf", suffix=0)
