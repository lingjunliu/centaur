
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_extract_glimpse_inputs():
    list_of_inputs = []

    # Input 1
    input_val = np.array([[[[0.0], [1.0], [2.0]], [[3.0], [4.0], [5.0]], [[6.0], [7.0], [8.0]]]], dtype=np.float32)
    size_val = np.array([2, 2], dtype=np.int32)
    offsets_val = np.array([[1, 1]], dtype=np.float32)
    centered_val = False
    normalized_val = False
    noise_val = 'uniform'
    name_val = 'glimpse1'

    input_dict = {
        "input": input_val,
        "size": size_val,
        "offsets": offsets_val,
        "centered": centered_val,
        "normalized": normalized_val,
        "noise": noise_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_val = np.random.rand(2, 10, 10, 3).astype(np.float32)
    size_val = np.array([5, 5], dtype=np.int32)
    offsets_val = np.array([[0.5, 0.5], [0.2, 0.8]], dtype=np.float32)
    centered_val = True
    normalized_val = True
    noise_val = 'gaussian'
    name_val = 'glimpse2'

    input_dict = {
        "input": input_val,
        "size": size_val,
        "offsets": offsets_val,
        "centered": centered_val,
        "normalized": normalized_val,
        "noise": noise_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_val = np.random.rand(1, 20, 30, 1).astype(np.float32)
    size_val = np.array([10, 15], dtype=np.int32)
    offsets_val = np.array([[5, 10]], dtype=np.float32)
    centered_val = False
    normalized_val = False
    noise_val = 'zero'
    name_val = 'glimpse3'

    input_dict = {
        "input": input_val,
        "size": size_val,
        "offsets": offsets_val,
        "centered": centered_val,
        "normalized": normalized_val,
        "noise": noise_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_val = np.random.rand(4, 50, 50, 3).astype(np.float32)
    size_val = np.array([20, 20], dtype=np.int32)
    offsets_val = np.random.rand(4, 2).astype(np.float32)
    centered_val = True
    normalized_val = True
    noise_val = 'uniform'
    name_val = 'glimpse4'

    input_dict = {
        "input": input_val,
        "size": size_val,
        "offsets": offsets_val,
        "centered": centered_val,
        "normalized": normalized_val,
        "noise": noise_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_val = np.random.rand(1, 5, 5, 1).astype(np.float32)
    size_val = np.array([3, 3], dtype=np.int32)
    offsets_val = np.array([[0.0, 0.0]], dtype=np.float32)
    centered_val = False
    normalized_val = True
    noise_val = 'gaussian'
    name_val = 'glimpse5'

    input_dict = {
        "input": input_val,
        "size": size_val,
        "offsets": offsets_val,
        "centered": centered_val,
        "normalized": normalized_val,
        "noise": noise_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_val = np.random.rand(2, 15, 25, 3).astype(np.float32)
    size_val = np.array([8, 12], dtype=np.int32)
    offsets_val = np.array([[2, 5], [7, 13]], dtype=np.float32)
    centered_val = False
    normalized_val = False
    noise_val = 'uniform'
    name_val = 'glimpse6'

    input_dict = {
        "input": input_val,
        "size": size_val,
        "offsets": offsets_val,
        "centered": centered_val,
        "normalized": normalized_val,
        "noise": noise_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_val = np.random.rand(1, 32, 32, 3).astype(np.float32)
    size_val = np.array([16, 16], dtype=np.int32)
    offsets_val = np.array([[0.0, 0.0]], dtype=np.float32)
    centered_val = True
    normalized_val = True
    noise_val = 'zero'
    name_val = 'glimpse7'

    input_dict = {
        "input": input_val,
        "size": size_val,
        "offsets": offsets_val,
        "centered": centered_val,
        "normalized": normalized_val,
        "noise": noise_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input_val = np.random.rand(3, 8, 8, 1).astype(np.float32)
    size_val = np.array([4, 4], dtype=np.int32)
    offsets_val = np.array([[-0.5, -0.5], [0.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    centered_val = True
    normalized_val = True
    noise_val = 'uniform'
    name_val = 'glimpse8'

    input_dict = {
        "input": input_val,
        "size": size_val,
        "offsets": offsets_val,
        "centered": centered_val,
        "normalized": normalized_val,
        "noise": noise_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_val = np.random.rand(1, 64, 64, 3).astype(np.float32)
    size_val = np.array([32, 32], dtype=np.int32)
    offsets_val = np.array([[32, 32]], dtype=np.float32)
    centered_val = False
    normalized_val = False
    noise_val = 'gaussian'
    name_val = 'glimpse9'

    input_dict = {
        "input": input_val,
        "size": size_val,
        "offsets": offsets_val,
        "centered": centered_val,
        "normalized": normalized_val,
        "noise": noise_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_val = np.random.rand(2, 28, 28, 1).astype(np.float32)
    size_val = np.array([14, 14], dtype=np.int32)
    offsets_val = np.array([[-0.2, 0.3], [0.4, -0.1]], dtype=np.float32)
    centered_val = True
    normalized_val = True
    noise_val = 'zero'
    name_val = 'glimpse10'

    input_dict = {
        "input": input_val,
        "size": size_val,
        "offsets": offsets_val,
        "centered": centered_val,
        "normalized": normalized_val,
        "noise": noise_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.extract_glimpse"] = tf_image_extract_glimpse_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.extract_glimpse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.extract_glimpse'.")

check_valid('tf.image.extract_glimpse', generated_inputs['tf.image.extract_glimpse'], lib="tf", suffix=0)
