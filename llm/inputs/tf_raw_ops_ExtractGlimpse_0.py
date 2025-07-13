
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_extractglimpse_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.rand(1, 28, 28, 3).astype(np.float32)
    size_tensor = np.array([10, 10]).astype(np.int32)
    offsets_tensor = np.array([[0.0, 0.0]]).astype(np.float32)
    centered = True
    normalized = True
    uniform_noise = True
    noise = "uniform"
    name = "glimpse_1"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "offsets": offsets_tensor,
        "centered": centered,
        "normalized": normalized,
        "uniform_noise": uniform_noise,
        "noise": noise,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.rand(4, 64, 64, 1).astype(np.float32)
    size_tensor = np.array([20, 20]).astype(np.int32)
    offsets_tensor = np.array([[0.2, 0.3], [0.8, 0.7], [0.1, 0.9], [0.5, 0.5]]).astype(np.float32)
    centered = False
    normalized = True
    uniform_noise = False
    noise = "gaussian"
    name = "glimpse_2"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "offsets": offsets_tensor,
        "centered": centered,
        "normalized": normalized,
        "uniform_noise": uniform_noise,
        "noise": noise,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.rand(2, 128, 128, 3).astype(np.float32)
    size_tensor = np.array([32, 32]).astype(np.int32)
    offsets_tensor = np.array([[10.0, 20.0], [50.0, 60.0]]).astype(np.float32)
    centered = True
    normalized = False
    uniform_noise = True
    noise = "uniform"
    name = "glimpse_3"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "offsets": offsets_tensor,
        "centered": centered,
        "normalized": normalized,
        "uniform_noise": uniform_noise,
        "noise": noise,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.rand(1, 32, 32, 1).astype(np.float32)
    size_tensor = np.array([16, 8]).astype(np.int32)
    offsets_tensor = np.array([[0.5, -0.5]]).astype(np.float32)
    centered = True
    normalized = True
    uniform_noise = True
    noise = "uniform"
    name = "glimpse_4"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "offsets": offsets_tensor,
        "centered": centered,
        "normalized": normalized,
        "uniform_noise": uniform_noise,
        "noise": noise,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.random.rand(3, 100, 50, 3).astype(np.float32)
    size_tensor = np.array([25, 10]).astype(np.int32)
    offsets_tensor = np.array([[20.0, 10.0], [60.0, 30.0], [80.0, 40.0]]).astype(np.float32)
    centered = False
    normalized = False
    uniform_noise = False
    noise = "gaussian"
    name = "glimpse_5"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "offsets": offsets_tensor,
        "centered": centered,
        "normalized": normalized,
        "uniform_noise": uniform_noise,
        "noise": noise,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.random.rand(1, 64, 64, 3).astype(np.float32)
    size_tensor = np.array([64, 64]).astype(np.int32)
    offsets_tensor = np.array([[0.0, 0.0]]).astype(np.float32)
    centered = True
    normalized = True
    uniform_noise = True
    noise = "uniform"
    name = "glimpse_6"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "offsets": offsets_tensor,
        "centered": centered,
        "normalized": normalized,
        "uniform_noise": uniform_noise,
        "noise": noise,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    input_tensor = np.random.rand(2, 32, 32, 1).astype(np.float32)
    size_tensor = np.array([16, 16]).astype(np.int32)
    offsets_tensor = np.array([[-0.5, 0.5], [0.5, -0.5]]).astype(np.float32)
    centered = True
    normalized = True
    uniform_noise = False
    noise = "gaussian"
    name = "glimpse_7"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "offsets": offsets_tensor,
        "centered": centered,
        "normalized": normalized,
        "uniform_noise": uniform_noise,
        "noise": noise,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.random.rand(1, 10, 10, 3).astype(np.float32)
    size_tensor = np.array([5, 5]).astype(np.int32)
    offsets_tensor = np.array([[2.0, 3.0]]).astype(np.float32)
    centered = False
    normalized = False
    uniform_noise = True
    noise = "uniform"
    name = "glimpse_8"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "offsets": offsets_tensor,
        "centered": centered,
        "normalized": normalized,
        "uniform_noise": uniform_noise,
        "noise": noise,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.random.rand(2, 256, 256, 3).astype(np.float32)
    size_tensor = np.array([128, 64]).astype(np.int32)
    offsets_tensor = np.array([[0.25, 0.75], [0.75, 0.25]]).astype(np.float32)
    centered = True
    normalized = True
    uniform_noise = False
    noise = "gaussian"
    name = "glimpse_9"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "offsets": offsets_tensor,
        "centered": centered,
        "normalized": normalized,
        "uniform_noise": uniform_noise,
        "noise": noise,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.random.rand(4, 24, 32, 1).astype(np.float32)
    size_tensor = np.array([8, 12]).astype(np.int32)
    offsets_tensor = np.array([[5.0, 7.0], [10.0, 15.0], [1.0, 2.0], [20.0, 30.0]]).astype(np.float32)
    centered = False
    normalized = False
    uniform_noise = True
    noise = "uniform"
    name = "glimpse_10"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "offsets": offsets_tensor,
        'centered': centered,
        'normalized': normalized,
        'uniform_noise': uniform_noise,
        'noise': noise,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ExtractGlimpse"] = tf_raw_ops_extractglimpse_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ExtractGlimpse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ExtractGlimpse'.")

check_valid('tf.raw_ops.ExtractGlimpse', generated_inputs['tf.raw_ops.ExtractGlimpse'], lib="tf", suffix=0)
