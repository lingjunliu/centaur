
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_extract_glimpse_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.rand(1, 28, 28, 3).astype(np.float32)
    size_tensor = np.array([10, 10], dtype=np.int32)
    offsets_tensor = np.array([[0.0, 0.0]], dtype=np.float32)
    centered = True
    normalized = True
    uniform_noise = True
    noise = "uniform"
    name = "glimpse1"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "offsets": offsets_tensor,
        "centered": centered,
        "normalized": normalized,
        "uniform_noise": True,
        "noise": "uniform",
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.rand(4, 64, 64, 1).astype(np.float32)
    size_tensor = np.array([20, 20], dtype=np.int32)
    offsets_tensor = np.array([[0.5, 0.5], [0.2, 0.8], [0.7, 0.3], [0.9, 0.1]], dtype=np.float32)
    centered = False
    normalized = False
    uniform_noise = False
    noise = "gaussian"
    name = "glimpse2"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "offsets": offsets_tensor,
        "centered": centered,
        "normalized": normalized,
        "uniform_noise": False,
        "noise": "gaussian",
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.rand(2, 128, 128, 3).astype(np.float32)
    size_tensor = np.array([32, 64], dtype=np.int32)
    offsets_tensor = np.array([[-0.5, -0.5], [0.5, 0.5]], dtype=np.float32)
    centered = True
    normalized = True
    uniform_noise = False
    noise = "zero"
    name = "glimpse3"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "offsets": offsets_tensor,
        "centered": centered,
        "normalized": normalized,
        "uniform_noise": False,
        "noise": "zero",
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.rand(1, 32, 32, 1).astype(np.float32)
    size_tensor = np.array([16, 8], dtype=np.int32)
    offsets_tensor = np.array([[10.0, 5.0]], dtype=np.float32)
    centered = False
    normalized = False
    uniform_noise = True
    noise = "uniform"
    name = "glimpse4"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "offsets": offsets_tensor,
        "centered": centered,
        "normalized": normalized,
        "uniform_noise": True,
        "noise": "uniform",
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.random.rand(8, 256, 256, 3).astype(np.float32)
    size_tensor = np.array([64, 64], dtype=np.int32)
    offsets_tensor = np.random.rand(8, 2).astype(np.float32)
    centered = True
    normalized = True
    uniform_noise = False
    noise = "gaussian"
    name = "glimpse5"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "offsets": offsets_tensor,
        "centered": centered,
        "normalized": normalized,
        "uniform_noise": False,
        "noise": "gaussian",
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6
    input_tensor = np.random.rand(2, 50, 50, 1).astype(np.float32)
    size_tensor = np.array([25, 25], dtype=np.int32)
    offsets_tensor = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    centered = True
    normalized = True
    uniform_noise = True
    noise = "uniform"
    name = "glimpse6"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "offsets": offsets_tensor,
        "centered": centered,
        "normalized": normalized,
        "uniform_noise": True,
        "noise": "uniform",
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.random.rand(3, 100, 75, 3).astype(np.float32)
    size_tensor = np.array([40, 30], dtype=np.int32)
    offsets_tensor = np.array([[20.0, 15.0], [50.0, 37.5], [80.0, 60.0]], dtype=np.float32)
    centered = False
    normalized = False
    uniform_noise = False
    noise = "gaussian"
    name = "glimpse7"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "offsets": offsets_tensor,
        "centered": centered,
        "normalized": normalized,
        "uniform_noise": False,
        "noise": "gaussian",
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.random.rand(1, 40, 40, 1).astype(np.float32)
    size_tensor = np.array([20, 20], dtype=np.int32)
    offsets_tensor = np.array([[-0.2, 0.3]], dtype=np.float32)
    centered = True
    normalized = True
    uniform_noise = False
    noise = "zero"
    name = "glimpse8"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "offsets": offsets_tensor,
        "centered": centered,
        "normalized": normalized,
        "uniform_noise": False,
        "noise": "zero",
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.random.rand(5, 16, 16, 3).astype(np.float32)
    size_tensor = np.array([8, 4], dtype=np.int32)
    offsets_tensor = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.2, 0.8], [0.8, 0.2]], dtype=np.float32)
    centered = False
    normalized = True
    uniform_noise = False
    noise = "gaussian"
    name = "glimpse9"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "offsets": offsets_tensor,
        "centered": centered,
        "normalized": normalized,
        "uniform_noise": False,
        "noise": "gaussian",
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.random.rand(10, 32, 64, 1).astype(np.float32)
    size_tensor = np.array([16, 32], dtype=np.int32)
    offsets_tensor = np.random.uniform(low=-1.0, high=1.0, size=(10, 2)).astype(np.float32)
    centered = True
    normalized = True
    uniform_noise = True
    noise = "uniform"
    name = "glimpse10"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "offsets": offsets_tensor,
        "centered": centered,
        "normalized": normalized,
        "uniform_noise": True,
        "noise": "uniform",
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ExtractGlimpse"] = tf_raw_ops_extract_glimpse_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ExtractGlimpse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ExtractGlimpse'.")

check_valid('tf.raw_ops.ExtractGlimpse', generated_inputs['tf.raw_ops.ExtractGlimpse'], lib="tf", suffix=0)
