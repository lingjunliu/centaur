
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ExtractVolumePatches_inputs():
    list_of_inputs = []

    # Input 1
    input_val = np.random.rand(1, 2, 2, 2, 1).astype(np.float32)
    list_of_inputs.append({
        'input': input_val,
        'ksizes': [1, 1, 1, 1, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': "VALID",
        'name': "extract_1"
    })

    # Input 2
    input_val = np.random.randint(0, 10, size=(2, 3, 3, 3, 2)).astype(np.int32)
    list_of_inputs.append({
        'input': input_val,
        'ksizes': [1, 2, 2, 2, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': "SAME",
        'name': "extract_2"
    })

    # Input 3
    input_val = np.random.rand(1, 4, 4, 4, 3).astype(np.float64)
    list_of_inputs.append({
        'input': input_val,
        'ksizes': [1, 2, 2, 2, 1],
        'strides': [1, 2, 2, 2, 1],
        'padding': "VALID",
        'name': "extract_3"
    })

    # Input 4
    input_val = np.random.randint(0, 256, size=(1, 5, 5, 5, 1)).astype(np.uint8)
    list_of_inputs.append({
        'input': input_val,
        'ksizes': [1, 3, 3, 3, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': "SAME",
        'name': "extract_4"
    })

    # Input 5
    input_val = np.random.randint(-100, 100, size=(3, 2, 2, 2, 4)).astype(np.int64)
    list_of_inputs.append({
        'input': input_val,
        'ksizes': [1, 1, 1, 1, 1],
        'strides': [1, 2, 2, 2, 1],
        'padding': "VALID",
        'name': "extract_5"
    })

    # Input 6
    input_val = np.random.rand(2, 3, 4, 5, 1).astype(np.float32)
    list_of_inputs.append({
        'input': input_val,
        'ksizes': [1, 2, 2, 2, 1],
        'strides': [1, 1, 2, 2, 1],
        'padding': "SAME",
        'name': "extract_6"
    })

    # Input 7
    input_val = np.random.randint(-10, 10, size=(1, 2, 2, 2, 2)).astype(np.int32)
    list_of_inputs.append({
        'input': input_val,
        'ksizes': [1, 2, 2, 2, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': "VALID",
        'name': "extract_7"
    })

    # Input 8
    input_val = np.random.rand(1, 3, 3, 3, 1).astype(np.float64)
    list_of_inputs.append({
        'input': input_val,
        'ksizes': [1, 3, 1, 1, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': "VALID",
        'name': "extract_8"
    })

    # Input 9
    input_val = np.random.randint(0, 256, size=(2, 2, 3, 2, 1)).astype(np.uint8)
    list_of_inputs.append({
        'input': input_val,
        'ksizes': [1, 1, 2, 1, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': "SAME",
        'name': "extract_9"
    })

    # Input 10
    input_val = np.random.randint(-100, 100, size=(1, 4, 2, 4, 3)).astype(np.int64)
    list_of_inputs.append({
        'input': input_val,
        'ksizes': [1, 2, 1, 2, 1],
        'strides': [1, 2, 1, 2, 1],
        'padding': "VALID",
        'name': "extract_10"
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.ExtractVolumePatches"] = tf_raw_ops_ExtractVolumePatches_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ExtractVolumePatches' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ExtractVolumePatches'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.ExtractVolumePatches', generated_inputs['tf.raw_ops.ExtractVolumePatches'], lib="tf", suffix=0)
