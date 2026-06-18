
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_ExtractImagePatches_inputs():
    list_of_inputs = []

    # Input 1
    images = np.random.randn(1, 4, 4, 1).astype(np.float32)
    ksizes = [1, 2, 2, 1]
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    name = "patch_1"
    list_of_inputs.append({
        "images": images,
        "ksizes": ksizes,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    })

    # Input 2
    images = np.random.randn(2, 8, 8, 3).astype(np.float32)
    ksizes = [1, 3, 3, 1]
    strides = [1, 2, 2, 1]
    rates = [1, 1, 1, 1]
    padding = "SAME"
    name = "patch_2"
    list_of_inputs.append({
        "images": images,
        "ksizes": ksizes,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    })

    # Input 3
    images = np.random.randint(0, 10, size=(1, 10, 10, 1)).astype(np.int32)
    ksizes = [1, 4, 4, 1]
    strides = [1, 1, 1, 1]
    rates = [1, 2, 2, 1]
    padding = "VALID"
    name = "patch_3"
    list_of_inputs.append({
        "images": images,
        "ksizes": ksizes,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    })

    # Input 4
    images = np.random.randn(4, 16, 16, 2).astype(np.float16)
    ksizes = [1, 2, 3, 1]
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "SAME"
    name = "patch_4"
    list_of_inputs.append({
        "images": images,
        "ksizes": ksizes,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    })

    # Input 5
    images = np.random.randint(0, 255, size=(1, 5, 5, 3)).astype(np.uint8)
    ksizes = [1, 3, 3, 1]
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    name = "patch_5"
    list_of_inputs.append({
        "images": images,
        "ksizes": ksizes,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    })

    # Input 6
    images = np.random.randn(1, 12, 12, 1).astype(np.float64)
    ksizes = [1, 5, 5, 1]
    strides = [1, 3, 3, 1]
    rates = [1, 1, 1, 1]
    padding = "SAME"
    name = "patch_6"
    list_of_inputs.append({
        "images": images,
        "ksizes": ksizes,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    })

    # Input 7
    images = np.random.randint(-128, 127, size=(1, 6, 6, 1)).astype(np.int8)
    ksizes = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    name = "patch_7"
    list_of_inputs.append({
        "images": images,
        "ksizes": ksizes,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    })

    # Input 8
    images = np.random.randint(-100, 100, size=(2, 4, 4, 4)).astype(np.int16)
    ksizes = [1, 2, 2, 1]
    strides = [1, 1, 1, 1]
    rates = [1, 2, 2, 1]
    padding = "SAME"
    name = "patch_8"
    list_of_inputs.append({
        "images": images,
        "ksizes": ksizes,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    })

    # Input 9
    images = (np.random.randn(1, 3, 3, 1) + 1j * np.random.randn(1, 3, 3, 1)).astype(np.complex64)
    ksizes = [1, 2, 2, 1]
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    name = "patch_9"
    list_of_inputs.append({
        "images": images,
        "ksizes": ksizes,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    })

    # Input 10
    images = np.random.randn(1, 14, 14, 3).astype(np.float32)
    ksizes = [1, 3, 3, 1]
    strides = [1, 3, 3, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    name = "patch_10"
    list_of_inputs.append({
        "images": images,
        "ksizes": ksizes,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.ExtractImagePatches"] = tf_raw_ops_ExtractImagePatches_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ExtractImagePatches' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ExtractImagePatches'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.ExtractImagePatches', generated_inputs['tf.raw_ops.ExtractImagePatches'], lib="tf", suffix=0)
