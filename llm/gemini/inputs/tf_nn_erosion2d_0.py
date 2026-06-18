
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_erosion2d_inputs():
    list_of_inputs = []

    # 1. Standard float32 VALID
    list_of_inputs.append({
        'value': np.random.uniform(-10, 10, (1, 3, 3, 1)).astype(np.float32),
        'filters': np.random.uniform(-5, 5, (2, 2, 1)).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': "VALID",
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "erosion_1"
    })

    # 2. Standard float32 SAME with strides
    list_of_inputs.append({
        'value': np.random.uniform(-10, 10, (2, 5, 5, 3)).astype(np.float32),
        'filters': np.random.uniform(-5, 5, (3, 3, 3)).astype(np.float32),
        'strides': [1, 2, 2, 1],
        'padding': "SAME",
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "erosion_2"
    })

    # 3. float64 with VALID and dilations
    list_of_inputs.append({
        'value': np.random.uniform(-10, 10, (1, 10, 10, 2)).astype(np.float64),
        'filters': np.random.uniform(-5, 5, (3, 3, 2)).astype(np.float64),
        'strides': [1, 1, 1, 1],
        'padding': "VALID",
        'data_format': "NHWC",
        'dilations': [1, 2, 2, 1],
        'name': "erosion_3"
    })

    # 4. Negatives, SAME padding, stride and dilations
    list_of_inputs.append({
        'value': np.random.uniform(-50, -10, (2, 8, 8, 4)).astype(np.float32),
        'filters': np.random.uniform(-5, 0, (2, 2, 4)).astype(np.float32),
        'strides': [1, 2, 1, 1],
        'padding': "SAME",
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "erosion_4"
    })

    # 5. float32 with VALID, dilations, and strides=1
    list_of_inputs.append({
        'value': np.random.uniform(0, 255, (1, 6, 6, 1)).astype(np.float32),
        'filters': np.random.uniform(0, 10, (3, 3, 1)).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': "VALID",
        'data_format': "NHWC",
        'dilations': [1, 2, 2, 1],
        'name': "erosion_5"
    })

    # 6. float64, SAME padding
    list_of_inputs.append({
        'value': np.random.uniform(-1, 1, (1, 4, 4, 1)).astype(np.float64),
        'filters': np.random.uniform(-1, 1, (2, 2, 1)).astype(np.float64),
        'strides': [1, 1, 1, 1],
        'padding': "SAME",
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "erosion_6"
    })

    # 7. Large strides, float32
    list_of_inputs.append({
        'value': np.random.uniform(-5, 5, (4, 7, 7, 3)).astype(np.float32),
        'filters': np.random.uniform(-2, 2, (3, 3, 3)).astype(np.float32),
        'strides': [1, 3, 3, 1],
        'padding': "SAME",
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "erosion_7"
    })

    # 8. 1x1 filter, float32, VALID
    list_of_inputs.append({
        'value': np.random.uniform(-10, 10, (1, 5, 5, 1)).astype(np.float32),
        'filters': np.random.uniform(-5, 5, (1, 1, 1)).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': "VALID",
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "erosion_8"
    })

    # 9. Large input with dilations and strides
    list_of_inputs.append({
        'value': np.random.uniform(-100, 100, (2, 12, 12, 2)).astype(np.float32),
        'filters': np.random.uniform(-10, 10, (4, 4, 2)).astype(np.float32),
        'strides': [1, 2, 2, 1],
        'padding': "VALID",
        'data_format': "NHWC",
        'dilations': [1, 2, 2, 1],
        'name': "erosion_9"
    })

    # 10. Non-square input and filter shapes, float64
    list_of_inputs.append({
        'value': np.random.uniform(-5, 5, (1, 3, 5, 2)).astype(np.float64),
        'filters': np.random.uniform(-1, 1, (2, 3, 2)).astype(np.float64),
        'strides': [1, 1, 1, 1],
        'padding': "SAME",
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "erosion_10"
    })

    return list_of_inputs

generated_inputs["tf.nn.erosion2d"] = tf_nn_erosion2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.erosion2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.erosion2d'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.erosion2d', generated_inputs['tf.nn.erosion2d'], lib="tf", suffix=0)
