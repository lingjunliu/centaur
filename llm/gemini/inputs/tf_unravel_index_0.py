
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_unravel_index_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D indices, 2D dims, int32
    indices1 = np.array([2, 5, 7], dtype=np.int32)
    dims1 = np.array([3, 3], dtype=np.int32)
    list_of_inputs.append({
        "indices": indices1,
        "dims": dims1,
        "name": "unravel_1"
    })

    # Input 2: 0-D index, 2D dims, int32
    indices2 = np.array(4, dtype=np.int32)
    dims2 = np.array([3, 3], dtype=np.int32)
    list_of_inputs.append({
        "indices": indices2,
        "dims": dims2,
        "name": "unravel_2"
    })

    # Input 3: 1D indices, 3D dims, int64
    indices3 = np.array([0, 7, 11], dtype=np.int64)
    dims3 = np.array([2, 2, 3], dtype=np.int64)
    list_of_inputs.append({
        "indices": indices3,
        "dims": dims3,
        "name": "unravel_3"
    })

    # Input 4: 0-D index, 3D dims, int64
    indices4 = np.array(5, dtype=np.int64)
    dims4 = np.array([2, 3, 2], dtype=np.int64)
    list_of_inputs.append({
        "indices": indices4,
        "dims": dims4,
        "name": "unravel_4"
    })

    # Input 5: 1D indices, 1D dims, int32
    indices5 = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    dims5 = np.array([5], dtype=np.int32)
    list_of_inputs.append({
        "indices": indices5,
        "dims": dims5,
        "name": "unravel_5"
    })

    # Input 6: 1D indices, 4D dims, int32
    indices6 = np.array([10, 20, 30], dtype=np.int32)
    dims6 = np.array([2, 3, 4, 2], dtype=np.int32)
    list_of_inputs.append({
        "indices": indices6,
        "dims": dims6,
        "name": "unravel_6"
    })

    # Input 7: 1D indices, large values, int64
    indices7 = np.array([1000000, 2000000], dtype=np.int64)
    dims7 = np.array([3000, 3000], dtype=np.int64)
    list_of_inputs.append({
        "indices": indices7,
        "dims": dims7,
        "name": "unravel_7"
    })

    # Input 8: 0-D index, large dims, int64
    indices8 = np.array(8000000, dtype=np.int64)
    dims8 = np.array([2000, 2000, 5], dtype=np.int64)
    list_of_inputs.append({
        "indices": indices8,
        "dims": dims8,
        "name": "unravel_8"
    })

    # Input 9: 1D indices, 2D dims, int32
    indices9 = np.arange(12, dtype=np.int32)
    dims9 = np.array([3, 4], dtype=np.int32)
    list_of_inputs.append({
        "indices": indices9,
        "dims": dims9,
        "name": "unravel_9"
    })

    # Input 10: 1D indices, 5D dims, int64
    indices10 = np.array([1, 15, 31], dtype=np.int64)
    dims10 = np.array([2, 2, 2, 2, 2], dtype=np.int64)
    list_of_inputs.append({
        "indices": indices10,
        "dims": dims10,
        "name": "unravel_10"
    })

    return list_of_inputs

generated_inputs["tf.unravel_index"] = tf_unravel_index_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.unravel_index' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.unravel_index'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.unravel_index', generated_inputs['tf.unravel_index'], lib="tf", suffix=0)
