
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_IndexedSlices_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    values = np.array([[1, 2], [3, 4]], dtype=np.float32)
    indices = np.array([0, 2], dtype=np.int32)
    dense_shape = np.array([4, 2], dtype=np.int32)

    input_dict = {
        "values": tf.constant(values).numpy(),
        "indices": tf.constant(indices).numpy(),
        "dense_shape": tf.constant(dense_shape).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different data type for values
    values = np.array([[1, 2], [3, 4]], dtype=np.int32)
    indices = np.array([1, 3], dtype=np.int32)
    dense_shape = np.array([5, 2], dtype=np.int32)

    input_dict = {
        "values": tf.constant(values).numpy(),
        "indices": tf.constant(indices).numpy(),
        "dense_shape": tf.constant(dense_shape).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D values
    values = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    dense_shape = np.array([3, 2, 2], dtype=np.int32)

    input_dict = {
        "values": tf.constant(values).numpy(),
        "indices": tf.constant(indices).numpy(),
        "dense_shape": tf.constant(dense_shape).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Single slice
    values = np.array([[1, 2]], dtype=np.float32)
    indices = np.array([2], dtype=np.int32)
    dense_shape = np.array([4, 2], dtype=np.int32)

    input_dict = {
        "values": tf.constant(values).numpy(),
        "indices": tf.constant(indices).numpy(),
        "dense_shape": tf.constant(dense_shape).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different shape for values
    values = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    indices = np.array([0, 2], dtype=np.int32)
    dense_shape = np.array([4, 3], dtype=np.int32)

    input_dict = {
        "values": tf.constant(values).numpy(),
        "indices": tf.constant(indices).numpy(),
        "dense_shape": tf.constant(dense_shape).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large dense shape
    values = np.array([[1, 2], [3, 4]], dtype=np.float32)
    indices = np.array([10, 20], dtype=np.int32)
    dense_shape = np.array([100, 2], dtype=np.int32)

    input_dict = {
        "values": tf.constant(values).numpy(),
        "indices": tf.constant(indices).numpy(),
        "dense_shape": tf.constant(dense_shape).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int64 indices
    values = np.array([[1, 2], [3, 4]], dtype=np.float32)
    indices = np.array([0, 2], dtype=np.int64)
    dense_shape = np.array([4, 2], dtype=np.int32)

    input_dict = {
        "values": tf.constant(values).numpy(),
        "indices": tf.constant(indices).numpy(),
        "dense_shape": tf.constant(dense_shape).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty values
    values = np.array([], dtype=np.float32).reshape(0,2)
    indices = np.array([], dtype=np.int32)
    dense_shape = np.array([4, 2], dtype=np.int32)

    input_dict = {
        "values": tf.constant(values).numpy(),
        "indices": tf.constant(indices).numpy(),
        "dense_shape": tf.constant(dense_shape).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    values = np.array([[1, 2], [3, 4]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    dense_shape = np.array([2, 2], dtype=np.int32)
    input_dict = {
        "values": tf.constant(values).numpy(),
        "indices": tf.constant(indices).numpy(),
        "dense_shape": tf.constant(dense_shape).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    values = np.array([1, 2, 3], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    dense_shape = np.array([3], dtype=np.int32)
    input_dict = {
        "values": tf.constant(values).numpy(),
        "indices": tf.constant(indices).numpy(),
        "dense_shape": tf.constant(dense_shape).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.IndexedSlices"] = tf_IndexedSlices_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.IndexedSlices' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.IndexedSlices'.")

check_valid('tf.IndexedSlices', generated_inputs['tf.IndexedSlices'], lib="tf", suffix=0)
