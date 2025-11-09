
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_IndexedSlices_2_inputs():
    list_of_inputs = []

    values = np.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]], dtype=np.float32)
    indices = np.array([1, 3], dtype=np.int32)
    dense_shape = [10, 3]
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    values = np.arange(12, dtype=np.float64).reshape(3, 2, 2)
    indices = np.array([0, 2, 6], dtype=np.int32)
    dense_shape = [7, 2, 2]
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    values = np.array([[10, -1, 5, 3]], dtype=np.int32)
    indices = np.array([3], dtype=np.int32)
    dense_shape = [4, 4]
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    values = np.array([True, False, True], dtype=bool)
    indices = np.array([0, 2, 4], dtype=np.int64)
    dense_shape = [5]
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    values = np.empty((0, 5), dtype=np.float32)
    indices = np.empty((0,), dtype=np.int32)
    dense_shape = [8, 5]
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    values = np.empty((2, 0, 3), dtype=np.float32)
    indices = np.array([2, 5], dtype=np.int32)
    dense_shape = [6, 0, 3]
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    values = np.empty((2, 2, 0, 1), dtype=np.float16)
    indices = np.array([0, 1], dtype=np.int64)
    dense_shape = [9, 2, 0, 1]
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    values = np.array([[-3, 7],
                       [8, -2],
                       [1, 1]], dtype=np.int16)
    indices = np.array([1, 1, 2], dtype=np.int32)
    dense_shape = [4, 2]
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    values = np.array([[[ -1.0]],
                       [[ -2.5]],
                       [[  0.0]],
                       [[  3.14]],
                       [[ -7.2]]], dtype=np.float32)
    indices = np.array([9, 2, 7, 3, 5], dtype=np.int32)
    dense_shape = [10, 1, 1]
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    values = (np.arange(18).reshape(2, 3, 3).astype(np.float32) + 1j * np.arange(18).reshape(2, 3, 3).astype(np.float32)).astype(np.complex64)
    indices = np.array([4, 1], dtype=np.int32)
    dense_shape = [50, 3, 3]
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    values = np.array([[[1, 2],
                        [3, 4]],
                       [[5, 6],
                        [7, 8]],
                       [[9, 10],
                        [11, 12]],
                       [[13, 14],
                        [15, 16]]], dtype=np.uint8)
    indices = np.array([0, 1, 3, 4], dtype=np.int32)
    dense_shape = [5, 2, 2]
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    values = np.empty((0,), dtype=np.float64)
    indices = np.empty((0,), dtype=np.int32)
    dense_shape = [10]
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    return list_of_inputs

generated_inputs["tf.IndexedSlices_2"] = tf_IndexedSlices_2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.IndexedSlices_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.IndexedSlices_2'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.IndexedSlices', generated_inputs['tf.IndexedSlices_2'], lib="tf", suffix=2)
