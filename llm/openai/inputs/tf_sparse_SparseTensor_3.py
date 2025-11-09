
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import torch  # ensuring availability if required by environment

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)
np.random.seed(42)

def tf_sparse_SparseTensor_inputs():
    list_of_inputs = []

    # Input 1: 2D int values
    indices = [[np.int64(0), np.int64(0)], [np.int64(1), np.int64(2)]]
    values = [np.int32(1), np.int32(2)]
    dense_shape = [np.int64(3), np.int64(4)]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float32 values with negatives
    indices = [[np.int64(0)], [np.int64(4)], [np.int64(9)]]
    values = [np.float32(1.5), np.float32(-2.0), np.float32(3.14)]
    dense_shape = [np.int64(10)]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D boolean values
    indices = [
        [np.int64(0), np.int64(1), np.int64(2)],
        [np.int64(1), np.int64(0), np.int64(3)],
        [np.int64(1), np.int64(2), np.int64(1)]
    ]
    values = [np.bool_(True), np.bool_(False), np.bool_(True)]
    dense_shape = [np.int64(2), np.int64(3), np.int64(4)]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D float32 values
    indices = [
        [np.int64(0), np.int64(0), np.int64(0), np.int64(0)],
        [np.int64(1), np.int64(1), np.int64(1), np.int64(1)],
        [np.int64(0), np.int64(1), np.int64(1), np.int64(0)]
    ]
    values = [np.float32(2.5), np.float32(-3.0), np.float32(0.75)]
    dense_shape = [np.int64(2), np.int64(2), np.int64(2), np.int64(2)]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float16 values, unsorted indices
    indices = [
        [np.int64(2), np.int64(2)],
        [np.int64(0), np.int64(1)],
        [np.int64(1), np.int64(0)]
    ]
    values = [np.float16(0.1), np.float16(-0.2), np.float16(1.5)]
    dense_shape = [np.int64(3), np.int64(3)]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D int64 values with negatives
    indices = [
        [np.int64(0), np.int64(0), np.int64(0)],
        [np.int64(2), np.int64(2), np.int64(2)],
        [np.int64(1), np.int64(1), np.int64(1)],
        [np.int64(0), np.int64(2), np.int64(1)]
    ]
    values = [np.int64(-1), np.int64(0), np.int64(2), np.int64(-5)]
    dense_shape = [np.int64(3), np.int64(3), np.int64(3)]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D boolean values
    indices = [[np.int64(1)], [np.int64(3)]]
    values = [np.bool_(True), np.bool_(False)]
    dense_shape = [np.int64(5)]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float64 values
    indices = [
        [np.int64(0), np.int64(0), np.int64(0)],
        [np.int64(0), np.int64(1), np.int64(2)],
        [np.int64(1), np.int64(1), np.int64(1)]
    ]
    values = [np.float64(0.0), np.float64(4.2), np.float64(5.5)]
    dense_shape = [np.int64(2), np.int64(2), np.int64(3)]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D int16 values
    indices = [
        [np.int64(3), np.int64(2), np.int64(1), np.int64(4)],
        [np.int64(0), np.int64(0), np.int64(0), np.int64(0)],
        [np.int64(2), np.int64(1), np.int64(0), np.int64(3)]
    ]
    values = [np.int16(7), np.int16(-8), np.int16(15)]
    dense_shape = [np.int64(4), np.int64(3), np.int64(2), np.int64(5)]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D uint8 values
    indices = [
        [np.int64(0), np.int64(0)],
        [np.int64(0), np.int64(1)],
        [np.int64(0), np.int64(3)],
        [np.int64(1), np.int64(2)]
    ]
    values = [np.uint8(255), np.uint8(0), np.uint8(128), np.uint8(64)]
    dense_shape = [np.int64(2), np.int64(4)]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 5D float32 values
    indices = [
        [np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0)],
        [np.int64(0), np.int64(1), np.int64(0), np.int64(1), np.int64(0)]
    ]
    values = [np.float32(0.25), np.float32(-1.75)]
    dense_shape = [np.int64(1), np.int64(2), np.int64(1), np.int64(2), np.int64(1)]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.sparse.SparseTensor_3"] = tf_sparse_SparseTensor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.SparseTensor_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.SparseTensor_3'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.sparse.SparseTensor', generated_inputs['tf.sparse.SparseTensor_3'], lib="tf", suffix=3)
