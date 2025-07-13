
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_sparse_concat_inputs():
    list_of_inputs = []

    # Input 1: Basic valid case
    indices = [np.array([[0, 0], [1, 2]], dtype=np.int64), np.array([[0, 1], [1, 0]], dtype=np.int64)]
    values = [np.array([1, 2]), np.array([3, 4])]
    shapes = [np.array([2, 3], dtype=np.int64), np.array([2, 3], dtype=np.int64)]
    concat_dim = 1
    name = "concat_example_1"
    input_dict = {'indices': indices, 'values': values, 'shapes': shapes, 'concat_dim': concat_dim, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: concat_dim = 0
    indices = [np.array([[0, 0], [0, 1]], dtype=np.int64), np.array([[0, 0], [0, 1]], dtype=np.int64)]
    values = [np.array([1, 2]), np.array([3, 4])]
    shapes = [np.array([1, 2], dtype=np.int64), np.array([1, 2], dtype=np.int64)]
    concat_dim = 0
    name = "concat_example_2"
    input_dict = {'indices': indices, 'values': values, 'shapes': shapes, 'concat_dim': concat_dim, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: More tensors to concatenate
    indices = [np.array([[0, 0]], dtype=np.int64), np.array([[0, 0]], dtype=np.int64), np.array([[0, 0]], dtype=np.int64)]
    values = [np.array([1]), np.array([2]), np.array([3])]
    shapes = [np.array([1, 1], dtype=np.int64), np.array([1, 1], dtype=np.int64), np.array([1, 1], dtype=np.int64)]
    concat_dim = 1
    name = "concat_example_3"
    input_dict = {'indices': indices, 'values': values, 'shapes': shapes, 'concat_dim': concat_dim, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different shapes along concat_dim
    indices = [np.array([[0, 0]], dtype=np.int64), np.array([[0, 0]], dtype=np.int64)]
    values = [np.array([1]), np.array([2])]
    shapes = [np.array([1, 2], dtype=np.int64), np.array([1, 3], dtype=np.int64)]
    concat_dim = 1
    name = "concat_example_4"
    input_dict = {'indices': indices, 'values': values, 'shapes': shapes, 'concat_dim': concat_dim, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5: Negative concat_dim (equivalent to -rank)
    indices = [np.array([[0, 0], [1, 2]], dtype=np.int64), np.array([[0, 1], [1, 0]], dtype=np.int64)]
    values = [np.array([1, 2]), np.array([3, 4])]
    shapes = [np.array([2, 3], dtype=np.int64), np.array([2, 3], dtype=np.int64)]
    concat_dim = -1
    name = "concat_example_5"
    input_dict = {'indices': indices, 'values': values, 'shapes': shapes, 'concat_dim': concat_dim, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float32 values
    indices = [np.array([[0, 0]], dtype=np.int64), np.array([[0, 0]], dtype=np.int64)]
    values = [np.array([1.0], dtype=np.float32), np.array([2.0], dtype=np.float32)]
    shapes = [np.array([1, 1], dtype=np.int64), np.array([1, 1], dtype=np.int64)]
    concat_dim = 1
    name = "concat_example_6"
    input_dict = {'indices': indices, 'values': values, 'shapes': shapes, 'concat_dim': concat_dim, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int64 values
    indices = [np.array([[0, 0]], dtype=np.int64), np.array([[0, 0]], dtype=np.int64)]
    values = [np.array([1], dtype=np.int64), np.array([2], dtype=np.int64)]
    shapes = [np.array([1, 1], dtype=np.int64), np.array([1, 1], dtype=np.int64)]
    concat_dim = 1
    name = "concat_example_7"
    input_dict = {'indices': indices, 'values': values, 'shapes': shapes, 'concat_dim': concat_dim, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: bool values
    indices = [np.array([[0, 0]], dtype=np.int64), np.array([[0, 0]], dtype=np.int64)]
    values = [np.array([True], dtype=np.bool_), np.array([False], dtype=np.bool_)]
    shapes = [np.array([1, 1], dtype=np.int64), np.array([1, 1], dtype=np.int64)]
    concat_dim = 1
    name = "concat_example_8"
    input_dict = {'indices': indices, 'values': values, 'shapes': shapes, 'concat_dim': concat_dim, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: string values
    indices = [np.array([[0, 0]], dtype=np.int64), np.array([[0, 0]], dtype=np.int64)]
    values = [np.array(["a"]), np.array(["b"])]
    shapes = [np.array([1, 1], dtype=np.int64), np.array([1, 1], dtype=np.int64)]
    concat_dim = 1
    name = "concat_example_9"
    input_dict = {'indices': indices, 'values': values, 'shapes': shapes, 'concat_dim': concat_dim, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int32 values
    indices = [np.array([[0, 0]], dtype=np.int64), np.array([[0, 0]], dtype=np.int64)]
    values = [np.array([1], dtype=np.int32), np.array([2], dtype=np.int32)]
    shapes = [np.array([1, 1], dtype=np.int64), np.array([1, 1], dtype=np.int64)]
    concat_dim = 1
    name = "concat_example_10"
    input_dict = {'indices': indices, 'values': values, 'shapes': shapes, 'concat_dim': concat_dim, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseConcat"] = tf_raw_ops_sparse_concat_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseConcat' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseConcat'.")

check_valid('tf.raw_ops.SparseConcat', generated_inputs['tf.raw_ops.SparseConcat'], lib="tf", suffix=0)
