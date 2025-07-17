
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_reduce_sum_inputs():
    list_of_inputs = []

    def get_sparse_tensor(indices, values, dense_shape):
        return tf.SparseTensor(indices, values, dense_shape)

    def get_tensor_size(tensor):
        return np.prod(tensor.dense_shape.numpy())

    def get_numpy_values(sparse_tensor):
        return sparse_tensor.values.numpy()

    # Input 1
    indices = np.array([[0, 0], [0, 2], [1, 1]], dtype=np.int64)
    values = np.array([1, 1, 1], dtype=np.int32)
    dense_shape = np.array([2, 3], dtype=np.int64)
    sp_input = get_sparse_tensor(indices, values, dense_shape)
    axis = None
    keepdims = False
    output_is_sparse = False
    name = "sparse_reduce_sum_1"
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    indices = np.array([[0, 0], [0, 2], [1, 1]], dtype=np.int64)
    values = np.array([1, 1, 1], dtype=np.int32)
    dense_shape = np.array([2, 3], dtype=np.int64)
    sp_input = get_sparse_tensor(indices, values, dense_shape)
    axis = [0]
    keepdims = False
    output_is_sparse = False
    name = "sparse_reduce_sum_2"
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    indices = np.array([[0, 0], [0, 2], [1, 1]], dtype=np.int64)
    values = np.array([1, 1, 1], dtype=np.int32)
    dense_shape = np.array([2, 3], dtype=np.int64)
    sp_input = get_sparse_tensor(indices, values, dense_shape)
    axis = [1]
    keepdims = False
    output_is_sparse = False
    name = "sparse_reduce_sum_3"
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    indices = np.array([[0, 0], [0, 2], [1, 1]], dtype=np.int64)
    values = np.array([1, 1, 1], dtype=np.int32)
    dense_shape = np.array([2, 3], dtype=np.int64)
    sp_input = get_sparse_tensor(indices, values, dense_shape)
    axis = [1]
    keepdims = True
    output_is_sparse = False
    name = "sparse_reduce_sum_4"
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    indices = np.array([[0, 0], [0, 2], [1, 1]], dtype=np.int64)
    values = np.array([1, 1, 1], dtype=np.int32)
    dense_shape = np.array([2, 3], dtype=np.int64)
    sp_input = get_sparse_tensor(indices, values, dense_shape)
    axis = [0, 1]
    keepdims = False
    output_is_sparse = False
    name = "sparse_reduce_sum_5"
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    indices = np.array([[0, 0], [0, 2], [1, 1]], dtype=np.int64)
    values = np.array([1, 1, 1], dtype=np.int32)
    dense_shape = np.array([2, 3], dtype=np.int64)
    sp_input = get_sparse_tensor(indices, values, dense_shape)
    axis = [0]
    keepdims = True
    output_is_sparse = True
    name = "sparse_reduce_sum_6"
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative axis
    indices = np.array([[0, 0], [0, 2], [1, 1]], dtype=np.int64)
    values = np.array([1, 1, 1], dtype=np.int32)
    dense_shape = np.array([2, 3], dtype=np.int64)
    sp_input = get_sparse_tensor(indices, values, dense_shape)
    axis = [-1]
    keepdims = False
    output_is_sparse = False
    name = "sparse_reduce_sum_7"
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D SparseTensor
    indices = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.int64)
    values = np.array([1, 2, 3, 4], dtype=np.int32)
    dense_shape = np.array([2, 2, 2], dtype=np.int64)
    sp_input = get_sparse_tensor(indices, values, dense_shape)
    axis = [0, 1]
    keepdims = False
    output_is_sparse = False
    name = "sparse_reduce_sum_8"
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D SparseTensor with keepdims
    indices = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.int64)
    values = np.array([1, 2, 3, 4], dtype=np.int32)
    dense_shape = np.array([2, 2, 2], dtype=np.int64)
    sp_input = get_sparse_tensor(indices, values, dense_shape)
    axis = [0, 1]
    keepdims = True
    output_is_sparse = False
    name = "sparse_reduce_sum_9"
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32 SparseTensor
    indices = np.array([[0, 0], [0, 2], [1, 1]], dtype=np.int64)
    values = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    dense_shape = np.array([2, 3], dtype=np.int64)
    sp_input = get_sparse_tensor(indices, values, dense_shape)
    axis = [1]
    keepdims = False
    output_is_sparse = False
    name = "sparse_reduce_sum_10"
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: All zeros
    indices = np.array([[0, 0], [0, 2], [1, 1]], dtype=np.int64)
    values = np.array([0, 0, 0], dtype=np.int32)
    dense_shape = np.array([2, 3], dtype=np.int64)
    sp_input = get_sparse_tensor(indices, values, dense_shape)
    axis = [0]
    keepdims = False
    output_is_sparse = False
    name = "sparse_reduce_sum_11"
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Empty SparseTensor
    indices = np.array([], dtype=np.int64).reshape(0, 2)
    values = np.array([], dtype=np.int32)
    dense_shape = np.array([2, 3], dtype=np.int64)
    sp_input = get_sparse_tensor(indices, values, dense_shape)
    axis = [0]
    keepdims = False
    output_is_sparse = False
    name = "sparse_reduce_sum_12"
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13: Reduce all dimensions
    indices = np.array([[0, 0], [0, 2], [1, 1]], dtype=np.int64)
    values = np.array([1, 1, 1], dtype=np.int32)
    dense_shape = np.array([2, 3], dtype=np.int64)
    sp_input = get_sparse_tensor(indices, values, dense_shape)
    axis = None
    keepdims = False
    output_is_sparse = False
    name = "sparse_reduce_sum_13"
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 14: Reduce all dimensions with keepdims
    indices = np.array([[0, 0], [0, 2], [1, 1]], dtype=np.int64)
    values = np.array([1, 1, 1], dtype=np.int32)
    dense_shape = np.array([2, 3], dtype=np.int64)
    sp_input = get_sparse_tensor(indices, values, dense_shape)
    axis = None
    keepdims = True
    output_is_sparse = False
    name = "sparse_reduce_sum_14"
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 15: Different dtype values
    indices = np.array([[0, 0], [0, 2], [1, 1]], dtype=np.int64)
    values = np.array([1, 1, 1], dtype=np.int64)
    dense_shape = np.array([2, 3], dtype=np.int64)
    sp_input = get_sparse_tensor(indices, values, dense_shape)
    axis = [0]
    keepdims = False
    output_is_sparse = False
    name = "sparse_reduce_sum_15"
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    values_np = get_numpy_values(sp_input)
    input_dict = {"sp_input": sp_input, "axis": axis, "keepdims": keepdims, "output_is_sparse": output_is_sparse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.reduce_sum"] = tf_sparse_reduce_sum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.reduce_sum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.reduce_sum'.")

check_valid('tf.sparse.reduce_sum', generated_inputs['tf.sparse.reduce_sum'], lib="tf", suffix=0)
