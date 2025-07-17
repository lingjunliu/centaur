
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_concat_inputs():
    list_of_inputs = []

    # Input 1: Basic case with axis=0
    indices1 = np.array([[0, 0], [1, 2]])
    values1 = np.array([1, 2])
    shape1 = np.array([2, 3])
    st1 = tf.SparseTensor(indices1, values1, shape1)

    indices2 = np.array([[0, 1], [1, 0]])
    values2 = np.array([3, 4])
    shape2 = np.array([2, 3])
    st2 = tf.SparseTensor(indices2, values2, shape2)

    input_dict = {"axis": 0, "sp_inputs": [st1, st2], "expand_nonconcat_dims": False, "name": "concat_0"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic case with axis=1
    indices1 = np.array([[0, 0], [1, 2]])
    values1 = np.array([1, 2])
    shape1 = np.array([2, 3])
    st1 = tf.SparseTensor(indices1, values1, shape1)

    indices2 = np.array([[0, 1], [1, 0]])
    values2 = np.array([3, 4])
    shape2 = np.array([2, 3])
    st2 = tf.SparseTensor(indices2, values2, shape2)

    input_dict = {"axis": 1, "sp_inputs": [st1, st2], "expand_nonconcat_dims": False, "name": "concat_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: expand_nonconcat_dims=True
    indices1 = np.array([[0, 0], [1, 2]])
    values1 = np.array([1, 2])
    shape1 = np.array([2, 3])
    st1 = tf.SparseTensor(indices1, values1, shape1)

    indices2 = np.array([[0, 1], [1, 0]])
    values2 = np.array([3, 4])
    shape2 = np.array([3, 3])
    st2 = tf.SparseTensor(indices2, values2, shape2)

    input_dict = {"axis": 0, "sp_inputs": [st1, st2], "expand_nonconcat_dims": True, "name": "concat_expand"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different data types
    indices1 = np.array([[0, 0], [1, 2]])
    values1 = np.array([1.0, 2.0], dtype=np.float32)
    shape1 = np.array([2, 3])
    st1 = tf.SparseTensor(indices1, values1, shape1)

    indices2 = np.array([[0, 1], [1, 0]])
    values2 = np.array([3.0, 4.0], dtype=np.float32)
    shape2 = np.array([2, 3])
    st2 = tf.SparseTensor(indices2, values2, shape2)

    input_dict = {"axis": 0, "sp_inputs": [st1, st2], "expand_nonconcat_dims": False, "name": "concat_float"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D SparseTensors
    indices1 = np.array([[0, 0, 0], [1, 1, 1]])
    values1 = np.array([1, 2])
    shape1 = np.array([2, 2, 2])
    st1 = tf.SparseTensor(indices1, values1, shape1)

    indices2 = np.array([[0, 1, 0], [1, 0, 1]])
    values2 = np.array([3, 4])
    shape2 = np.array([2, 2, 2])
    st2 = tf.SparseTensor(indices2, values2, shape2)

    input_dict = {"axis": 0, "sp_inputs": [st1, st2], "expand_nonconcat_dims": False, "name": "concat_3d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: axis=-1 (last axis)
    indices1 = np.array([[0, 0], [1, 2]])
    values1 = np.array([1, 2])
    shape1 = np.array([2, 3])
    st1 = tf.SparseTensor(indices1, values1, shape1)

    indices2 = np.array([[0, 1], [1, 0]])
    values2 = np.array([3, 4])
    shape2 = np.array([2, 3])
    st2 = tf.SparseTensor(indices2, values2, shape2)

    input_dict = {"axis": -1, "sp_inputs": [st1, st2], "expand_nonconcat_dims": False, "name": "concat_-1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: String values
    indices1 = np.array([[0, 0], [1, 2]])
    values1 = np.array(["a", "b"])
    shape1 = np.array([2, 3])
    st1 = tf.SparseTensor(indices1, values1, shape1)

    indices2 = np.array([[0, 1], [1, 0]])
    values2 = np.array(["c", "d"])
    shape2 = np.array([2, 3])
    st2 = tf.SparseTensor(indices2, values2, shape2)

    input_dict = {"axis": 0, "sp_inputs": [st1, st2], "expand_nonconcat_dims": False, "name": "concat_string"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: One empty SparseTensor
    indices1 = np.array([[0, 0], [1, 2]])
    values1 = np.array([1, 2])
    shape1 = np.array([2, 3])
    st1 = tf.SparseTensor(indices1, values1, shape1)

    indices2 = np.array([], dtype=np.int64).reshape(0, 2)
    values2 = np.array([])
    shape2 = np.array([2, 3])
    st2 = tf.SparseTensor(indices2, values2, shape2)

    input_dict = {"axis": 0, "sp_inputs": [st1, st2], "expand_nonconcat_dims": False, "name": "concat_empty"}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9: concat axis with different rank
    indices1 = np.array([[0, 0], [1, 2]])
    values1 = np.array([1, 2])
    shape1 = np.array([2, 3])
    st1 = tf.SparseTensor(indices1, values1, shape1)

    indices2 = np.array([[0, 1], [1, 0]])
    values2 = np.array([3, 4])
    shape2 = np.array([3, 3])
    st2 = tf.SparseTensor(indices2, values2, shape2)
    input_dict = {"axis": 0, "sp_inputs": [st1, st2], "expand_nonconcat_dims": True, "name": "concat_diff_rank"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Negative axis with expand nonconcat
    indices1 = np.array([[0, 0], [1, 2]])
    values1 = np.array([1, 2])
    shape1 = np.array([2, 3])
    st1 = tf.SparseTensor(indices1, values1, shape1)

    indices2 = np.array([[0, 1], [1, 0]])
    values2 = np.array([3, 4])
    shape2 = np.array([3, 3])
    st2 = tf.SparseTensor(indices2, values2, shape2)

    input_dict = {"axis": -1, "sp_inputs": [st1, st2], "expand_nonconcat_dims": True, "name": "concat_neg_axis"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Empty inputs
    input_dict = {"axis": 0, "sp_inputs": [], "expand_nonconcat_dims": False, "name": "concat_empty_inputs"}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.concat"] = tf_sparse_concat_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.concat' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.concat'.")

check_valid('tf.sparse.concat', generated_inputs['tf.sparse.concat'], lib="tf", suffix=0)
