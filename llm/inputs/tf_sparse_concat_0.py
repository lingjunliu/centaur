
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_concat_inputs():
    list_of_inputs = []

    # Input 1
    indices1 = np.array([[0, 2], [1, 0], [1, 1]], dtype=np.int64)
    values1 = np.array(["a", "b", "c"]).astype(np.unicode_)
    shape1 = np.array([2, 3], dtype=np.int64)
    sp_input1 = tf.sparse.SparseTensor(indices1, values1, shape1)

    indices2 = np.array([[0, 1], [0, 2]], dtype=np.int64)
    values2 = np.array(["d", "e"]).astype(np.unicode_)
    shape2 = np.array([2, 4], dtype=np.int64)
    sp_input2 = tf.sparse.SparseTensor(indices2, values2, shape2)
    axis = np.int32(1)
    sp_inputs = [sp_input1, sp_input2]
    expand_nonconcat_dims = False
    name = "concat_sparse"

    input_dict = {
        "axis": axis,
        "sp_inputs": sp_inputs,
        "expand_nonconcat_dims": expand_nonconcat_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    indices1 = np.array([[0, 2], [1, 0], [2, 1]], dtype=np.int64)
    values1 = np.array(["a", "b", "c"]).astype(np.unicode_)
    shape1 = np.array([3, 3], dtype=np.int64)
    sp_input1 = tf.sparse.SparseTensor(indices1, values1, shape1)

    indices2 = np.array([[0, 1], [0, 2]], dtype=np.int64)
    values2 = np.array(["d", "e"]).astype(np.unicode_)
    shape2 = np.array([2, 4], dtype=np.int64)
    sp_input2 = tf.sparse.SparseTensor(indices2, values2, shape2)
    axis = np.int32(1)
    sp_inputs = [sp_input1, sp_input2]
    expand_nonconcat_dims = True
    name = None

    input_dict = {
        "axis": axis,
        "sp_inputs": sp_inputs,
        "expand_nonconcat_dims": expand_nonconcat_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different axis
    indices1 = np.array([[0, 2], [1, 0], [1, 1]], dtype=np.int64)
    values1 = np.array(["a", "b", "c"]).astype(np.unicode_)
    shape1 = np.array([2, 3], dtype=np.int64)
    sp_input1 = tf.sparse.SparseTensor(indices1, values1, shape1)

    indices2 = np.array([[0, 1], [1, 2]], dtype=np.int64)
    values2 = np.array(["d", "e"]).astype(np.unicode_)
    shape2 = np.array([2, 3], dtype=np.int64)
    sp_input2 = tf.sparse.SparseTensor(indices2, values2, shape2)
    axis = np.int32(0)
    sp_inputs = [sp_input1, sp_input2]
    expand_nonconcat_dims = False
    name = "concat_sparse_0"

    input_dict = {
        "axis": axis,
        "sp_inputs": sp_inputs,
        "expand_nonconcat_dims": expand_nonconcat_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Three inputs
    indices1 = np.array([[0, 2], [1, 0]], dtype=np.int64)
    values1 = np.array(["a", "b"]).astype(np.unicode_)
    shape1 = np.array([2, 3], dtype=np.int64)
    sp_input1 = tf.sparse.SparseTensor(indices1, values1, shape1)

    indices2 = np.array([[0, 1]], dtype=np.int64)
    values2 = np.array(["d"]).astype(np.unicode_)
    shape2 = np.array([2, 4], dtype=np.int64)
    sp_input2 = tf.sparse.SparseTensor(indices2, values2, shape2)

    indices3 = np.array([[1, 2]], dtype=np.int64)
    values3 = np.array(["f"]).astype(np.unicode_)
    shape3 = np.array([2, 2], dtype=np.int64)
    sp_input3 = tf.sparse.SparseTensor(indices3, values3, shape3)

    axis = np.int32(1)
    sp_inputs = [sp_input1, sp_input2, sp_input3]
    expand_nonconcat_dims = True
    name = "concat_sparse_3"

    input_dict = {
        "axis": axis,
        "sp_inputs": sp_inputs,
        "expand_nonconcat_dims": expand_nonconcat_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty SparseTensors
    indices1 = np.array([], dtype=np.int64).reshape(0, 2)
    values1 = np.array([]).astype(np.unicode_)
    shape1 = np.array([2, 3], dtype=np.int64)
    sp_input1 = tf.sparse.SparseTensor(indices1, values1, shape1)

    indices2 = np.array([], dtype=np.int64).reshape(0, 2)
    values2 = np.array([]).astype(np.unicode_)
    shape2 = np.array([2, 4], dtype=np.int64)
    sp_input2 = tf.sparse.SparseTensor(indices2, values2, shape2)

    axis = np.int32(1)
    sp_inputs = [sp_input1, sp_input2]
    expand_nonconcat_dims = False
    name = "concat_empty"

    input_dict = {
        "axis": axis,
        "sp_inputs": sp_inputs,
        "expand_nonconcat_dims": expand_nonconcat_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7: axis = -1
    indices1 = np.array([[0, 2], [1, 0]], dtype=np.int64)
    values1 = np.array(["a", "b"]).astype(np.unicode_)
    shape1 = np.array([2, 3], dtype=np.int64)
    sp_input1 = tf.sparse.SparseTensor(indices1, values1, shape1)

    indices2 = np.array([[0, 1]], dtype=np.int64)
    values2 = np.array(["d"]).astype(np.unicode_)
    shape2 = np.array([2, 4], dtype=np.int64)
    sp_input2 = tf.sparse.SparseTensor(indices2, values2, shape2)

    axis = np.int32(-1)
    sp_inputs = [sp_input1, sp_input2]
    expand_nonconcat_dims = True
    name = "concat_sparse_neg"

    input_dict = {
        "axis": axis,
        "sp_inputs": sp_inputs,
        "expand_nonconcat_dims": expand_nonconcat_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Integer values
    indices1 = np.array([[0, 2], [1, 0]], dtype=np.int64)
    values1 = np.array([1, 2]).astype(np.int32)
    shape1 = np.array([2, 3], dtype=np.int64)
    sp_input1 = tf.sparse.SparseTensor(indices1, values1, shape1)

    indices2 = np.array([[0, 1]], dtype=np.int64)
    values2 = np.array([3]).astype(np.int32)
    shape2 = np.array([2, 4], dtype=np.int64)
    sp_input2 = tf.sparse.SparseTensor(indices2, values2, shape2)

    axis = np.int32(1)
    sp_inputs = [sp_input1, sp_input2]
    expand_nonconcat_dims = True
    name = "concat_int"

    input_dict = {
        "axis": axis,
        "sp_inputs": sp_inputs,
        "expand_nonconcat_dims": expand_nonconcat_dims,
        "name": name
    }
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
