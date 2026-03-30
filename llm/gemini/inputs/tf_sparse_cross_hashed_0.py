
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_cross_hashed_inputs():
    list_of_inputs = []

    # Input 1: Basic example with sparse tensors
    indices1 = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.int64)
    values1 = np.array(["a", "b", "c"], dtype=np.string_)
    shape1 = np.array([2, 2], dtype=np.int64)
    st1 = tf.SparseTensor(indices1, values1, shape1)

    indices2 = np.array([[0, 0], [1, 0]], dtype=np.int64)
    values2 = np.array(["d", "e"], dtype=np.string_)
    shape2 = np.array([2, 1], dtype=np.int64)
    st2 = tf.SparseTensor(indices2, values2, shape2)

    dense = np.array([["f"], ["g"]], dtype=np.string_)
    inputs = [st1, st2, dense]
    num_buckets = 100
    hash_key = 12345
    name = "sparse_cross_1"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: No num_buckets (num_buckets=0)
    indices1 = np.array([[0, 0], [1, 0]], dtype=np.int64)
    values1 = np.array(["a", "b"], dtype=np.string_)
    shape1 = np.array([2, 1], dtype=np.int64)
    st1 = tf.SparseTensor(indices1, values1, shape1)
    dense = np.array([["f"], ["g"]], dtype=np.string_)
    inputs = [st1, dense]
    num_buckets = 0
    hash_key = 54321
    name = "sparse_cross_2"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different hash_key
    indices1 = np.array([[0, 0]], dtype=np.int64)
    values1 = np.array(["a"], dtype=np.string_)
    shape1 = np.array([1, 1], dtype=np.int64)
    st1 = tf.SparseTensor(indices1, values1, shape1)
    dense = np.array([["f"]], dtype=np.string_)
    inputs = [st1, dense]
    num_buckets = 50
    hash_key = 98765
    name = "sparse_cross_3"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Only sparse tensor input
    indices1 = np.array([[0, 0], [1, 0]], dtype=np.int64)
    values1 = np.array(["a", "b"], dtype=np.string_)
    shape1 = np.array([2, 1], dtype=np.int64)
    st1 = tf.SparseTensor(indices1, values1, shape1)
    inputs = [st1]
    num_buckets = 10
    hash_key = 11223
    name = "sparse_cross_4"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Only dense tensor input
    dense = np.array([["f"], ["g"]], dtype=np.string_)
    inputs = [dense]
    num_buckets = 20
    hash_key = 44556
    name = "sparse_cross_5"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6: Empty sparse tensor
    indices1 = np.array([], dtype=np.int64).reshape(0, 2)
    values1 = np.array([], dtype=np.string_)
    shape1 = np.array([2, 2], dtype=np.int64)
    st1 = tf.SparseTensor(indices1, values1, shape1)
    dense = np.array([["f"], ["g"]], dtype=np.string_)

    inputs = [st1, dense]
    num_buckets = 30
    hash_key = 77889
    name = "sparse_cross_6"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Removing inputs which might cause issues with shape inference. Removing 2D empty array and replacing with scalar one
    # dense = np.array([[]], dtype=np.string_)
    # inputs = [dense]
    # num_buckets = 40
    # hash_key = 33445
    # name = "sparse_cross_7"
    # input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    # list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Three sparse tensors
    indices1 = np.array([[0, 0]], dtype=np.int64)
    values1 = np.array(["a"], dtype=np.string_)
    shape1 = np.array([1, 1], dtype=np.int64)
    st1 = tf.SparseTensor(indices1, values1, shape1)

    indices2 = np.array([[0, 0]], dtype=np.int64)
    values2 = np.array(["b"], dtype=np.string_)
    shape2 = np.array([1, 1], dtype=np.int64)
    st2 = tf.SparseTensor(indices2, values2, shape2)

    indices3 = np.array([[0, 0]], dtype=np.int64)
    values3 = np.array(["c"], dtype=np.string_)
    shape3 = np.array([1, 1], dtype=np.int64)
    st3 = tf.SparseTensor(indices3, values3, shape3)

    inputs = [st1, st2, st3]
    num_buckets = 60
    hash_key = 66778
    name = "sparse_cross_8"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Three dense tensors
    dense1 = np.array([["a"]], dtype=np.string_)
    dense2 = np.array([["b"]], dtype=np.string_)
    dense3 = np.array([["c"]], dtype=np.string_)
    inputs = [dense1, dense2, dense3]
    num_buckets = 70
    hash_key = 88990
    name = "sparse_cross_9"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mix of 2 Sparse and 2 Dense tensors
    indices1 = np.array([[0, 0]], dtype=np.int64)
    values1 = np.array(["a"], dtype=np.string_)
    shape1 = np.array([1, 1], dtype=np.int64)
    st1 = tf.SparseTensor(indices1, values1, shape1)

    indices2 = np.array([[0, 0]], dtype=np.int64)
    values2 = np.array(["b"], dtype=np.string_)
    shape2 = np.array([1, 1], dtype=np.int64)
    st2 = tf.SparseTensor(indices2, values2, shape2)

    dense1 = np.array([["c"]], dtype=np.string_)
    dense2 = np.array([["d"]], dtype=np.string_)

    inputs = [st1, st2, dense1, dense2]
    num_buckets = 80
    hash_key = 23456
    name = "sparse_cross_10"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.cross_hashed"] = tf_sparse_cross_hashed_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.cross_hashed' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.cross_hashed'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.sparse.cross_hashed', generated_inputs['tf.sparse.cross_hashed'], lib="tf", suffix=0)
