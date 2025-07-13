
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_cross_hashed_inputs():
    list_of_inputs = []

    # Input 1
    inputs = [
        tf.sparse.SparseTensor(indices=[[0, 0], [1, 0]], values=np.array(["a", "b"]).astype(np.string_), dense_shape=[2, 1]),
        tf.constant(np.array([["c"], ["d"]]).astype(np.string_))
    ]
    num_buckets = 10
    hash_key = 123
    name = "sparse_cross_hashed_1"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    inputs = [
        tf.sparse.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [1, 1]], values=np.array(["a", "b", "c", "d"]).astype(np.string_), dense_shape=[2, 2]),
        tf.constant(np.array([["e", "f"], ["g", "h"]]).astype(np.string_))
    ]
    num_buckets = 0
    hash_key = None
    name = "sparse_cross_hashed_2"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    inputs = [
        tf.sparse.SparseTensor(indices=[[0, 0]], values=np.array(["a"]).astype(np.string_), dense_shape=[1, 1]),
        tf.constant(np.array([["b"]]).astype(np.string_))
    ]
    num_buckets = 100
    hash_key = 456
    name = "sparse_cross_hashed_3"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    inputs = [
        tf.sparse.SparseTensor(indices=[[0, 0], [1, 1]], values=np.array(["a", "b"]).astype(np.string_), dense_shape=[2, 2]),
        tf.constant(np.array([["c", "d"], ["e", "f"]]).astype(np.string_)),
    ]
    num_buckets = 1
    hash_key = 789
    name = "sparse_cross_hashed_4"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    inputs = [
        tf.sparse.SparseTensor(indices=[[0, 0]], values=np.array(["test"]).astype(np.string_), dense_shape=[1, 1])
    ]
    num_buckets = 5
    hash_key = 987
    name = "sparse_cross_hashed_5"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    inputs = [
        tf.constant(np.array([["a", "b"], ["c", "d"]]).astype(np.string_))
    ]
    num_buckets = 1234
    hash_key = 654
    name = "sparse_cross_hashed_6"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    inputs = [
        tf.sparse.SparseTensor(indices=[[0, 0], [1, 0], [0, 1], [1, 1]], values=np.array(["a", "b", "c", "d"]).astype(np.string_), dense_shape=[2, 2]),
        tf.constant(np.array([["e", "f"], ["g", "h"]]).astype(np.string_)),
    ]
    num_buckets = 2
    hash_key = 321
    name = "sparse_cross_hashed_7"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    inputs = [
        tf.sparse.SparseTensor(indices=[[0, 0], [0, 1]], values=np.array(["x", "y"]).astype(np.string_), dense_shape=[1, 2])
    ]
    num_buckets = 7
    hash_key = 444
    name = "sparse_cross_hashed_8"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    inputs = [
        tf.sparse.SparseTensor(indices=[[0, 0]], values=np.array(["z"]).astype(np.string_), dense_shape=[1, 1]),
        tf.constant(np.array([["w"]]).astype(np.string_))
    ]
    num_buckets = 13
    hash_key = 555
    name = "sparse_cross_hashed_9"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    inputs = [
        tf.sparse.SparseTensor(indices=[[0, 0], [1, 0]], values=np.array(["p", "q"]).astype(np.string_), dense_shape=[2, 1]),
        tf.constant(np.array([["r"], ["s"]]).astype(np.string_)),
    ]
    num_buckets = 17
    hash_key = 777
    name = "sparse_cross_hashed_10"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.cross_hashed"] = tf_sparse_cross_hashed_inputs()
for i in range(len(generated_inputs["tf.sparse.cross_hashed"])):
  generated_inputs["tf.sparse.cross_hashed"][i]["inputs"] = [tf.convert_to_tensor(x) if not isinstance(x, tf.SparseTensor) else x for x in generated_inputs["tf.sparse.cross_hashed"][i]["inputs"]]

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.cross_hashed' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.cross_hashed'.")

check_valid('tf.sparse.cross_hashed', generated_inputs['tf.sparse.cross_hashed'], lib="tf", suffix=0)
