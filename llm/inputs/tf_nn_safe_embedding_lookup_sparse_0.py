
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_safe_embedding_lookup_sparse_inputs():
    list_of_inputs = []

    def create_sparse_tensor(indices, values, dense_shape):
        return tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

    def create_ragged_tensor(ragged_list):
        return tf.ragged.constant(ragged_list)

    # Input 1
    embedding_weights = [np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)]
    sparse_ids = create_sparse_tensor(indices=[[0, 0], [0, 1], [1, 0]], values=[0, 1, 2], dense_shape=[2, 2])
    sparse_weights = create_sparse_tensor(indices=[[0, 0], [0, 1], [1, 0]], values=[1.0, 2.0, 3.0], dense_shape=[2, 2])
    combiner = "mean"
    default_id = 0
    max_norm = None
    name = "embedding_lookup"
    allow_fast_lookup = False

    input_dict = {
        "embedding_weights": embedding_weights,
        "sparse_ids": sparse_ids,
        "sparse_weights": sparse_weights,
        "combiner": combiner,
        "default_id": default_id,
        "max_norm": max_norm,
        "name": name,
        "allow_fast_lookup": allow_fast_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    embedding_weights = [np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)]
    sparse_ids = create_sparse_tensor(indices=[[0, 0], [0, 1], [1, 0]], values=[0, 1, -1], dense_shape=[2, 2])
    sparse_weights = create_sparse_tensor(indices=[[0, 0], [0, 1], [1, 0]], values=[1.0, 2.0, 3.0], dense_shape=[2, 2])
    combiner = "sum"
    default_id = 0
    max_norm = 1.0
    name = "embedding_lookup_sum"
    allow_fast_lookup = True

    input_dict = {
        "embedding_weights": embedding_weights,
        "sparse_ids": sparse_ids,
        "sparse_weights": sparse_weights,
        "combiner": combiner,
        "default_id": default_id,
        "max_norm": max_norm,
        "name": name,
        "allow_fast_lookup": allow_fast_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    embedding_weights = [np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
                         np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float32)]
    sparse_ids = create_sparse_tensor(indices=[[0, 0], [0, 1], [1, 0]], values=[0, 1, 2], dense_shape=[2, 2])
    sparse_weights = create_sparse_tensor(indices=[[0, 0], [0, 1], [1, 0]], values=[1.0, 2.0, 0.0], dense_shape=[2, 2])
    combiner = "sqrtn"
    default_id = 1
    max_norm = 2.0
    name = "embedding_lookup_sqrtn"
    allow_fast_lookup = False

    input_dict = {
        "embedding_weights": embedding_weights,
        "sparse_ids": sparse_ids,
        "sparse_weights": sparse_weights,
        "combiner": combiner,
        "default_id": default_id,
        "max_norm": max_norm,
        "name": name,
        "allow_fast_lookup": allow_fast_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    embedding_weights = [np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)]
    sparse_ids = create_sparse_tensor(indices=[[0, 0], [1, 1]], values=[0, 2], dense_shape=[2, 2])
    sparse_weights = None
    combiner = "mean"
    default_id = 1
    max_norm = None
    name = "embedding_lookup_none_weights"
    allow_fast_lookup = False

    input_dict = {
        "embedding_weights": embedding_weights,
        "sparse_ids": sparse_ids,
        "sparse_weights": sparse_weights,
        "combiner": combiner,
        "default_id": default_id,
        "max_norm": max_norm,
        "name": name,
        "allow_fast_lookup": allow_fast_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    embedding_weights = [np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
                         np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float32)]
    sparse_ids = create_sparse_tensor(indices=[[0, 0], [0, 1], [1, 0], [1,1]], values=[0, 1, 2, 3], dense_shape=[2, 2])
    sparse_weights = create_sparse_tensor(indices=[[0, 0], [0, 1], [1, 0], [1,1]], values=[1.0, 2.0, 0.5, 1.5], dense_shape=[2, 2])
    combiner = "mean"
    default_id = 0
    max_norm = 2.0
    name = "embedding_lookup_sqrtn"
    allow_fast_lookup = False

    input_dict = {
        "embedding_weights": embedding_weights,
        "sparse_ids": sparse_ids,
        "sparse_weights": sparse_weights,
        "combiner": combiner,
        "default_id": default_id,
        "max_norm": max_norm,
        "name": name,
        "allow_fast_lookup": allow_fast_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    embedding_weights = [np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)]
    sparse_ids = create_sparse_tensor(indices=[[0, 0], [0, 1], [1, 0]], values=[0, 1, 2], dense_shape=[2, 2])
    sparse_weights = create_sparse_tensor(indices=[[0, 0], [0, 1], [1, 0]], values=[1.0, 2.0, 3.0], dense_shape=[2, 2])
    combiner = "mean"
    default_id = 0
    max_norm = None
    name = None
    allow_fast_lookup = False

    input_dict = {
        "embedding_weights": embedding_weights,
        "sparse_ids": sparse_ids,
        "sparse_weights": sparse_weights,
        "combiner": combiner,
        "default_id": default_id,
        "max_norm": max_norm,
        "name": name,
        "allow_fast_lookup": allow_fast_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7
    embedding_weights = [np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)]
    sparse_ids = create_sparse_tensor(indices=[[0, 0], [0, 1], [1, 0], [1,1]], values=[0, 1, 0, 1], dense_shape=[2, 2])
    sparse_weights = create_sparse_tensor(indices=[[0, 0], [0, 1], [1, 0], [1,1]], values=[1.0, 2.0, 0.5, 1.5], dense_shape=[2, 2])
    combiner = "sum"
    default_id = 0
    max_norm = 2.0
    name = "embedding_lookup_sqrtn"
    allow_fast_lookup = False

    input_dict = {
        "embedding_weights": embedding_weights,
        "sparse_ids": sparse_ids,
        "sparse_weights": sparse_weights,
        "combiner": combiner,
        "default_id": default_id,
        "max_norm": max_norm,
        "name": name,
        "allow_fast_lookup": allow_fast_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 - RaggedTensor
    embedding_weights = [np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)]
    sparse_ids = create_ragged_tensor([[0, 1], [2]])
    sparse_weights = create_ragged_tensor([[1.0, 2.0], [3.0]])
    combiner = "mean"
    default_id = 0
    max_norm = None
    name = "embedding_lookup_ragged"
    allow_fast_lookup = False

    input_dict = {
        "embedding_weights": embedding_weights,
        "sparse_ids": sparse_ids,
        "sparse_weights": sparse_weights,
        "combiner": combiner,
        "default_id": default_id,
        "max_norm": max_norm,
        "name": name,
        "allow_fast_lookup": allow_fast_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 - RaggedTensor with default id
    embedding_weights = [np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)]
    sparse_ids = create_ragged_tensor([[], [2]])
    sparse_weights = create_ragged_tensor([[], [3.0]])
    combiner = "mean"
    default_id = 1
    max_norm = None
    name = "embedding_lookup_ragged"
    allow_fast_lookup = False

    input_dict = {
        "embedding_weights": embedding_weights,
        "sparse_ids": sparse_ids,
        "sparse_weights": sparse_weights,
        "combiner": combiner,
        "default_id": default_id,
        "max_norm": max_norm,
        "name": name,
        "allow_fast_lookup": allow_fast_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 - RaggedTensor no weights
    embedding_weights = [np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)]
    sparse_ids = create_ragged_tensor([[0, 1], [2]])
    sparse_weights = None
    combiner = "mean"
    default_id = 0
    max_norm = None
    name = "embedding_lookup_ragged"
    allow_fast_lookup = False

    input_dict = {
        "embedding_weights": embedding_weights,
        "sparse_ids": sparse_ids,
        "sparse_weights": sparse_weights,
        "combiner": combiner,
        "default_id": default_id,
        "max_norm": max_norm,
        "name": name,
        "allow_fast_lookup": allow_fast_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 - embedding_weights as numpy array
    embedding_weights = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    sparse_ids = create_sparse_tensor(indices=[[0, 0], [0, 1], [1, 0]], values=[0, 1, 2], dense_shape=[2, 2])
    sparse_weights = create_sparse_tensor(indices=[[0, 0], [0, 1], [1, 0]], values=[1.0, 2.0, 3.0], dense_shape=[2, 2])
    combiner = "mean"
    default_id = 0
    max_norm = None
    name = "embedding_lookup"
    allow_fast_lookup = False

    input_dict = {
        "embedding_weights": [embedding_weights],
        "sparse_ids": sparse_ids,
        "sparse_weights": sparse_weights,
        "combiner": combiner,
        "default_id": default_id,
        "max_norm": max_norm,
        "name": name,
        "allow_fast_lookup": allow_fast_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Different shape for embedding weights
    embedding_weights = [np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)]
    sparse_ids = create_sparse_tensor(indices=[[0, 0], [0, 1], [1, 0]], values=[0, 1, 0], dense_shape=[2, 2])
    sparse_weights = create_sparse_tensor(indices=[[0, 0], [0, 1], [1, 0]], values=[1.0, 2.0, 3.0], dense_shape=[2, 2])
    combiner = "mean"
    default_id = 0
    max_norm = None
    name = "embedding_lookup_diff_shape"
    allow_fast_lookup = False

    input_dict = {
        "embedding_weights": embedding_weights,
        "sparse_ids": sparse_ids,
        "sparse_weights": sparse_weights,
        "combiner": combiner,
        "default_id": default_id,
        "max_norm": max_norm,
        "name": name,
        "allow_fast_lookup": allow_fast_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13: sparse_ids as numpy array, sparse_weights as numpy array
    embedding_weights = [np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)]
    indices = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    values = np.array([0, 1, 2], dtype=np.int64)
    dense_shape = np.array([2, 2], dtype=np.int64)
    sparse_ids = create_sparse_tensor(indices=indices, values=values, dense_shape=dense_shape)
    
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    sparse_weights = create_sparse_tensor(indices=indices, values=values, dense_shape=dense_shape)
    combiner = "mean"
    default_id = 0
    max_norm = None
    name = "embedding_lookup"
    allow_fast_lookup = False

    input_dict = {
        "embedding_weights": embedding_weights,
        "sparse_ids": sparse_ids,
        "sparse_weights": sparse_weights,
        "combiner": combiner,
        "default_id": default_id,
        "max_norm": max_norm,
        "name": name,
        "allow_fast_lookup": allow_fast_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 14: empty sparse tensor
    embedding_weights = [np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)]
    sparse_ids = create_sparse_tensor(indices=np.array([]).reshape(0,2), values=np.array([]), dense_shape=[2, 2])
    sparse_weights = create_sparse_tensor(indices=np.array([]).reshape(0,2), values=np.array([]), dense_shape=[2, 2])
    combiner = "mean"
    default_id = 0
    max_norm = None
    name = "embedding_lookup"
    allow_fast_lookup = False

    input_dict = {
        "embedding_weights": embedding_weights,
        "sparse_ids": sparse_ids,
        "sparse_weights": sparse_weights,
        "combiner": combiner,
        "default_id": default_id,
        "max_norm": max_norm,
        "name": name,
        "allow_fast_lookup": allow_fast_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 15: Higher rank sparse ids and weights
    embedding_weights = [np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)]
    sparse_ids = create_sparse_tensor(indices=[[0, 0, 0], [0, 0, 1], [0, 1, 0]], values=[0, 1, 2], dense_shape=[1, 2, 2])
    sparse_weights = create_sparse_tensor(indices=[[0, 0, 0], [0, 0, 1], [0, 1, 0]], values=[1.0, 2.0, 3.0], dense_shape=[1, 2, 2])
    combiner = "mean"
    default_id = 0
    max_norm = None
    name = "embedding_lookup"
    allow_fast_lookup = False

    input_dict = {
        "embedding_weights": embedding_weights,
        "sparse_ids": sparse_ids,
        "sparse_weights": sparse_weights,
        "combiner": combiner,
        "default_id": default_id,
        "max_norm": max_norm,
        "name": name,
        "allow_fast_lookup": allow_fast_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.safe_embedding_lookup_sparse"] = tf_nn_safe_embedding_lookup_sparse_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.safe_embedding_lookup_sparse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.safe_embedding_lookup_sparse'.")

check_valid('tf.nn.safe_embedding_lookup_sparse', generated_inputs['tf.nn.safe_embedding_lookup_sparse'], lib="tf", suffix=0)
