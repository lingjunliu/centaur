
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_nn_safe_embedding_lookup_sparse_inputs():
    list_of_inputs = []

    def create_sparse_tensor(indices, values, dense_shape, dtype=np.int64):
        if len(indices) == 0:
            indices = np.empty(shape=[0, len(dense_shape)], dtype=np.int64)
        else:
            indices = np.array(indices, dtype=np.int64)
        return tf.SparseTensor(indices=indices,
                               values=np.array(values, dtype=dtype),
                               dense_shape=np.array(dense_shape, dtype=np.int64))

    # Input 1
    embedding_weights = [np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)]
    sparse_ids = create_sparse_tensor([[0, 0], [0, 1], [1, 0]], [0, 1, 2], [2, 2])
    sparse_weights = create_sparse_tensor([[0, 0], [0, 1], [1, 0]], [1.0, 2.0, 3.0], [2, 2], dtype=np.float32)
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

    # Input 2
    embedding_weights = [np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)]
    sparse_ids = create_sparse_tensor([[0, 0], [0, 1]], [0, 1], [1, 2])
    sparse_weights = None
    combiner = "sum"
    default_id = 0
    max_norm = 1.0
    name = "test_embedding_lookup"
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
    embedding_weights = [np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32),
                         np.array([[0.7, 0.8], [0.9, 1.0]], dtype=np.float32)]
    sparse_ids = create_sparse_tensor([[0, 0], [0, 1], [1, 0]], [0, 1, 4], [2, 2])
    sparse_weights = create_sparse_tensor([[0, 0], [0, 1], [1, 0]], [1.0, 2.0, 3.0], [2, 2], dtype=np.float32)
    combiner = "sqrtn"
    default_id = 0
    max_norm = 2.0
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

    # Input 4
    embedding_weights = [np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)]
    sparse_ids = create_sparse_tensor([[0, 0], [0, 1], [1, 0]], [0, 1, -1], [2, 2])
    sparse_weights = create_sparse_tensor([[0, 0], [0, 1], [1, 0]], [1.0, 2.0, 3.0], [2, 2], dtype=np.float32)
    combiner = "mean"
    default_id = 1
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

    # Input 5
    embedding_weights = [np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)]
    sparse_ids = create_sparse_tensor([[0, 0], [0, 1]], [0, 1], [1, 2])
    sparse_weights = create_sparse_tensor([[0, 0], [0, 1]], [1.0, -0.5], [1, 2], dtype=np.float32)
    combiner = "sum"
    default_id = 1
    max_norm = 1.0
    name = "test_embedding_lookup"
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

    # Input 6 - RaggedTensor
    embedding_weights = [np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)]
    sparse_ids = tf.ragged.constant([[0, 1], [2]])
    sparse_weights = tf.ragged.constant([[1.0, 2.0], [3.0]])
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

    # Input 7 - RaggedTensor with None weights
    embedding_weights = [np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)]
    sparse_ids = tf.ragged.constant([[0, 1], [2]])
    sparse_weights = None
    combiner = "sum"
    default_id = 0
    max_norm = 1.0
    name = "test_embedding_lookup"
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

    # Input 8 - Multi-dimensional SparseTensor
    embedding_weights = [np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)]
    sparse_ids = create_sparse_tensor([[0, 0, 0], [0, 0, 1]], [0, 1], [1, 1, 2])
    sparse_weights = create_sparse_tensor([[0, 0, 0], [0, 0, 1]], [1.0, 2.0], [1, 1, 2], dtype=np.float32)
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

    # Input 9 - Multi-dimensional RaggedTensor
    embedding_weights = [np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)]
    sparse_ids = tf.ragged.constant([[[0, 1], [2]]])
    sparse_weights = tf.ragged.constant([[[1.0, 2.0], [3.0]]])
    combiner = "sum"
    default_id = 0
    max_norm = 1.0
    name = "test_embedding_lookup"
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

    # Input 10 - Empty sparse_ids.
    embedding_weights = [np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)]
    sparse_ids = create_sparse_tensor([], [], [1, 1])
    sparse_weights = create_sparse_tensor([], [], [1, 1], dtype=np.float32)
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

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.safe_embedding_lookup_sparse"] = tf_nn_safe_embedding_lookup_sparse_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.safe_embedding_lookup_sparse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.safe_embedding_lookup_sparse'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.safe_embedding_lookup_sparse', generated_inputs['tf.nn.safe_embedding_lookup_sparse'], lib="tf", suffix=0)
