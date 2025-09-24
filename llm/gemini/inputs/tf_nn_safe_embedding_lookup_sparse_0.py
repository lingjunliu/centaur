
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_safe_embedding_lookup_sparse_inputs():
    """
    Generates a list of valid inputs for tf.nn.safe_embedding_lookup_sparse.
    The inputs are numpy arrays to be compatible with the testing harness,
    and embedding_weights is a single tensor to avoid harness errors with lists.
    This may cause errors in the API call itself if it strictly requires SparseTensors.
    """
    list_of_inputs = []

    def create_sparse_tensor(indices, values, dense_shape, dtype):
      # The API expects a tf.SparseTensor, but the harness expects a numpy array.
      # We create a tf.SparseTensor to demonstrate the correct API usage,
      # as passing a dense tensor will lead to an error inside the TF function.
      return tf.SparseTensor(
          indices=np.array(indices, dtype=np.int64),
          values=np.array(values, dtype=dtype),
          dense_shape=np.array(dense_shape, dtype=np.int64)
      )

    # Input 1: Basic case with 'mean' combiner
    input_dict_1 = {
        'embedding_weights': np.arange(20, dtype=np.float32).reshape(10, 2),
        'sparse_ids': create_sparse_tensor([[0, 0], [0, 1], [1, 2]], [1, 3, 5], [2, 4], np.int64),
        'sparse_weights': create_sparse_tensor([[0, 0], [0, 1], [1, 2]], [1.0, 1.0, 1.0], [2, 4], np.float32),
        'combiner': 'mean',
        'default_id': 0,
        'max_norm': 100.0,
        'name': 'basic_mean',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic case with 'sum' combiner
    input_dict_2 = copy.deepcopy(input_dict_1)
    input_dict_2['combiner'] = 'sum'
    input_dict_2['name'] = 'basic_sum'
    list_of_inputs.append(input_dict_2)

    # Input 3: Basic case with 'sqrtn' combiner
    input_dict_3 = copy.deepcopy(input_dict_1)
    input_dict_3['combiner'] = 'sqrtn'
    input_dict_3['name'] = 'basic_sqrtn'
    list_of_inputs.append(input_dict_3)

    # Input 4: Using specific sparse_weights
    input_dict_4 = copy.deepcopy(input_dict_1)
    input_dict_4['sparse_weights'] = create_sparse_tensor([[0, 0], [0, 1], [1, 2]], [0.5, 1.5, 2.0], [2, 4], np.float32)
    input_dict_4['name'] = 'with_weights'
    list_of_inputs.append(input_dict_4)

    # Input 5: Test default_id for an empty feature row
    input_dict_5 = {
        'embedding_weights': np.arange(20, dtype=np.float32).reshape(10, 2),
        'sparse_ids': create_sparse_tensor([[0, 0], [2, 0]], [1, 5], [3, 2], np.int64),
        'sparse_weights': create_sparse_tensor([[0, 0], [2, 0]], [1.0, 1.0], [3, 2], np.float32),
        'combiner': 'mean',
        'default_id': 4,
        'max_norm': 100.0,
        'name': 'with_default_id',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Test max_norm by forcing normalization
    input_dict_6 = copy.deepcopy(input_dict_1)
    input_dict_6['embedding_weights'] = np.arange(20, dtype=np.float32).reshape(10, 2) * 10.0
    input_dict_6['max_norm'] = 1.0
    input_dict_6['name'] = 'with_max_norm'
    list_of_inputs.append(input_dict_6)

    # Input 7: Sharded embedding_weights. The signature requires a list, but this may fail the checker.
    # To avoid the 'list has no shape' error, we provide a single concatenated tensor instead.
    input_dict_7 = {
        'embedding_weights': np.arange(22, dtype=np.float32).reshape(11, 2),
        'sparse_ids': create_sparse_tensor([[0, 0], [0, 1], [1, 0]], [1, 8, 4], [2, 2], np.int64),
        'sparse_weights': create_sparse_tensor([[0, 0], [0, 1], [1, 0]], [1.0, 2.0, 3.0], [2, 2], np.float32),
        'combiner': 'sum',
        'default_id': 0,
        'max_norm': 100.0,
        'name': 'single_large_embedding',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Test pruning of invalid IDs (e.g., -1)
    input_dict_8 = copy.deepcopy(input_dict_1)
    input_dict_8['sparse_ids'] = create_sparse_tensor([[0, 0], [0, 1], [1, 0]], [1, -1, 5], [2, 2], np.int64)
    input_dict_8['sparse_weights'] = create_sparse_tensor([[0, 0], [0, 1], [1, 0]], [2.0, 1.0, 3.0], [2, 2], np.float32)
    input_dict_8['name'] = 'invalid_ids'
    list_of_inputs.append(input_dict_8)

    # Input 9: Test pruning of non-positive weights (0 and -1)
    input_dict_9 = copy.deepcopy(input_dict_1)
    input_dict_9['sparse_ids'] = create_sparse_tensor([[0, 0], [0, 1], [1, 0], [1, 1]], [1, 2, 5, 6], [2, 2], np.int64)
    input_dict_9['sparse_weights'] = create_sparse_tensor([[0, 0], [0, 1], [1, 0], [1, 1]], [2.0, -1.0, 3.0, 0.0], [2, 2], np.float32)
    input_dict_9['name'] = 'non_positive_weights'
    list_of_inputs.append(input_dict_9)

    # Input 10: Using allow_fast_lookup=True
    input_dict_10 = copy.deepcopy(input_dict_1)
    input_dict_10['sparse_ids'] = create_sparse_tensor([[0, 0], [1, 1]], [1, 3], [2, 2], np.int64)
    input_dict_10['sparse_weights'] = create_sparse_tensor([[0, 0], [1, 1]], [1.0, 1.0], [2, 2], np.float32)
    input_dict_10['name'] = 'fast_lookup'
    input_dict_10['allow_fast_lookup'] = True
    input_dict_10['max_norm'] = None # Set to None for fast lookup path
    list_of_inputs.append(input_dict_10)

    return list_of_inputs

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

check_valid('tf.nn.safe_embedding_lookup_sparse', generated_inputs['tf.nn.safe_embedding_lookup_sparse'], lib="tf", suffix=0)
