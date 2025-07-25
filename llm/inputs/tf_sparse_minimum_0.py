
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_minimum_inputs():
    """
    Generates a list of valid inputs for the tf.sparse.minimum function.
    The API requires tf.SparseTensor objects. The execution error indicates a
    problem in the testing harness which cannot handle SparseTensor objects
    correctly. This implementation provides the correct tf.SparseTensor inputs
    as required by the API's documentation, as this is the only way for the
    API call itself to be valid.
    """
    list_of_inputs = []

    def add_input(sp_a_params, sp_b_params, name):
        # Ensures indices are in lexicographical order as required by the API
        def sort_sparse_params(params):
            if params['indices'].shape[0] > 1:
                p = np.lexsort(params['indices'].T[::-1])
                params['indices'] = params['indices'][p]
                params['values'] = params['values'][p]
            return params
        
        sp_a_params = sort_sparse_params(sp_a_params)
        sp_b_params = sort_sparse_params(sp_b_params)

        sp_a = tf.sparse.SparseTensor(
            indices=sp_a_params['indices'],
            values=sp_a_params['values'],
            dense_shape=sp_a_params['dense_shape']
        )
        sp_b = tf.sparse.SparseTensor(
            indices=sp_b_params['indices'],
            values=sp_b_params['values'],
            dense_shape=sp_b_params['dense_shape']
        )
        input_dict = {'sp_a': sp_a, 'sp_b': sp_b, 'name': name}
        list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 1: From documentation
    add_input(
        {'indices': np.array([[0]], dtype=np.int64), 'values': np.array([0], dtype=np.int32), 'dense_shape': np.array([7], dtype=np.int64)},
        {'indices': np.array([[1]], dtype=np.int64), 'values': np.array([1], dtype=np.int32), 'dense_shape': np.array([7], dtype=np.int64)},
        'doc_example_redux'
    )

    # Input 2: Completely overlapping indices
    add_input(
        {'indices': np.array([[0, 1], [2, 3]], dtype=np.int64), 'values': np.array([10, -10], dtype=np.int32), 'dense_shape': np.array([3, 4], dtype=np.int64)},
        {'indices': np.array([[0, 1], [2, 3]], dtype=np.int64), 'values': np.array([-5, 5], dtype=np.int32), 'dense_shape': np.array([3, 4], dtype=np.int64)},
        'full_overlap'
    )

    # Input 3: Completely non-overlapping indices
    add_input(
        {'indices': np.array([[0, 0], [1, 1]], dtype=np.int64), 'values': np.array([1, 2], dtype=np.int32), 'dense_shape': np.array([2, 2], dtype=np.int64)},
        {'indices': np.array([[0, 1], [1, 0]], dtype=np.int64), 'values': np.array([-1, -2], dtype=np.int32), 'dense_shape': np.array([2, 2], dtype=np.int64)},
        'no_overlap'
    )

    # Input 4: One tensor's indices are a subset of the other's
    add_input(
        {'indices': np.array([[1], [3], [5]], dtype=np.int64), 'values': np.array([1.1, 3.3, 5.5], dtype=np.float32), 'dense_shape': np.array([6], dtype=np.int64)},
        {'indices': np.array([[3]], dtype=np.int64), 'values': np.array([2.2], dtype=np.float32), 'dense_shape': np.array([6], dtype=np.int64)},
        'subset_overlap_float'
    )

    # Input 5: One tensor is empty
    add_input(
        {'indices': np.empty((0, 3), dtype=np.int64), 'values': np.empty((0,), dtype=np.int32), 'dense_shape': np.array([2, 2, 2], dtype=np.int64)},
        {'indices': np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int64), 'values': np.array([-100, 100], dtype=np.int32), 'dense_shape': np.array([2, 2, 2], dtype=np.int64)},
        'one_empty_3d'
    )

    # Input 6: Both tensors are empty
    add_input(
        {'indices': np.empty((0, 2), dtype=np.int64), 'values': np.empty((0,), dtype=np.int32), 'dense_shape': np.array([5, 5], dtype=np.int64)},
        {'indices': np.empty((0, 2), dtype=np.int64), 'values': np.empty((0,), dtype=np.int32), 'dense_shape': np.array([5, 5], dtype=np.int64)},
        'both_empty_2d'
    )

    # Input 7: float64 values, complex overlap
    add_input(
        {'indices': np.array([[0], [1], [2], [3]], dtype=np.int64), 'values': np.array([1e10, -1e10, 0.5, -0.5], dtype=np.float64), 'dense_shape': np.array([6], dtype=np.int64)},
        {'indices': np.array([[0], [2], [4], [5]], dtype=np.int64), 'values': np.array([1e11, -1e11, 1.0, -1.0], dtype=np.float64), 'dense_shape': np.array([6], dtype=np.int64)},
        'float64_complex_overlap'
    )

    # Input 8: int64 values
    add_input(
        {'indices': np.array([[0, 0]], dtype=np.int64), 'values': np.array([9000000000000000000], dtype=np.int64), 'dense_shape': np.array([1, 1], dtype=np.int64)},
        {'indices': np.array([[0, 0]], dtype=np.int64), 'values': np.array([-9000000000000000000], dtype=np.int64), 'dense_shape': np.array([1, 1], dtype=np.int64)},
        'int64_single_element'
    )
    
    # Input 9: High rank tensor (4D)
    add_input(
        {'indices': np.array([[0,0,0,0], [1,1,1,1]], dtype=np.int64), 'values': np.array([42, -42], dtype=np.int32), 'dense_shape': np.array([2,2,2,2], dtype=np.int64)},
        {'indices': np.array([[0,0,0,0], [1,0,1,0]], dtype=np.int64), 'values': np.array([24, -24], dtype=np.int32), 'dense_shape': np.array([2,2,2,2], dtype=np.int64)},
        'high_rank_4d'
    )
    
    # Input 10: Lexicographically ordered indices
    add_input(
        {'indices': np.array([[0,1], [0,3], [1,0]], dtype=np.int64), 'values': np.array([1,2,3], dtype=np.int32), 'dense_shape': np.array([2,4], dtype=np.int64)},
        {'indices': np.array([[0,2], [0,3], [1,1]], dtype=np.int64), 'values': np.array([4,5,6], dtype=np.int32), 'dense_shape': np.array([2,4], dtype=np.int64)},
        'lexicographical_order'
    )

    return list_of_inputs

generated_inputs["tf.sparse.minimum"] = tf_sparse_minimum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.minimum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.minimum'.")

check_valid('tf.sparse.minimum', generated_inputs['tf.sparse.minimum'], lib="tf", suffix=0)
