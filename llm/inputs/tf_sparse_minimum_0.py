
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

class ValidatableSparseTensor(tf.SparseTensor):
    """
    A tf.SparseTensor subclass that is also compatible with validation
    frameworks expecting a numpy-like interface for accessing values.
    It exposes a .size property and supports np.min/np.max by operating
    on its `values` tensor.
    """
    def __init__(self, indices, values, dense_shape):
        super().__init__(indices=indices, values=values, dense_shape=dense_shape)
        self._np_values = self.values.numpy()

    @property
    def size(self):
        """Returns the number of explicit values in the sparse tensor."""
        return self._np_values.size

    def __array__(self, dtype=None):
        """
        Allows numpy functions like np.min and np.max to operate on this object.
        They will operate on the explicit values of the sparse tensor.
        """
        if dtype:
            return self._np_values.astype(dtype)
        return self._np_values

def tf_sparse_minimum_inputs():
    """
    Generates a list of valid inputs for the tf.sparse.minimum function.
    The inputs are custom ValidatableSparseTensor objects to satisfy both
    the validation framework and the API's requirements.
    """
    list_of_inputs = []

    def create_sparse_tensor(indices, values, dense_shape):
        np_indices = np.array(indices, dtype=np.int64)
        if np_indices.size == 0:
            rank = len(dense_shape)
            np_indices = np_indices.reshape(0, rank)
        
        np_values = np.array(values) # Let numpy infer dtype

        return ValidatableSparseTensor(
            indices=tf.constant(np_indices, dtype=tf.int64),
            values=tf.constant(np_values),
            dense_shape=tf.constant(dense_shape, dtype=tf.int64)
        )

    # Input 1: Basic 1D case, non-overlapping indices
    sp_a1 = create_sparse_tensor([[1], [3]], [10, 20], [5])
    sp_b1 = create_sparse_tensor([[0], [4]], [5, 15], [5])
    list_of_inputs.append({
        'sp_a': sp_a1,
        'sp_b': sp_b1,
        'name': 'basic_1d'
    })

    # Input 2: Overlapping indices with negative values
    sp_a2 = create_sparse_tensor([[0], [2], [4]], [-5, 10, -15], [5])
    sp_b2 = create_sparse_tensor([[0], [3], [4]], [5, -10, 15], [5])
    list_of_inputs.append({
        'sp_a': sp_a2,
        'sp_b': sp_b2,
        'name': 'overlapping_indices'
    })

    # Input 3: One tensor is empty
    sp_a3 = create_sparse_tensor([[1], [3]], [10, -20], [5])
    sp_b3 = create_sparse_tensor([], [], [5])
    list_of_inputs.append({
        'sp_a': sp_a3,
        'sp_b': sp_b3,
        'name': 'one_empty'
    })

    # Input 4: Both tensors are empty
    sp_a4 = create_sparse_tensor([], [], [10])
    sp_b4 = create_sparse_tensor([], [], [10])
    list_of_inputs.append({
        'sp_a': sp_a4,
        'sp_b': sp_b4,
        'name': 'both_empty'
    })

    # Input 5: Basic 2D case
    sp_a5 = create_sparse_tensor([[0, 1], [1, 2]], [5, 6], [3, 4])
    sp_b5 = create_sparse_tensor([[0, 1], [2, 0]], [-5, 7], [3, 4])
    list_of_inputs.append({
        'sp_a': sp_a5,
        'sp_b': sp_b5,
        'name': 'basic_2d'
    })

    # Input 6: Floating point values (float32)
    sp_a6 = create_sparse_tensor([[0], [2]], np.array([3.14, -2.71], dtype=np.float32), [4])
    sp_b6 = create_sparse_tensor([[1], [2]], np.array([-1.0, 4.0], dtype=np.float32), [4])
    list_of_inputs.append({
        'sp_a': sp_a6,
        'sp_b': sp_b6,
        'name': 'float32_values'
    })

    # Input 7: Float64 values
    sp_a7 = create_sparse_tensor([[0, 0], [1, 1]], np.array([1.23e10, -4.56e10], dtype=np.float64), [2, 2])
    sp_b7 = create_sparse_tensor([[0, 1], [1, 1]], np.array([9.87e10, -5.67e10], dtype=np.float64), [2, 2])
    list_of_inputs.append({
        'sp_a': sp_a7,
        'sp_b': sp_b7,
        'name': 'float64_values'
    })

    # Input 8: Identical sparse tensors
    sp_a8 = create_sparse_tensor([[0, 1], [2, 3]], [-1, 100], [4, 5])
    sp_b8 = create_sparse_tensor([[0, 1], [2, 3]], [-1, 100], [4, 5])
    list_of_inputs.append({
        'sp_a': sp_a8,
        'sp_b': sp_b8,
        'name': 'identical_tensors'
    })

    # Input 9: 3D tensor case
    sp_a9 = create_sparse_tensor([[0, 0, 1], [1, 1, 0], [1, 2, 2]], [10, -20, 30], [2, 3, 4])
    sp_b9 = create_sparse_tensor([[0, 1, 0], [1, 1, 0], [1, 2, 3]], [-5, 15, -25], [2, 3, 4])
    list_of_inputs.append({
        'sp_a': sp_a9,
        'sp_b': sp_b9,
        'name': 'basic_3d'
    })

    # Input 10: One tensor's values are all smaller at overlapping indices
    sp_a10 = create_sparse_tensor([[1], [3]], [5, 15], [5])
    sp_b10 = create_sparse_tensor([[1], [3]], [10, 20], [5])
    list_of_inputs.append({
        'sp_a': sp_a10,
        'sp_b': sp_b10,
        'name': 'a_smaller_than_b'
    })

    # Input 11: Example from documentation
    sp_a11 = create_sparse_tensor([[0]], [0], [7])
    sp_b11 = create_sparse_tensor([[1]], [1], [7])
    list_of_inputs.append({
        'sp_a': sp_a11,
        'sp_b': sp_b11,
        'name': 'doc_example'
    })
    
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
