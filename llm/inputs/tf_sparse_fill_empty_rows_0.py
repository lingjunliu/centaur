
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_fill_empty_rows_inputs():
    """
    Generates a list of valid inputs for the tf.sparse.fill_empty_rows function.
    """
    list_of_inputs = []

    # Helper function to create input dictionaries.
    # The API requires a tf.SparseTensor. We will provide this directly, as providing
    # a dense tensor or numpy array causes a TypeError within the API call.
    def create_input_dict(indices, values, dense_shape, default_value, name=None):
        # Handle the case of empty indices correctly by providing a 2D array of shape [0, rank].
        if not indices:
            rank = len(dense_shape)
            indices_np = np.empty((0, rank), dtype=np.int64)
        else:
            indices_np = np.array(indices, dtype=np.int64)

        # The sp_input must be a SparseTensor.
        sp_input = tf.SparseTensor(
            indices=indices_np,
            values=values,
            dense_shape=np.array(dense_shape, dtype=np.int64)
        )

        # The default_value must be a scalar tensor of a matching dtype.
        default_value_tensor = tf.constant(default_value, dtype=sp_input.dtype)

        input_dict = {
            'sp_input': sp_input,
            'default_value': default_value_tensor,
            'name': name
        }
        return input_dict

    # Input 1: Basic example from the documentation
    list_of_inputs.append(create_input_dict(
        indices=[[0, 1], [0, 3], [2, 0], [3, 1]],
        values=np.array([10, 20, 30, 40], dtype=np.int32),
        dense_shape=[5, 6],
        default_value=0,
        name='basic_int_example'
    ))

    # Input 2: Floating point values with negative default
    list_of_inputs.append(create_input_dict(
        indices=[[1, 1]],
        values=np.array([3.14], dtype=np.float32),
        dense_shape=[3, 2],
        default_value=-1.0
    ))

    # Input 3: String values
    list_of_inputs.append(create_input_dict(
        indices=[[0, 0], [2, 1]],
        values=np.array(['hello', 'world'], dtype=object),
        dense_shape=[3, 3],
        default_value='missing',
        name='string_example'
    ))

    # Input 4: A completely empty SparseTensor
    list_of_inputs.append(create_input_dict(
        indices=[],
        values=np.array([], dtype=np.int32),
        dense_shape=[4, 5],
        default_value=99
    ))

    # Input 5: A full SparseTensor (no empty rows)
    list_of_inputs.append(create_input_dict(
        indices=[[0, 0], [1, 1], [2, 2]],
        values=np.array([1, 2, 3], dtype=np.int32),
        dense_shape=[3, 4],
        default_value=-9
    ))

    # Input 6: Negative integer values
    list_of_inputs.append(create_input_dict(
        indices=[[0, 1], [2, 2]],
        values=np.array([-10, -20], dtype=np.int32),
        dense_shape=[4, 3],
        default_value=-99
    ))

    # Input 7: Empty rows at the beginning and end
    list_of_inputs.append(create_input_dict(
        indices=[[1, 0], [2, 3]],
        values=np.array([5.5, 6.6], dtype=np.float64),
        dense_shape=[4, 5],
        default_value=0.0,
        name='float64_example'
    ))

    # Input 8: Empty columns at the end (should have no effect)
    list_of_inputs.append(create_input_dict(
        indices=[[0, 0], [1, 1]],
        values=np.array([101, 202], dtype=np.int64),
        dense_shape=[3, 5],
        default_value=0
    ))

    # Input 9: Unordered indices (the function should handle this by reordering)
    list_of_inputs.append(create_input_dict(
        indices=[[3, 1], [0, 1], [2, 0], [0, 3]],
        values=np.array([40, 10, 30, 20], dtype=np.int32),
        dense_shape=[5, 6],
        default_value=0,
        name='unordered_indices'
    ))

    # Input 10: Larger shape with sparse entries
    list_of_inputs.append(create_input_dict(
        indices=[[1, 10], [5, 50], [8, 99]],
        values=np.array([100, 200, 300], dtype=np.int32),
        dense_shape=[10, 100],
        default_value=42
    ))

    return [copy.deepcopy(d) for d in list_of_inputs]

generated_inputs["tf.sparse.fill_empty_rows"] = tf_sparse_fill_empty_rows_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.fill_empty_rows' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.fill_empty_rows'.")

check_valid('tf.sparse.fill_empty_rows', generated_inputs['tf.sparse.fill_empty_rows'], lib="tf", suffix=0)
