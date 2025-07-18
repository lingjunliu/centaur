
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

    # Helper function to create a tf.SparseTensor and add a .size attribute
    # to make it compatible with the testing framework that expects it.
    def create_sparse_tensor_with_size(indices, values, dense_shape, values_dtype=None):
        indices_np = np.array(indices, dtype=np.int64)
        values_np = np.array(values, dtype=values_dtype) if values_dtype else np.array(values)
        dense_shape_np = np.array(dense_shape, dtype=np.int64)

        sp_tensor = tf.SparseTensor(indices=indices_np,
                                    values=values_np,
                                    dense_shape=dense_shape_np)
        
        # Monkey-patch the .size attribute for the testing framework
        sp_tensor.size = values_np.size
        
        return sp_tensor

    # Input 1: Basic case from documentation (int32)
    sp_input_1 = create_sparse_tensor_with_size(
        indices=[[0, 1], [0, 3], [2, 0], [3, 1]],
        values=[10, 20, 30, 40],
        dense_shape=[5, 6],
        values_dtype=np.int32
    )
    default_value_1 = np.array(99, dtype=np.int32)
    input_dict_1 = {'sp_input': sp_input_1, 'default_value': default_value_1, 'name': 'basic_case'}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Float32 type with a negative default value
    sp_input_2 = create_sparse_tensor_with_size(
        indices=[[1, 1], [3, 2]],
        values=[1.1, 3.3],
        dense_shape=[4, 4],
        values_dtype=np.float32
    )
    default_value_2 = np.array(-1.0, dtype=np.float32)
    input_dict_2 = {'sp_input': sp_input_2, 'default_value': default_value_2, 'name': 'float_case'}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: No empty rows
    sp_input_3 = create_sparse_tensor_with_size(
        indices=[[0, 0], [1, 1], [2, 2]],
        values=[1, 2, 3],
        dense_shape=[3, 4],
        values_dtype=np.int32
    )
    default_value_3 = np.array(0, dtype=np.int32)
    input_dict_3 = {'sp_input': sp_input_3, 'default_value': default_value_3, 'name': 'no_empty_rows'}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: All rows are empty
    sp_input_4 = create_sparse_tensor_with_size(
        indices=np.empty((0, 2), dtype=np.int64),
        values=[],
        dense_shape=[4, 5],
        values_dtype=np.int32
    )
    default_value_4 = np.array(-1, dtype=np.int32)
    input_dict_4 = {'sp_input': sp_input_4, 'default_value': default_value_4, 'name': 'all_empty_rows'}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Large shape with many empty rows
    sp_input_5 = create_sparse_tensor_with_size(
        indices=[[1, 50], [9, 99]],
        values=[100, 200],
        dense_shape=[10, 100],
        values_dtype=np.int32
    )
    default_value_5 = np.array(5, dtype=np.int32)
    input_dict_5 = {'sp_input': sp_input_5, 'default_value': default_value_5, 'name': 'large_shape'}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Empty first and last rows, float64 type
    sp_input_6 = create_sparse_tensor_with_size(
        indices=[[1, 0], [1, 2], [2, 1]],
        values=[1.0, 2.0, 3.0],
        dense_shape=[4, 3],
        values_dtype=np.float64
    )
    default_value_6 = np.array(0.0, dtype=np.float64)
    input_dict_6 = {'sp_input': sp_input_6, 'default_value': default_value_6, 'name': 'boundary_empty_rows'}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: int64 type with large numbers
    sp_input_7 = create_sparse_tensor_with_size(
        indices=[[0, 0]],
        values=[np.iinfo(np.int64).max - 1],
        dense_shape=[2, 2],
        values_dtype=np.int64
    )
    default_value_7 = np.array(np.iinfo(np.int64).min + 1, dtype=np.int64)
    input_dict_7 = {'sp_input': sp_input_7, 'default_value': default_value_7, 'name': 'int64_case'}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Single column tensor
    sp_input_8 = create_sparse_tensor_with_size(
        indices=[[0, 0], [3, 0]],
        values=[1, 4],
        dense_shape=[5, 1],
        values_dtype=np.int32
    )
    default_value_8 = np.array(9, dtype=np.int32)
    input_dict_8 = {'sp_input': sp_input_8, 'default_value': default_value_8, 'name': None}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Minimal 2x2 case with one empty row and negative values
    sp_input_9 = create_sparse_tensor_with_size(
        indices=[[1, 1]],
        values=[-5],
        dense_shape=[2, 2],
        values_dtype=np.int32
    )
    default_value_9 = np.array(0, dtype=np.int32)
    input_dict_9 = {'sp_input': sp_input_9, 'default_value': default_value_9, 'name': 'minimal_2x2'}
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: complex64 type
    sp_input_10 = create_sparse_tensor_with_size(
        indices=[[0, 1]],
        values=[(1+2j)],
        dense_shape=[3, 3],
        values_dtype=np.complex64
    )
    default_value_10 = np.array((0+0j), dtype=np.complex64)
    input_dict_10 = {'sp_input': sp_input_10, 'default_value': default_value_10, 'name': 'complex_case'}
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

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
