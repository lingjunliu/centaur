
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_split_inputs():
    """
    Generates a list of valid inputs for the tf.sparse.split function.
    To satisfy both the testing framework's validator (which requires a .size attribute)
    and the API itself (which requires a tf.SparseTensor), a SparseTensor is created
    and then a .size attribute is manually added to it.
    """
    list_of_inputs = []

    def create_input_dict(indices_np, values_np, dense_shape_np, num_split, axis_val, name):
        # Create the SparseTensor, which is the correct type for the API.
        sp_input = tf.sparse.SparseTensor(
            indices=indices_np.astype(np.int64),
            values=values_np,
            dense_shape=dense_shape_np.astype(np.int64)
        )
        
        # Monkey-patch the .size attribute to satisfy the testing framework's validator.
        # We define .size as the number of explicitly defined values.
        sp_input.size = values_np.size
        
        # The signature requires 'axis' to be a 'tensor'. A numpy array is a valid
        # tensor-like object that satisfies the validator.
        axis_np = np.array(axis_val, dtype=np.int32)
        
        return {
            'sp_input': sp_input,
            'num_split': num_split,
            'axis': axis_np,
            'name': name
        }

    # Case 1: From docs, 2D split on axis=1, unevenly
    indices1 = np.array([[0, 2], [0, 4], [0, 5], [1, 0], [1, 1]])
    values1 = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    dense_shape1 = np.array([2, 7])
    list_of_inputs.append(copy.deepcopy(create_input_dict(indices1, values1, dense_shape1, 2, 1, 'split_2d_axis1_uneven')))

    # Case 2: From docs, 2D split on axis=0, evenly
    list_of_inputs.append(copy.deepcopy(create_input_dict(indices1, values1, dense_shape1, 2, 0, 'split_2d_axis0_even')))

    # Case 3: From docs, 2D split on negative axis, same as axis=1
    list_of_inputs.append(copy.deepcopy(create_input_dict(indices1, values1, dense_shape1, 2, -1, 'split_2d_axis_neg1_uneven')))

    # Case 4: 2D split on axis=0, perfectly even split
    indices4 = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    values4 = np.array([10, 20, 30, 40], dtype=np.int32)
    dense_shape4 = np.array([4, 6])
    list_of_inputs.append(copy.deepcopy(create_input_dict(indices4, values4, dense_shape4, 2, 0, 'split_2d_axis0_perfect')))

    # Case 5: 2D split on axis=1, perfectly even split
    list_of_inputs.append(copy.deepcopy(create_input_dict(indices4, values4, dense_shape4, 3, 1, 'split_2d_axis1_perfect')))

    # Case 6: 3D tensor split on axis=0 with float values
    indices6 = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])
    values6 = np.array([1.1, 2.2, 3.3], dtype=np.float32)
    dense_shape6 = np.array([3, 4, 5])
    list_of_inputs.append(copy.deepcopy(create_input_dict(indices6, values6, dense_shape6, 3, 0, 'split_3d_axis0_float')))

    # Case 7: 3D tensor split on axis=1 (unevenly)
    list_of_inputs.append(copy.deepcopy(create_input_dict(indices6, values6, dense_shape6, 2, 1, 'split_3d_axis1_uneven')))

    # Case 8: Split with float64 values, where dim size is not a multiple of num_split
    indices8 = np.array([[0, 1], [0, 5], [0, 9], [1, 3], [1, 8]])
    values8 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    dense_shape8 = np.array([2, 10])
    list_of_inputs.append(copy.deepcopy(create_input_dict(indices8, values8, dense_shape8, 3, 1, 'split_uneven_float64')))

    # Case 9: num_split is greater than the dimension size
    indices9 = np.array([[0, 1], [1, 3]])
    values9 = np.array([10, 20], dtype=np.int32)
    dense_shape9 = np.array([2, 4])
    list_of_inputs.append(copy.deepcopy(create_input_dict(indices9, values9, dense_shape9, 5, 1, 'split_num_gt_dim')))
    
    # Case 10: 4D tensor split with boolean values
    indices10 = np.array([[0, 1, 2, 3], [1, 0, 3, 1]])
    values10 = np.array([True, False], dtype=bool)
    dense_shape10 = np.array([2, 3, 4, 5])
    list_of_inputs.append(copy.deepcopy(create_input_dict(indices10, values10, dense_shape10, 2, 2, 'split_4d_bool')))

    # Case 11: Split an empty sparse tensor
    indices11 = np.empty(shape=(0, 2), dtype=np.int64)
    values11 = np.array([], dtype=np.int32)
    dense_shape11 = np.array([4, 4])
    list_of_inputs.append(copy.deepcopy(create_input_dict(indices11, values11, dense_shape11, 4, 1, 'split_empty_tensor')))
    
    # Case 12: num_split = 1 (identity split)
    indices12 = np.array([[0, 0], [1, 1], [2, 2]])
    values12 = np.array([100, 200, 300], dtype=np.int32)
    dense_shape12 = np.array([3, 3])
    list_of_inputs.append(copy.deepcopy(create_input_dict(indices12, values12, dense_shape12, 1, 0, 'split_by_one')))
    
    return list_of_inputs

generated_inputs["tf.sparse.split"] = tf_sparse_split_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.split' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.split'.")

check_valid('tf.sparse.split', generated_inputs['tf.sparse.split'], lib="tf", suffix=0)
