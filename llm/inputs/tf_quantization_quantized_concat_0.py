
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_quantization_quantized_concat_inputs():
    """
    Generates a list of valid inputs for the tf.quantization.quantized_concat function.
    """
    list_of_inputs = []

    # All 'values' tensors are of type np.int32, as this is the only type
    # with a registered kernel (DT_QINT32) that can be created from numpy.
    # To satisfy the test harness' expectation for a 'tensor_list',
    # lists of tensors are stacked into a single numpy array.
    # This implies all tensors in the list must have the same shape.

    # Input 1: Concatenate two 2D tensors of same shape, along dim 1
    t1_1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    t2_1 = np.array([[5, 6], [7, 8]], dtype=np.int32)
    input_dict_1 = {
        'concat_dim': 1,
        'values': [t1_1, t2_1],
        'input_mins': [np.array(0.0, dtype=np.float32), np.array(0.0, dtype=np.float32)],
        'input_maxes': [np.array(255.0, dtype=np.float32), np.array(255.0, dtype=np.float32)],
        'name': 'concat_2d_qint32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Concatenate two 2D tensors of different shapes, along dim 0
    t1_2 = np.array([[-10, 20], [-30, 40]], dtype=np.int32)
    t2_2 = np.array([[50, -60]], dtype=np.int32)
    input_dict_2 = {
        'concat_dim': 0,
        'values': [t1_2, t2_2],
        'input_mins': [np.array(-128.0, dtype=np.float32), np.array(-100.0, dtype=np.float32)],
        'input_maxes': [np.array(127.0, dtype=np.float32), np.array(100.0, dtype=np.float32)],
        'name': 'concat_2d_qint32_varied_range'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Concatenate three 2D tensors of different shapes
    t1_3 = np.array([[1000], [2000]], dtype=np.int32)
    t2_3 = np.array([[3000, 4000], [5000, 6000]], dtype=np.int32)
    t3_3 = np.array([[7000], [8000]], dtype=np.int32)
    input_dict_3 = {
        'concat_dim': 1,
        'values': [t1_3, t2_3, t3_3],
        'input_mins': [np.array(0.0, dtype=np.float32), np.array(0.0, dtype=np.float32), np.array(500.0, dtype=np.float32)],
        'input_maxes': [np.array(10000.0, dtype=np.float32), np.array(10000.0, dtype=np.float32), np.array(10000.0, dtype=np.float32)],
        'name': 'concat_3_2d_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Concatenate 1D tensors
    t1_4 = np.array([1, 2, 3], dtype=np.int32)
    t2_4 = np.array([4, 5], dtype=np.int32)
    input_dict_4 = {
        'concat_dim': 0,
        'values': [t1_4, t2_4],
        'input_mins': [np.array(0.0, dtype=np.float32), np.array(1.0, dtype=np.float32)],
        'input_maxes': [np.array(255.0, dtype=np.float32), np.array(254.0, dtype=np.float32)],
        'name': 'concat_1d_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    
    # Input 5: Concatenate 3D tensors
    t1_5 = np.zeros((1, 2, 2), dtype=np.int32)
    t2_5 = np.ones((2, 2, 2), dtype=np.int32)
    input_dict_5 = {
        'concat_dim': 0,
        'values': [t1_5, t2_5],
        'input_mins': [np.array(-1.0, dtype=np.float32), np.array(-1.0, dtype=np.float32)],
        'input_maxes': [np.array(1.0, dtype=np.float32), np.array(1.0, dtype=np.float32)],
        'name': 'concat_3d_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    
    # Input 6: Concatenate four 2D tensors
    tensors_6 = [
        np.full((1, 5), 10, dtype=np.int32),
        np.full((2, 5), 20, dtype=np.int32),
        np.full((1, 5), 30, dtype=np.int32),
        np.full((3, 5), 40, dtype=np.int32),
    ]
    input_dict_6 = {
        'concat_dim': 0,
        'values': tensors_6,
        'input_mins': [np.array(f, dtype=np.float32) for f in [0.0] * 4],
        'input_maxes': [np.array(f, dtype=np.float32) for f in [127.0] * 4],
        'name': 'concat_4_2d_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: concat_dim = 2 for a 3D tensor
    t1_7 = np.zeros((2, 3, 1), dtype=np.int32)
    t2_7 = np.ones((2, 3, 4), dtype=np.int32)
    input_dict_7 = {
        'concat_dim': 2, 
        'values': [t1_7, t2_7],
        'input_mins': [np.array(-128.0, dtype=np.float32), np.array(-128.0, dtype=np.float32)],
        'input_maxes': [np.array(127.0, dtype=np.float32), np.array(127.0, dtype=np.float32)],
        'name': 'concat_dim_2_3d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    # Input 8: Using name=None
    t1_8 = np.zeros((5, 5), dtype=np.int32)
    t2_8 = np.ones((5, 5), dtype=np.int32)
    input_dict_8 = {
        'concat_dim': 0,
        'values': [t1_8, t2_8],
        'input_mins': [np.array(0.0, dtype=np.float32), np.array(0.0, dtype=np.float32)],
        'input_maxes': [np.array(1.0, dtype=np.float32), np.array(1.0, dtype=np.float32)],
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: 4D tensors
    t1_9 = np.zeros((1, 2, 2, 1), dtype=np.int32)
    t2_9 = np.ones((1, 2, 2, 2), dtype=np.int32)
    input_dict_9 = {
        'concat_dim': 3,
        'values': [t1_9, t2_9],
        'input_mins': [np.array(-1.0, dtype=np.float32), np.array(0.0, dtype=np.float32)],
        'input_maxes': [np.array(1.0, dtype=np.float32), np.array(2.0, dtype=np.float32)],
        'name': 'concat_4d_qint32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: 2D tensors concatenated along dimension 1
    t1_10 = np.array([[-10, 20], [-30, 40]], dtype=np.int32)
    t2_10 = np.array([[50], [-60]], dtype=np.int32)
    input_dict_10 = {
        'concat_dim': 1,
        'values': [t1_10, t2_10],
        'input_mins': [np.array(-128.0, dtype=np.float32), np.array(-128.0, dtype=np.float32)],
        'input_maxes': [np.array(127.0, dtype=np.float32), np.array(127.0, dtype=np.float32)],
        'name': 'concat_2d_dim1_qint32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.quantization.quantized_concat"] = tf_quantization_quantized_concat_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.quantization.quantized_concat' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.quantization.quantized_concat'.")

check_valid('tf.quantization.quantized_concat', generated_inputs['tf.quantization.quantized_concat'], lib="tf", suffix=0)
