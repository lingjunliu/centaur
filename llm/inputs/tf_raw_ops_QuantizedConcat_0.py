
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_quantized_concat_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.QuantizedConcat function.
    """
    list_of_inputs = []

    # Case 1: Basic 1D concatenation (dim 0) with quint8
    input_dict = {
        'name': 'concat_1d_dim0_quint8',
        'concat_dim': np.array(0, dtype=np.int32),
        'values': np.array([
            np.array([1, 2, 3], dtype=np.uint8),
            np.array([4, 5], dtype=np.uint8)
        ], dtype=object),
        'input_mins': np.array([0.0, 0.0], dtype=np.float32),
        'input_maxes': np.array([255.0, 255.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 2D concatenation along dimension 0 with qint8
    input_dict = {
        'name': 'concat_2d_dim0_qint8',
        'concat_dim': np.array(0, dtype=np.int32),
        'values': np.array([
            np.array([[-10, -20], [-30, -40]], dtype=np.int8),
            np.array([[-50, -60]], dtype=np.int8)
        ], dtype=object),
        'input_mins': np.array([-128.0, -100.0], dtype=np.float32),
        'input_maxes': np.array([127.0, 100.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 2D concatenation along dimension 1 with qint32
    input_dict = {
        'name': 'concat_2d_dim1_qint32',
        'concat_dim': np.array(1, dtype=np.int32),
        'values': np.array([
            np.array([[1000], [2000]], dtype=np.int32),
            np.array([[3000, 4000], [5000, 6000]], dtype=np.int32)
        ], dtype=object),
        'input_mins': np.array([-10000.0, -10000.0], dtype=np.float32),
        'input_maxes': np.array([10000.0, 10000.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Concatenating three 2D tensors along dimension 0
    input_dict = {
        'name': 'concat_3_tensors_2d_dim0',
        'concat_dim': np.array(0, dtype=np.int32),
        'values': np.array([
            np.array([[1, 2]], dtype=np.uint8),
            np.array([[3, 4]], dtype=np.uint8),
            np.array([[5, 6]], dtype=np.uint8)
        ], dtype=object),
        'input_mins': np.array([0.0, 1.0, 2.0], dtype=np.float32),
        'input_maxes': np.array([10.0, 11.0, 12.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 5: 3D concatenation along dimension 0
    input_dict = {
        'name': 'concat_3d_dim0',
        'concat_dim': np.array(0, dtype=np.int32),
        'values': np.array([
            np.arange(8, dtype=np.int8).reshape((2, 2, 2)),
            np.arange(8, 16, dtype=np.int8).reshape((2, 2, 2))
        ], dtype=object),
        'input_mins': np.array([-10.0, -20.0], dtype=np.float32),
        'input_maxes': np.array([10.0, 20.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 6: 3D concatenation along dimension 1
    input_dict = {
        'name': 'concat_3d_dim1',
        'concat_dim': np.array(1, dtype=np.int32),
        'values': np.array([
            np.arange(8, dtype=np.int8).reshape((2, 2, 2)),
            np.arange(12, dtype=np.int8).reshape((2, 3, 2))
        ], dtype=object),
        'input_mins': np.array([-10.0, -20.0], dtype=np.float32),
        'input_maxes': np.array([10.0, 20.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 7: 3D concatenation along dimension 2
    input_dict = {
        'name': 'concat_3d_dim2',
        'concat_dim': np.array(2, dtype=np.int32),
        'values': np.array([
            np.arange(8, dtype=np.int8).reshape((2, 2, 2)),
            np.arange(4, dtype=np.int8).reshape((2, 2, 1))
        ], dtype=object),
        'input_mins': np.array([-10.0, -20.0], dtype=np.float32),
        'input_maxes': np.array([10.0, 20.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 8: Concatenating with a tensor that has a zero-sized dimension
    input_dict = {
        'name': 'concat_with_empty_dim',
        'concat_dim': np.array(0, dtype=np.int32),
        'values': np.array([
            np.zeros((0, 5, 5), dtype=np.uint8),
            np.ones((2, 5, 5), dtype=np.uint8)
        ], dtype=object),
        'input_mins': np.array([0.0, 0.0], dtype=np.float32),
        'input_maxes': np.array([1.0, 1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 9: High rank (4D) tensor concatenation
    input_dict = {
        'name': 'concat_4d',
        'concat_dim': np.array(3, dtype=np.int32),
        'values': np.array([
            np.zeros((1, 2, 3, 4), dtype=np.int8),
            np.ones((1, 2, 3, 5), dtype=np.int8)
        ], dtype=object),
        'input_mins': np.array([-128.0, -128.0], dtype=np.float32),
        'input_maxes': np.array([127.0, 127.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 10: Four 1D tensors with varying min/max ranges
    input_dict = {
        'name': 'concat_4_tensors_1d',
        'concat_dim': np.array(0, dtype=np.int32),
        'values': np.array([
            np.array([0, 1], dtype=np.uint8),
            np.array([10], dtype=np.uint8),
            np.array([20, 21, 22], dtype=np.uint8),
            np.array([30], dtype=np.uint8)
        ], dtype=object),
        'input_mins': np.array([-1.0, 5.0, 15.0, 25.0], dtype=np.float32),
        'input_maxes': np.array([1.0, 15.0, 25.0, 35.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QuantizedConcat"] = get_quantized_concat_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QuantizedConcat' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedConcat'.")

check_valid('tf.raw_ops.QuantizedConcat', generated_inputs['tf.raw_ops.QuantizedConcat'], lib="tf", suffix=0)
