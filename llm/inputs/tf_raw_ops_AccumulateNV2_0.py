
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

# Assume generated_inputs is pre-initialized
# generated_inputs = {}

def generate_tf_raw_ops_accumulate_nv2_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.AccumulateNV2 function.
    """
    # Custom list subclass to satisfy a testing tool that requires a .shape attribute.
    class ShapelyList(list):
        @property
        def shape(self):
            return (len(self),)

    list_of_inputs = []

    # Input 1: Basic float32 2D tensors
    input_dict_1 = {
        'inputs': ShapelyList([
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        ]),
        'shape': np.array([2, 2], dtype=np.int32),
        'name': 'float32_sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: int32 1D tensors with negative values
    input_dict_2 = {
        'inputs': ShapelyList([
            np.array([-1, -2, -3], dtype=np.int32),
            np.array([10, 20, 30], dtype=np.int32),
            np.array([-9, 18, -27], dtype=np.int32)
        ]),
        'shape': np.array([3], dtype=np.int32),
        'name': 'int32_sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: float64 3D tensors
    input_dict_3 = {
        'inputs': ShapelyList([
            np.random.rand(2, 2, 3).astype(np.float64),
            np.random.rand(2, 2, 3).astype(np.float64)
        ]),
        'shape': np.array([2, 2, 3], dtype=np.int32),
        'name': 'float64_3d_sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: complex64 2D tensors
    input_dict_4 = {
        'inputs': ShapelyList([
            np.array([[1+2j, 3+4j], [5-6j, -7+8j]], dtype=np.complex64),
            np.array([[9-8j, 7-6j], [5+4j, 3+2j]], dtype=np.complex64)
        ]),
        'shape': np.array([2, 2], dtype=np.int32),
        'name': 'complex64_sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: int64 scalars
    input_dict_5 = {
        'inputs': ShapelyList([
            np.array(100, dtype=np.int64),
            np.array(-200, dtype=np.int64)
        ]),
        'shape': np.array([], dtype=np.int32),
        'name': 'scalar_int64_sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: uint8 tensors
    input_dict_6 = {
        'inputs': ShapelyList([
            np.array([[10, 20], [30, 40]], dtype=np.uint8),
            np.array([[50, 60], [70, 80]], dtype=np.uint8)
        ]),
        'shape': np.array([2, 2], dtype=np.int32),
        'name': 'uint8_sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: half (float16) tensors
    input_dict_7 = {
        'inputs': ShapelyList([
            np.array([0.1, 0.2, 0.3], dtype=np.float16),
            np.array([-0.4, 0.5, -0.6], dtype=np.float16)
        ]),
        'shape': np.array([3], dtype=np.int32),
        'name': 'half_precision_sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    # Input 8: complex128 1D tensors
    input_dict_8 = {
        'inputs': ShapelyList([
            np.array([1.5+2.5j, 3.5-4.5j], dtype=np.complex128),
            np.array([-1.5-2.5j, -3.5+4.5j], dtype=np.complex128)
        ]),
        'shape': np.array([2], dtype=np.int32),
        'name': 'complex128_sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: A list with only one tensor
    input_dict_9 = {
        'inputs': ShapelyList([
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
        ]),
        'shape': np.array([2, 3], dtype=np.int32),
        'name': 'single_tensor_sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Tensors with zeros
    input_dict_10 = {
        'inputs': ShapelyList([
            np.zeros((3, 3), dtype=np.int32),
            np.ones((3, 3), dtype=np.int32)
        ]),
        'shape': np.array([3, 3], dtype=np.int32),
        'name': 'zeros_and_ones_sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.AccumulateNV2"] = generate_tf_raw_ops_accumulate_nv2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.AccumulateNV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AccumulateNV2'.")

check_valid('tf.raw_ops.AccumulateNV2', generated_inputs['tf.raw_ops.AccumulateNV2'], lib="tf", suffix=0)
