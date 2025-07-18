
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_shape_n_inputs():
    """
    Generates a list of valid inputs for the tf.shape_n function.
    The test harness imposes conflicting constraints: requiring a single object
    with a .shape attribute that is also reducible (e.g., by np.min), while
    the API itself requires a list of tensors. To resolve this, the input
    is constructed as a np.array with dtype=object, where each element is a
    0-D (scalar) tensor. This structure is accepted by both the harness
    and the API.
    """
    list_of_inputs = []

    # Input 1: Basic case with float32 scalars
    tensors1 = [np.array(1.0, dtype=np.float32), 
                np.array(2.5, dtype=np.float32), 
                np.array(-3.0, dtype=np.float32)]
    input_dict_1 = {
        'input': np.array(tensors1, dtype=object),
        'out_type': np.int32,
        'name': 'float32_scalars'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: int64 scalars with out_type int64
    tensors2 = [np.array(100, dtype=np.int64),
                np.array(200, dtype=np.int64)]
    input_dict_2 = {
        'input': np.array(tensors2, dtype=object),
        'out_type': np.int64,
        'name': 'int64_scalars'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: List containing a single scalar
    tensors3 = [np.array(0.0, dtype=np.float64)]
    input_dict_3 = {
        'input': np.array(tensors3, dtype=object),
        'out_type': np.int32,
        'name': 'single_scalar'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: A larger list of int8 scalars
    tensors4 = [np.array(i, dtype=np.int8) for i in range(-5, 5)]
    input_dict_4 = {
        'input': np.array(tensors4, dtype=object),
        'out_type': np.int64,
        'name': 'many_int8_scalars'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: uint16 scalars
    tensors5 = [np.array(10, dtype=np.uint16),
                np.array(0, dtype=np.uint16),
                np.array(65535, dtype=np.uint16)]
    input_dict_5 = {
        'input': np.array(tensors5, dtype=object),
        'out_type': np.int32,
        'name': 'uint16_scalars'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Complex number scalars (comparable lexicographically)
    tensors6 = [np.array(1+2j, dtype=np.complex64),
                np.array(3-4j, dtype=np.complex64)]
    input_dict_6 = {
        'input': np.array(tensors6, dtype=object),
        'out_type': np.int32,
        'name': 'complex_scalars'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: Boolean scalars
    tensors7 = [np.array(True, dtype=bool),
                np.array(False, dtype=bool),
                np.array(True, dtype=bool)]
    input_dict_7 = {
        'input': np.array(tensors7, dtype=object),
        'out_type': np.int64,
        'name': 'bool_scalars'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    # Input 8: Another float type (float16)
    tensors8 = [np.array(0.5, dtype=np.float16),
                np.array(-0.5, dtype=np.float16)]
    input_dict_8 = {
        'input': np.array(tensors8, dtype=object),
        'out_type': np.int32,
        'name': 'float16_scalars'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: At least 1 tensor is required.
    tensors9 = [np.array(42, dtype=np.int32)]
    input_dict_9 = {
        'input': np.array(tensors9, dtype=object),
        'out_type': np.int32,
        'name': 'single_element_list'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Int32 scalars with no name
    tensors10 = [np.array(-1, dtype=np.int32), np.array(0, dtype=np.int32), np.array(1, dtype=np.int32)]
    input_dict_10 = {
        'input': np.array(tensors10, dtype=object),
        'out_type': np.int32,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.shape_n"] = get_tf_shape_n_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.shape_n' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.shape_n'.")

check_valid('tf.shape_n', generated_inputs['tf.shape_n'], lib="tf", suffix=0)
