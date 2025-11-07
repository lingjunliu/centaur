
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np

def tfexperimental_numpy_promote_types_inputs():
    list_of_inputs = []
    
    # Input 1: int32 and int64
    input_dict = {
        "type1": np.int32,
        "type2": np.int64
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 2: float32 and float64
    input_dict = {
        "type1": np.float32,
        "type2": np.float64
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 3: uint8 and int32
    input_dict = {
        "type1": np.uint8,
        "type2": np.int32
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 4: bool and int64
    input_dict = {
        "type1": np.bool_,
        "type2": np.int64
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 5: complex64 and float32
    input_dict = {
        "type1": np.complex64,
        "type2": np.float32
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 6: int8 and uint8
    input_dict = {
        "type1": np.int8,
        "type2": np.uint8
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 7: float64 and complex128
    input_dict = {
        "type1": np.float64,
        "type2": np.complex128
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 8: int16 and uint32
    input_dict = {
        "type1": np.int16,
        "type2": np.uint32
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 9: float32 and bool
    input_dict = {
        "type1": np.float32,
        "type2": np.bool_
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 10: int64 and complex64
    input_dict = {
        "type1": np.int64,
        "type2": np.complex64
    }
    list_of_inputs.append(input_dict.copy())
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.promote_types"] = tfexperimental_numpy_promote_types_inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.promote_types' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.promote_types'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.promote_types', generated_inputs['tf.experimental.numpy.promote_types'], lib="tf", suffix=0)
