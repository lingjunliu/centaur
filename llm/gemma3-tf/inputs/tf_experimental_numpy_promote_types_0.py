
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_promote_types_inputs():
    list_of_inputs = []

    input_dict = {
        'type1': np.dtype('int32'),
        'type2': np.dtype('float32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'type1': np.dtype('float64'),
        'type2': np.dtype('complex128')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'type1': np.dtype('int8'),
        'type2': np.dtype('int16')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'type1': np.dtype('uint8'),
        'type2': np.dtype('uint16')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'type1': np.dtype('complex64'),
        'type2': np.dtype('complex128')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'type1': np.dtype('float16'),
        'type2': np.dtype('float32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'type1': np.dtype('int64'),
        'type2': np.dtype('int32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'type1': np.dtype('bool'),
        'type2': np.dtype('int32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'type1': np.dtype('object'),
        'type2': np.dtype('str_')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'type1': np.dtype('float32'),
        'type2': np.dtype('float32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.promote_types"] = tf_promote_types_inputs()

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
