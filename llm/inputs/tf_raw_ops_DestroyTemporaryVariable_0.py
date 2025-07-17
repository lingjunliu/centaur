
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_destroy_temporary_variable_inputs():
    list_of_inputs = []

    # Input 1, valid
    ref = tf.Variable(np.array([1, 2, 3], dtype=np.int32)).numpy()
    var_name = "temp_var_1"
    input_dict = {
        "ref": ref,
        "var_name": var_name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    ref = tf.Variable(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)).numpy()
    var_name = "temp_var_2"
    input_dict = {
        "ref": ref,
        "var_name": var_name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    ref = tf.Variable(np.array([True, False, True], dtype=np.bool_)).numpy()
    var_name = "temp_var_3"
    input_dict = {
        "ref": ref,
        "var_name": var_name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid
    ref = tf.Variable(np.array([[-1, -2], [-3, -4]], dtype=np.int64)).numpy()
    var_name = "temp_var_4"
    input_dict = {
        "ref": ref,
        "var_name": var_name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid
    ref = tf.Variable(np.array([1.5, 2.5, 3.5], dtype=np.float64)).numpy()
    var_name = "temp_var_5"
    input_dict = {
        "ref": ref,
        "var_name": var_name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid
    ref = tf.Variable(np.array([b'a', b'b', b'c'], dtype=np.string_)).numpy()
    var_name = "temp_var_6"
    input_dict = {
        "ref": ref,
        "var_name": var_name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid
    ref = tf.Variable(np.array([complex(1,1), complex(2,2)], dtype=np.complex64)).numpy()
    var_name = "temp_var_7"
    input_dict = {
        "ref": ref,
        "var_name": var_name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid - empty array
    ref = tf.Variable(np.array([], dtype=np.int32)).numpy()
    var_name = "temp_var_8"
    input_dict = {
        "ref": ref,
        "var_name": var_name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid
    ref = tf.Variable(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)).numpy()
    var_name = "temp_var_9"
    input_dict = {
        "ref": ref,
        "var_name": var_name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid
    ref = tf.Variable(np.array(10, dtype=np.int32)).numpy()
    var_name = "temp_var_10"
    input_dict = {
        "ref": ref,
        "var_name": var_name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    for i in range(len(list_of_inputs)):
        list_of_inputs[i]['name'] = f'destroy_op_{i+1}'

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DestroyTemporaryVariable"] = tf_raw_ops_destroy_temporary_variable_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DestroyTemporaryVariable' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DestroyTemporaryVariable'.")

check_valid('tf.raw_ops.DestroyTemporaryVariable', generated_inputs['tf.raw_ops.DestroyTemporaryVariable'], lib="tf", suffix=0)
