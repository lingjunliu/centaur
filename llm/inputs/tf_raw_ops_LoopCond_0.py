
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_LoopCond_inputs():
    list_of_inputs = []

    # Input 1: Scalar boolean
    input_val = np.array(True, dtype=np.bool_)
    input_dict = {"input": tf.convert_to_tensor(input_val), "name": "loop_cond_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Scalar boolean (False)
    input_val = np.array(False, dtype=np.bool_)
    input_dict = {"input": tf.convert_to_tensor(input_val), "name": "loop_cond_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array, single element
    input_val = np.array([True], dtype=np.bool_)
    input_dict = {"input": tf.convert_to_tensor(input_val[0]), "name": "loop_cond_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 1D array, single element False
    input_val = np.array([False], dtype=np.bool_)
    input_dict = {"input": tf.convert_to_tensor(input_val[0]), "name": "loop_cond_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array, single element
    input_val = np.array([[True]], dtype=np.bool_)
    input_dict = {"input": tf.convert_to_tensor(input_val[0][0]), "name": "loop_cond_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array, single element False
    input_val = np.array([[False]], dtype=np.bool_)
    input_dict = {"input": tf.convert_to_tensor(input_val[0][0]), "name": "loop_cond_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Using different name
    input_val = np.array(True, dtype=np.bool_)
    input_dict = {"input": tf.convert_to_tensor(input_val), "name": "different_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty string name
    input_val = np.array(False, dtype=np.bool_)
    input_dict = {"input": tf.convert_to_tensor(input_val), "name": ""}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array, single element True
    input_val = np.array([[[True]]], dtype=np.bool_)
    input_dict = {"input": tf.convert_to_tensor(input_val[0][0][0]), "name": "loop_cond_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D array, single element False
    input_val = np.array([[[False]]], dtype=np.bool_)
    input_dict = {"input": tf.convert_to_tensor(input_val[0][0][0]), "name": "loop_cond_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: tf.bool
    input_val = tf.constant(True, dtype=tf.bool)
    input_dict = {"input": input_val, "name": "loop_cond_11"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: tf.bool (False)
    input_val = tf.constant(False, dtype=tf.bool)
    input_dict = {"input": input_val, "name": "loop_cond_12"}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.LoopCond"] = tf_raw_ops_LoopCond_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.LoopCond' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.LoopCond'.")

check_valid('tf.raw_ops.LoopCond', generated_inputs['tf.raw_ops.LoopCond'], lib="tf", suffix=0)
