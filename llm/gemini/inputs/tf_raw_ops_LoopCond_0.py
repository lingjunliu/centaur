
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_loopcond_inputs():
    list_of_inputs = []

    # Input 1: bool scalar True
    input_tensor = np.array(True, dtype=np.bool_)
    input_dict = {"name": "loop_cond_true", "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: bool scalar False
    input_tensor = np.array(False, dtype=np.bool_)
    input_dict = {"name": "loop_cond_false", "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: bool tensor rank 0 True
    input_tensor = np.array(True, dtype=np.bool_)
    input_dict = {"name": "loop_cond_rank0_true", "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: bool tensor rank 0 False
    input_tensor = np.array(False, dtype=np.bool_)
    input_dict = {"name": "loop_cond_rank0_false", "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: bool array (rank 0) True
    input_tensor = np.array(True, dtype=np.bool_)
    input_dict = {"name": "loop_cond_arr0_true", "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: bool array (rank 0) False
    input_tensor = np.array(False, dtype=np.bool_)
    input_dict = {"name": "loop_cond_arr0_false", "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7:  True with different name
    input_tensor = np.array(True, dtype=np.bool_)
    input_dict = {"name": "different_name", "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: False with different name
    input_tensor = np.array(False, dtype=np.bool_)
    input_dict = {"name": "another_name", "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Rank 0 bool true with empty name
    input_tensor = np.array(True, dtype=np.bool_)
    input_dict = {"name": "", "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Rank 0 bool false with empty name
    input_tensor = np.array(False, dtype=np.bool_)
    input_dict = {"name": "", "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.LoopCond"] = tf_raw_ops_loopcond_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.LoopCond' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.LoopCond'.")

check_valid('tf.raw_ops.LoopCond', generated_inputs['tf.raw_ops.LoopCond'], lib="tf", suffix=0)
