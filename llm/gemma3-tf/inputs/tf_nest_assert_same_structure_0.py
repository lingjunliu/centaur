
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_nest_assert_same_structure_inputs():
    list_of_inputs = []
    input_dict1 = {
        "nest1": np.array([1, 2, 3]),
        "nest2": np.array([4, 5, 6]),
        "check_types": True,
        "expand_composites": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input_dict2 = {
        "nest1": np.array([[1, 2], [3, 4]]),
        "nest2": np.array([[5, 6], [7, 8]]),
        "check_types": True,
        "expand_composites": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input_dict3 = {
        "nest1": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
        "nest2": np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]]),
        "check_types": True,
        "expand_composites": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input_dict4 = {
        "nest1": np.array([1, 2, 3]),
        "nest2": np.array([[4, 5], [6, 7]]),
        "check_types": False,
        "expand_composites": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input_dict5 = {
        "nest1": np.array([-1, -2, -3]),
        "nest2": np.array([1, 2, 3]),
        "check_types": True,
        "expand_composites": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input_dict6 = {
        "nest1": np.array([1.0, 2.0, 3.0]),
        "nest2": np.array([4.0, 5.0, 6.0]),
        "check_types": True,
        "expand_composites": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input_dict7 = {
        "nest1": np.array([1, 2]),
        "nest2": np.array([[3, 4], [5, 6]]),
        "check_types": False,
        "expand_composites": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input_dict8 = {
        "nest1": np.array([]),
        "nest2": np.array([]),
        "check_types": True,
        "expand_composites": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input_dict9 = {
        "nest1": np.array([1]),
        "nest2": np.array([[2]]),
        "check_types": False,
        "expand_composites": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input_dict10 = {
        "nest1": np.array([1, 2, 3]),
        "nest2": np.array([1, 2, 3]),
        "check_types": True,
        "expand_composites": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.nest.assert_same_structure"] = tf_nest_assert_same_structure_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nest.assert_same_structure' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nest.assert_same_structure'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nest.assert_same_structure', generated_inputs['tf.nest.assert_same_structure'], lib="tf", suffix=0)
