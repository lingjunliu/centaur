
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_sets_intersection_inputs():
    list_of_inputs = []

    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[2, 4, -6], [5, 7, 9]])
    validate_indices = True
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    b = np.array([[[3, 5, 7], [8, 10, 12]], [[1, 2, 4], [6, 9, 11]]])
    validate_indices = False
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[[1], [2]], [[3], [4]]])
    b = np.array([[[1, 5], [2, 6]], [[3, 7], [4, 8]]])
    validate_indices = True
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[[1, 2, 3]]])
    b = np.array([[[1, 2]]])
    validate_indices = False
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([[[1, -2, 3]]])
    b = np.array([[[1, 2]]])
    validate_indices = True
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([[[1, 2], [3, 4]]])
    b = np.array([[[1, 2], [3, 4]]])
    validate_indices = False
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.sets.intersection"] = tf_sets_intersection_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sets.intersection' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sets.intersection'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.sets.intersection', generated_inputs['tf.sets.intersection'], lib="tf", suffix=0)
