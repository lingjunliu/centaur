
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_in_top_k_inputs():
    list_of_inputs = []
    
    # Input 1: Basic case with 2D predictions and 1D targets
    target = np.array([0, 1, 3], dtype=np.int32)
    pred = np.array([
        [1.2, -0.3, 2.8, 5.2],
        [0.1, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.3, 0.3]
    ], dtype=np.float32)
    k = 2
    name = "test1"
    
    input_dict = {
        "targets": target,
        "predictions": pred,
        "k": k,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: With negative values in predictions
    target = np.array([2, 0, 1], dtype=np.int32)
    pred = np.array([
        [-1.0, -2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ], dtype=np.float32)
    k = 2
    name = "test2"
    
    input_dict = {
        "targets": target,
        "predictions": pred,
        "k": k,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: With inf values in predictions
    target = np.array([0, 1], dtype=np.int32)
    pred = np.array([
        [np.inf, 1.0],
        [2.0, np.inf]
    ], dtype=np.float32)
    k = 1
    name = "test3"
    
    input_dict = {
        "targets": target,
        "predictions": pred,
        "k": k,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: With nan values in predictions
    target = np.array([0, 1], dtype=np.int32)
    pred = np.array([
        [np.nan, 1.0],
        [2.0, 3.0]
    ], dtype=np.float32)
    k = 2
    name = "test4"
    
    input_dict = {
        "targets": target,
        "prediction": pred,
        "k": k,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: With ties (multiple classes have same prediction value)
    target = np.array([0, 1], dtype=np.int32)
    pred = np.array([
        [1.0, 1.0],
        [2.0, 2.0]
    ], dtype=np.float32)
    k = 2
    name = "test5"
    
    input_dict = {
        "targets": target,
        "prediction": pred,
        "k": k,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: with large k value
    target = np.array([0, 1], dtype=np.int32)
    pred = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ], dtype=np.float32)
    k = 5
    name = "test6"
    
    input_dict = {
        "targets": target,
        "prediction": pred,
        "k": k,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: with k=1 (edge case)
    target = np.array([2, 0], dtype=np.int32)
    pred = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ], dtype=np.float32)
    k = 1
    name = "test7"
    
    input_dict = {
        "targets": target,
        "prediction": pred,
        "k": k,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: With large number of classes (more than k)
    target = np.array([0, 1, 2], dtype=np.int32)
    pred = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0, 9.0, 10.0],
        [11.0, 12.0, 13.0, 14.0, 15.0]
    ], dtype=np.float32)
    k = 3
    name = "test8"
    
    input_dict = {
        "targets": target,
        "prediction": pred,
        "k": k,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: With different number of classes
    target = np.array([0, 1], dtype=np.int32)
    pred = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ], dtype=np.float32)
    k = 1
    name = "test9"
    
    input_dict = {
        "targets": target,
        "prediction": pred,
        "k": k,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: With different number of classes and negative values
    target = np.array([1, 0], dtype=np.int32)
    pred = np.array([
        [-4.0, -3.0],
        [1.0, 2.0]
    ], dtype=np.float32)
    k = 2
    name = "test10"
    
    input_dict = {
        "targets": target,
        "prediction": pred,
        "k": k,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.in_top_k"] = tf_math_in_top_k_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.in_top_k' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.in_top_k'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.in_top_k', generated_inputs['tf.math.in_top_k'], lib="tf", suffix=0)
