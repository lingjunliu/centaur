
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_PreventGradient_inputs():
    list_of_inputs = []

    # Input 1: Basic integer tensor
    input1 = np.array([1, 2, 3], dtype=np.int32)
    input_dict1 = {"input": input1, "message": "Gradient not allowed here!", "name": "prevent_grad_1"}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Float tensor with a different message
    input2 = np.array([1.0, 2.5, -3.2], dtype=np.float32)
    input_dict2 = {"input": input2, "message": "No differentiation here", "name": "prevent_grad_2"}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Empty string message
    input3 = np.array([[1, 2], [3, 4]], dtype=np.int64)
    input_dict3 = {"input": input3, "message": "", "name": "prevent_grad_3"}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Boolean tensor
    input4 = np.array([True, False, True], dtype=np.bool_)
    input_dict4 = {"input": input4, "message": "Boolean gradient is forbidden", "name": "prevent_grad_4"}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Multi-dimensional float tensor
    input5 = np.random.rand(2, 3, 4).astype(np.float64)
    input_dict5 = {"input": input5, "message": "High-dimensional gradient block", "name": "prevent_grad_5"}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Complex tensor
    input6 = np.array([1+1j, 2-2j, 3+0j], dtype=np.complex128)
    input_dict6 = {"input": input6, "message": "Complex gradients are tricky!", "name": "prevent_grad_6"}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Negative values
    input7 = np.array([-1, -2, -3], dtype=np.int32)
    input_dict7 = {"input": input7, "message": "Preventing gradient for negative values", "name": "prevent_grad_7"}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Zero dimension array
    input8 = np.array(10, dtype=np.int32)
    input_dict8 = {"input": input8, "message": "Scalar gradient prevention", "name": "prevent_grad_8"}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Larger tensor
    input9 = np.random.randint(0, 100, size=(10, 10)).astype(np.int32)
    input_dict9 = {"input": input9, "message": "Large tensor gradient prevention", "name": "prevent_grad_9"}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Tensor with zero values.
    input10 = np.array([0, 0, 0], dtype=np.int32)
    input_dict10 = {"input": input10, "message": "Prevent gradient for zero values", "name": "prevent_grad_10"}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.PreventGradient"] = tf_raw_ops_PreventGradient_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.PreventGradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.PreventGradient'.")

check_valid('tf.raw_ops.PreventGradient', generated_inputs['tf.raw_ops.PreventGradient'], lib="tf", suffix=0)
