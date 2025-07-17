
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_prevent_gradient_inputs():
    list_of_inputs = []

    # Input 1: Simple float tensor
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"input": input_tensor, "message": "", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Int tensor with a message
    input_tensor = np.array([4, 5, 6], dtype=np.int32)
    input_dict = {"input": input_tensor, "message": "No gradient allowed here!", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float tensor
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    input_dict = {"input": input_tensor, "message": "", "name": "prevent_grad_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Bool tensor
    input_tensor = np.array([True, False, True], dtype=np.bool_)
    input_dict = {"input": input_tensor, "message": "Boolean tensor, no grad!", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex tensor
    input_tensor = np.array([1+1j, 2+2j], dtype=np.complex64)
    input_dict = {"input": input_tensor, "message": "Complex numbers!", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tensor with negative values
    input_tensor = np.array([-1.0, -2.0, 0.0, 1.0, 2.0], dtype=np.float32)
    input_dict = {"input": input_tensor, "message": "", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D tensor
    input_tensor = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict = {"input": input_tensor, "message": "3D tensor, prevent gradient.", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty tensor
    input_tensor = np.array([], dtype=np.float32)
    input_dict = {"input": input_tensor, "message": "Empty tensor", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large tensor
    input_tensor = np.random.rand(100, 100).astype(np.float32)
    input_dict = {"input": input_tensor, "message": "Large tensor, no gradient here.", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Uint8 tensor
    input_tensor = np.array([1, 2, 3], dtype=np.uint8)
    input_dict = {"input": input_tensor, "message": "Uint8 tensor", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.PreventGradient"] = tf_raw_ops_prevent_gradient_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.PreventGradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.PreventGradient'.")

check_valid('tf.raw_ops.PreventGradient', generated_inputs['tf.raw_ops.PreventGradient'], lib="tf", suffix=0)
