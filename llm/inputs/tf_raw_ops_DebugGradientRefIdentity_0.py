
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_DebugGradientRefIdentity_inputs():
    list_of_inputs = []

    # Input 1: Simple float32 tensor
    input_tensor = tf.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    input_dict = {"input": input_tensor.ref(), "name": "debug_grad_id_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Int32 tensor with negative values
    input_tensor = tf.Variable(np.array([-1, 0, 1], dtype=np.int32))
    input_dict = {"input": input_tensor.ref(), "name": "debug_grad_id_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 tensor
    input_tensor = tf.Variable(np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float64))
    input_dict = {"input": input_tensor.ref(), "name": "debug_grad_id_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Bool tensor
    input_tensor = tf.Variable(np.array([True, False, True], dtype=np.bool_))
    input_dict = {"input": input_tensor.ref(), "name": "debug_grad_id_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex64 tensor
    input_tensor = tf.Variable(np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64))
    input_dict = {"input": input_tensor.ref(), "name": "debug_grad_id_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D int64 tensor
    input_tensor = tf.Variable(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64))
    input_dict = {"input": input_tensor.ref(), "name": "debug_grad_id_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty float32 tensor
    input_tensor = tf.Variable(np.array([], dtype=np.float32))
    input_dict = {"input": input_tensor.ref(), "name": "debug_grad_id_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tensor with zero dimension
    input_tensor = tf.Variable(np.zeros((0, 5), dtype=np.float32))
    input_dict = {"input": input_tensor.ref(), "name": "debug_grad_id_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Float16 tensor
    input_tensor = tf.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float16))
    input_dict = {"input": input_tensor.ref(), "name": "debug_grad_id_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Bfloat16 tensor
    input_tensor = tf.Variable(np.array([1.0, 2.0, 3.0], dtype=tf.bfloat16.as_numpy_dtype))
    input_dict = {"input": input_tensor.ref(), "name": "debug_grad_id_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DebugGradientRefIdentity"] = tf_raw_ops_DebugGradientRefIdentity_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DebugGradientRefIdentity' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DebugGradientRefIdentity'.")

check_valid('tf.raw_ops.DebugGradientRefIdentity', generated_inputs['tf.raw_ops.DebugGradientRefIdentity'], lib="tf", suffix=0)
