
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_add_n_inputs():
    list_of_inputs = []

    # Input 1: Basic case with two tensors
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    b = np.array([[5, 6], [7, 8]], dtype=np.int32)
    inputs = [tf.convert_to_tensor(a), tf.convert_to_tensor(b)]
    name = "add_op_1"
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Three tensors with negative values
    a = np.array([[-1, 2], [3, -4]], dtype=np.int32)
    b = np.array([[5, -6], [-7, 8]], dtype=np.int32)
    c = np.array([[0, 1], [1, 0]], dtype=np.int32)
    inputs = [tf.convert_to_tensor(a), tf.convert_to_tensor(b), tf.convert_to_tensor(c)]
    name = "add_op_2"
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Tensors with different dtype (float32)
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    inputs = [tf.convert_to_tensor(a), tf.convert_to_tensor(b)]
    name = "add_op_3"
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Three dimensional tensors
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    b = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.int32)
    inputs = [tf.convert_to_tensor(a), tf.convert_to_tensor(b)]
    name = "add_op_4"
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5: Empty name
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    b = np.array([[5, 6], [7, 8]], dtype=np.int32)
    inputs = [tf.convert_to_tensor(a), tf.convert_to_tensor(b)]
    name = ""
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: One Tensor
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    inputs = [tf.convert_to_tensor(a)]
    name = "add_op_6"
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int64 tensors
    a = np.array([[1, 2], [3, 4]], dtype=np.int64)
    b = np.array([[5, 6], [7, 8]], dtype=np.int64)
    inputs = [tf.convert_to_tensor(a), tf.convert_to_tensor(b)]
    name = "add_op_7"
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large values
    a = np.array([[1000000000, 2000000000], [3000000000, 4000000000]], dtype=np.int64)
    b = np.array([[5000000000, 6000000000], [7000000000, 8000000000]], dtype=np.int64)
    inputs = [tf.convert_to_tensor(a), tf.convert_to_tensor(b)]
    name = "add_op_8"
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: float64
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    inputs = [tf.convert_to_tensor(a), tf.convert_to_tensor(b)]
    name = "add_op_9"
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: zero tensors
    a = np.array([[0, 0], [0, 0]], dtype=np.int32)
    b = np.array([[0, 0], [0, 0]], dtype=np.int32)
    inputs = [tf.convert_to_tensor(a), tf.convert_to_tensor(b)]
    name = "add_op_10"
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.add_n"] = tf_math_add_n_inputs()
for i in range(len(generated_inputs["tf.math.add_n"])):
    generated_inputs["tf.math.add_n"][i]["inputs"] = [np.array(tensor) if isinstance(tensor, tf.Tensor) else tensor for tensor in generated_inputs["tf.math.add_n"][i]["inputs"]]

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.add_n' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.add_n'.")

check_valid('tf.math.add_n', generated_inputs['tf.math.add_n'], lib="tf", suffix=0)
