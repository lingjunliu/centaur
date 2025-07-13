
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_inplace_sub_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    i = np.array([0], dtype=np.int32)
    v = np.array([[0.5, 1.0]], dtype=np.float32)
    name = "inplace_sub_1"
    input_dict = {"x": tf.convert_to_tensor(x), "i": tf.convert_to_tensor(i), "v": tf.convert_to_tensor(v), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    i = np.array([1], dtype=np.int32)
    v = np.array([[1.5, 2.0]], dtype=np.float32)
    name = "inplace_sub_2"
    input_dict = {"x": tf.convert_to_tensor(x), "i": tf.convert_to_tensor(i), "v": tf.convert_to_tensor(v), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    i = np.array([0, 2], dtype=np.int32)
    v = np.array([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]], dtype=np.float32)
    name = "inplace_sub_3"
    input_dict = {"x": tf.convert_to_tensor(x), "i": tf.convert_to_tensor(i), "v": tf.convert_to_tensor(v), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    i = np.array([0], dtype=np.int32)
    v = np.array([[1, 1]], dtype=np.int32)
    name = "inplace_sub_4"
    input_dict = {"x": tf.convert_to_tensor(x), "i": tf.convert_to_tensor(i), "v": tf.convert_to_tensor(v), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    i = np.array([0, 1], dtype=np.int32)
    v = np.array([[1, 1], [2, 2]], dtype=np.int32)
    name = "inplace_sub_5"
    input_dict = {"x": tf.convert_to_tensor(x), "i": tf.convert_to_tensor(i), "v": tf.convert_to_tensor(v), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    i = np.array([1], dtype=np.int32)
    v = np.array([[1.0, 1.0]], dtype=np.float64)
    name = "inplace_sub_6"
    input_dict = {"x": tf.convert_to_tensor(x), "i": tf.convert_to_tensor(i), "v": tf.convert_to_tensor(v), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=np.float64)
    i = np.array([0], dtype=np.int32)
    v = np.array([[0.5, 0.5, 0.5]], dtype=np.float64)
    name = "inplace_sub_7"
    input_dict = {"x": tf.convert_to_tensor(x), "i": tf.convert_to_tensor(i), "v": tf.convert_to_tensor(v), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    i = np.array([0, 1], dtype=np.int32)
    v = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.int32)
    name = "inplace_sub_8"
    input_dict = {"x": tf.convert_to_tensor(x), "i": tf.convert_to_tensor(i), "v": tf.convert_to_tensor(v), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (negative values)
    x = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    i = np.array([0, 1], dtype=np.int32)
    v = np.array([[0.5, -1.0], [-1.5, 2.0]], dtype=np.float32)
    name = "inplace_sub_9"
    input_dict = {"x": tf.convert_to_tensor(x), "i": tf.convert_to_tensor(i), "v": tf.convert_to_tensor(v), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (different name)
    x = np.array([[5, 10], [15, 20]], dtype=np.int32)
    i = np.array([1], dtype=np.int32)
    v = np.array([[1, 2]], dtype=np.int32)
    name = "another_name"
    input_dict = {"x": tf.convert_to_tensor(x), "i": tf.convert_to_tensor(i), "v": tf.convert_to_tensor(v), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.InplaceSub"] = tf_raw_ops_inplace_sub_inputs()
tf.experimental.numpy.experimental_enable_numpy_behavior()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.InplaceSub' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.InplaceSub'.")

check_valid('tf.raw_ops.InplaceSub', generated_inputs['tf.raw_ops.InplaceSub'], lib="tf", suffix=0)
