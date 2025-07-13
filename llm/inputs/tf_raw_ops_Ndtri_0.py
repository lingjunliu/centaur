
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ndtri_inputs():
    list_of_inputs = []

    # Input 1: Simple float32 tensor
    x = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 tensor
    x = np.array([-0.5, 0, 1.2], dtype=np.float64)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float64), "name": "my_ndtri"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: half tensor
    x = np.array([0.2, 0.7, -0.3], dtype=np.float16)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float16), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float32 tensor with a different name
    x = np.array([0.4, 0.6, -0.4], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32), "name": "another_ndtri"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multidimensional float32 tensor
    x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multidimensional float64 tensor with negative values
    x = np.array([[-0.1, 0.2], [0.3, -0.4]], dtype=np.float64)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float64), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger float32 tensor
    x = np.random.rand(5, 5).astype(np.float32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float32 with only one element
    x = np.array([0.5], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Another multidimensional tensor with a name
    x = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32), "name": "yet_another_ndtri"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float16 tensor with negative values
    x = np.array([-0.2, 0.1, -0.5], dtype=np.float16)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float16), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs
generated_inputs = {}
generated_inputs["tf.raw_ops.Ndtri"] = tf_raw_ops_ndtri_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Ndtri' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Ndtri'.")

check_valid('tf.raw_ops.Ndtri', generated_inputs['tf.raw_ops.Ndtri'], lib="tf", suffix=0)
