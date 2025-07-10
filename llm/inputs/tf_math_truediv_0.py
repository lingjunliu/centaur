
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_truediv_inputs():
    list_of_inputs = []

    # Input 1: Basic float division
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    name = "basic_float"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer division, automatic casting
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([2, 2, 2], dtype=np.int32)
    name = "integer_auto_cast"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different shapes, broadcasting
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array(2.0, dtype=np.float32)
    name = "broadcasting"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative values
    x = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
    y = np.array([2.0, -2.0, 2.0], dtype=np.float32)
    name = "negative_values"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional array
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32)
    name = "multi_dimensional"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Integer division with int64
    x = np.array([1, 2, 3], dtype=np.int64)
    y = np.array([2, 2, 2], dtype=np.int64)
    name = "integer_int64"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Division by zero (will produce inf)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([0.0, 2.0, 2.0], dtype=np.float32)
    name = "division_by_zero"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger tensor
    x = np.random.rand(10, 10).astype(np.float32)
    y = np.random.rand(10, 10).astype(np.float32)
    name = "larger_tensor"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: x is scalar
    x = np.array(5.0, dtype=np.float32)
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    name = "x_scalar"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: x and y are both scalars
    x = np.array(5.0, dtype=np.float32)
    y = np.array(2.0, dtype=np.float32)
    name = "both_scalars"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_math_truediv_inputs()
for i in range(len(inputs)):
    inputs[i]['x'] = np.array(inputs[i]['x'])
    inputs[i]['y'] = np.array(inputs[i]['y'])
generated_inputs["tf.math.truediv"] = inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.truediv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.truediv'.")

check_valid('tf.math.truediv', generated_inputs['tf.math.truediv'], lib="tf", suffix=0)
