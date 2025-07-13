
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_approximate_equal_inputs():
    list_of_inputs = []

    # Input 1: Basic float comparison
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([1.01, 1.99, 3.0], dtype=np.float32)
    tolerance = 0.05
    name = "basic_float"
    input_dict = {"x": x, "y": y, "tolerance": tolerance, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer comparison
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([1, 2, 4], dtype=np.int32)
    tolerance = 1.0
    name = "integer_comparison"
    input_dict = {"x": x, "y": y, "tolerance": tolerance, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Large tolerance
    x = np.array([1.0, 2.0], dtype=np.float64)
    y = np.array([5.0, 1.0], dtype=np.float64)
    tolerance = 5.0
    name = "large_tolerance"
    input_dict = {"x": x, "y": y, "tolerance": tolerance, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative values
    x = np.array([-1.0, -2.0], dtype=np.float32)
    y = np.array([-1.01, -1.99], dtype=np.float32)
    tolerance = 0.05
    name = "negative_values"
    input_dict = {"x": x, "y": y, "tolerance": tolerance, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Zero tolerance
    x = np.array([1.0, 2.0], dtype=np.float32)
    y = np.array([1.0, 2.0], dtype=np.float32)
    tolerance = 0.0
    name = "zero_tolerance"
    input_dict = {"x": x, "y": y, "tolerance": tolerance, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multi-dimensional array
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([[1.1, 1.9], [3.0, 4.1]], dtype=np.float32)
    tolerance = 0.2
    name = "multi_dimensional"
    input_dict = {"x": x, "y": y, "tolerance": tolerance, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7: Different magnitudes
    x = np.array([1e-6, 1e6], dtype=np.float32)
    y = np.array([1.1e-6, 0.9e6], dtype=np.float32)
    tolerance = 0.2e6
    name = "different_magnitudes"
    input_dict = {"x": x, "y": y, "tolerance": tolerance, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Integer comparison with negative tolerance
    x = np.array([-1, -2, -3], dtype=np.int32)
    y = np.array([-1, -2, -4], dtype=np.int32)
    tolerance = 1.0
    name = "integer_comparison_negative"
    input_dict = {"x": x, "y": y, "tolerance": tolerance, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small tolerance with small numbers
    x = np.array([1e-8, 2e-8], dtype=np.float32)
    y = np.array([1.1e-8, 1.9e-8], dtype=np.float32)
    tolerance = 0.2e-8
    name = "small_tolerance_small_numbers"
    input_dict = {"x": x, "y": y, "tolerance": tolerance, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: uint8 comparison
    x = np.array([100, 200], dtype=np.uint8)
    y = np.array([105, 195], dtype=np.uint8)
    tolerance = 10.0
    name = "uint8_comparison"
    input_dict = {"x": x, "y": y, "tolerance": tolerance, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: complex64 comparison
    x = np.array([1+1j, 2+2j], dtype=np.complex64)
    y = np.array([1.1+0.9j, 1.9+2.1j], dtype=np.complex64)
    tolerance = 0.3
    name = "complex64_comparison"
    input_dict = {"x": x, "y": y, "tolerance": tolerance, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ApproximateEqual"] = tf_raw_ops_approximate_equal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ApproximateEqual' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApproximateEqual'.")

check_valid('tf.raw_ops.ApproximateEqual', generated_inputs['tf.raw_ops.ApproximateEqual'], lib="tf", suffix=0)
