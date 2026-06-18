
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_stateless_parameterized_truncated_normal_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D shape with 0-D parameters
    input_dict = {
        'shape': np.array([5], dtype=np.int32),
        'seed': np.array([1, 2], dtype=np.int32),
        'means': np.array(0.0, dtype=np.float32),
        'stddevs': np.array(1.0, dtype=np.float32),
        'minvals': np.array(-2.0, dtype=np.float32),
        'maxvals': np.array(2.0, dtype=np.float32),
        'name': 'simple_1d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D shape with suffix-matching 1D broadcasted parameter
    input_dict = {
        'shape': np.array([3, 4], dtype=np.int32),
        'seed': np.array([42, 43], dtype=np.int32),
        'means': np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        'stddevs': np.array(0.5, dtype=np.float32),
        'minvals': np.array(-1.0, dtype=np.float32),
        'maxvals': np.array(5.0, dtype=np.float32),
        'name': 'broadcast_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float64 parameters with matching 2D shape suffix
    input_dict = {
        'shape': np.array([10, 1], dtype=np.int64),
        'seed': np.array([100, 200], dtype=np.int64),
        'means': np.array([-5.0], dtype=np.float64),
        'stddevs': np.array([2.5], dtype=np.float64),
        'minvals': np.array([-10.0], dtype=np.float64),
        'maxvals': np.array([0.0], dtype=np.float64),
        'name': 'float64_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D shape, multi-dimensional broadcasting resulting in exact suffix [3, 4]
    input_dict = {
        'shape': np.array([2, 3, 4], dtype=np.int32),
        'seed': np.array([5, 5], dtype=np.int32),
        'means': np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
        'stddevs': np.array([1.0, 1.5, 2.0, 2.5], dtype=np.float32),
        'minvals': np.array(-5.0, dtype=np.float32),
        'maxvals': np.array(5.0, dtype=np.float32),
        'name': 'multi_broadcast_3d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Extremely tight boundaries with 0-D parameters
    input_dict = {
        'shape': np.array([100], dtype=np.int32),
        'seed': np.array([7, 11], dtype=np.int32),
        'means': np.array(0.0, dtype=np.float32),
        'stddevs': np.array(1.0, dtype=np.float32),
        'minvals': np.array(-0.1, dtype=np.float32),
        'maxvals': np.array(0.1, dtype=np.float32),
        'name': 'tight_bounds'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large standard deviation and matched exact-shape parameters
    input_dict = {
        'shape': np.array([2, 2], dtype=np.int32),
        'seed': np.array([12, 34], dtype=np.int32),
        'means': np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        'stddevs': np.array([[100.0, 100.0], [100.0, 100.0]], dtype=np.float32),
        'minvals': np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        'maxvals': np.array([[50.0, 50.0], [50.0, 50.0]], dtype=np.float32),
        'name': 'large_stddev'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: One-sided truncation with 0-D parameters
    input_dict = {
        'shape': np.array([5, 5], dtype=np.int32),
        'seed': np.array([99, 99], dtype=np.int32),
        'means': np.array(0.0, dtype=np.float32),
        'stddevs': np.array(1.0, dtype=np.float32),
        'minvals': np.array(0.0, dtype=np.float32),
        'maxvals': np.array(1000.0, dtype=np.float32),
        'name': 'one_sided_truncation'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Deeply nested shape with exact suffix matching 3D tensors
    input_dict = {
        'shape': np.array([2, 2, 2, 2], dtype=np.int32),
        'seed': np.array([1, 1], dtype=np.int32),
        'means': np.zeros((2, 2, 2), dtype=np.float32),
        'stddevs': np.ones((2, 2, 2), dtype=np.float32),
        'minvals': np.ones((2, 2, 2), dtype=np.float32) * -3.0,
        'maxvals': np.ones((2, 2, 2), dtype=np.float32) * 3.0,
        'name': 'deep_4d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Suffix-matching 1D parameter broadcasted to shape [8]
    input_dict = {
        'shape': np.array([8], dtype=np.int32),
        'seed': np.array([456, 789], dtype=np.int32),
        'means': np.array(-10.0, dtype=np.float32),
        'stddevs': np.array(5.0, dtype=np.float32),
        'minvals': np.array(-20.0, dtype=np.float32),
        'maxvals': np.array([-5.0] * 8, dtype=np.float32),
        'name': 'negative_means'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D shape with 0-D parameters
    input_dict = {
        'shape': np.array([3, 3, 3], dtype=np.int32),
        'seed': np.array([0, 0], dtype=np.int32),
        'means': np.array(5.0, dtype=np.float32),
        'stddevs': np.array(0.1, dtype=np.float32),
        'minvals': np.array(4.8, dtype=np.float32),
        'maxvals': np.array(5.2, dtype=np.float32),
        'name': 'scalar_broadcast'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.random.stateless_parameterized_truncated_normal"] = tf_random_stateless_parameterized_truncated_normal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.random.stateless_parameterized_truncated_normal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.stateless_parameterized_truncated_normal'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.random.stateless_parameterized_truncated_normal', generated_inputs['tf.random.stateless_parameterized_truncated_normal'], lib="tf", suffix=0)
