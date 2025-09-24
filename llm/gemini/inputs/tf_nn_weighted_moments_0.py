
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_nn_weighted_moments_inputs():
    """
    Generates a list of valid inputs for the tf.nn.weighted_moments function.
    To avoid internal TensorFlow errors, all inputs use `keepdims=True` and
    ensure that the dtypes of `x` and `frequency_weights` are consistent floating-point types.
    """
    list_of_inputs = []

    # Input 1: Basic 1D case with float32
    input_dict_1 = {
        'x': np.array([1., 2., 3., 4., 5.], dtype=np.float32),
        'axes': np.array([0], dtype=np.int32),
        'frequency_weights': np.array([1., 0.5, 1., 0.5, 1.], dtype=np.float32),
        'keepdims': True,
        'name': 'basic_1d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic 2D case, reduce along axis 0
    input_dict_2 = {
        'x': np.array([[1., 2.], [3., 4.]], dtype=np.float32),
        'axes': np.array([0], dtype=np.int32),
        'frequency_weights': np.array([[0.5, 1.5], [1.0, 1.0]], dtype=np.float32),
        'keepdims': True,
        'name': '2d_axis0'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 2D case, reduce along axis 1 with keepdims=True
    input_dict_3 = {
        'x': np.array([[1., 2.], [3., 4.]], dtype=np.float32),
        'axes': np.array([1], dtype=np.int32),
        'frequency_weights': np.array([[0.5, 1.5], [1.0, 1.0]], dtype=np.float32),
        'keepdims': True,
        'name': '2d_axis1_keepdims'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 2D case, reduce all elements (multiple axes)
    input_dict_4 = {
        'x': np.array([[-1., 2., -5.], [3., -4., 6.]], dtype=np.float32),
        'axes': np.array([0, 1], dtype=np.int32),
        'frequency_weights': np.array([[1., 2., 1.], [3., 1., 2.]], dtype=np.float32),
        'keepdims': True,
        'name': '2d_all_axes_negative_x'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Broadcastable scalar weight
    input_dict_5 = {
        'x': np.array([[1., 2.], [3., 4.]], dtype=np.float32),
        'axes': np.array([0], dtype=np.int32),
        'frequency_weights': np.array(2.0, dtype=np.float32),
        'keepdims': True,
        'name': 'broadcast_scalar_weight'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Broadcastable vector weight
    input_dict_6 = {
        'x': np.arange(12, dtype=np.float32).reshape(3, 4),
        'axes': np.array([0], dtype=np.int32),
        'frequency_weights': np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32),
        'keepdims': True,
        'name': 'broadcast_vector_weight'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: 3D tensor, reduce along one axis
    input_dict_7 = {
        'x': np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        'axes': np.array([1], dtype=np.int32),
        'frequency_weights': np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4),
        'keepdims': True,
        'name': '3d_axis1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: 3D tensor, reduce multiple axes with keepdims=True
    input_dict_8 = {
        'x': np.array([[[-1., 2.], [-3., 4.]], [[5., -6.], [7., -8.]]], dtype=np.float32),
        'axes': np.array([0, 2], dtype=np.int32),
        'frequency_weights': np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8], dtype=np.float32).reshape(2, 2, 2),
        'keepdims': True,
        'name': '3d_multi_axis_keepdims'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Float64 dtype for x and weights
    input_dict_9 = {
        'x': np.array([[10., 20.], [30., 40.]], dtype=np.float64),
        'axes': np.array([0, 1], dtype=np.int32),
        'frequency_weights': np.array([[1., 0.5], [0.25, 0.125]], dtype=np.float64),
        'keepdims': True,
        'name': 'float64_dtype'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: 4D tensor with broadcastable weights across dimensions
    input_dict_10 = {
        'x': np.ones((2, 3, 4, 5), dtype=np.float32),
        'axes': np.array([1, 3], dtype=np.int32),
        'frequency_weights': np.arange(1, 6, dtype=np.float32).reshape(1, 1, 1, 5),
        'keepdims': True,
        'name': '4d_broadcast_weights'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))
    
    # Input 11: Reduce along negative axis
    input_dict_11 = {
        'x': np.arange(12, dtype=np.float32).reshape(3, 4),
        'axes': np.array([-1], dtype=np.int32),
        'frequency_weights': np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32),
        'keepdims': True,
        'name': 'negative_axis'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Input 12: Another float64 case
    input_dict_12 = {
        'x': np.arange(1, 7, dtype=np.float64).reshape(2,3),
        'axes': np.array([0, 1], dtype=np.int32),
        'frequency_weights': np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64).reshape(2,3),
        'keepdims': True,
        'name': 'float64_case2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs["tf.nn.weighted_moments"] = tf_nn_weighted_moments_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.weighted_moments' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.weighted_moments'.")

check_valid('tf.nn.weighted_moments', generated_inputs['tf.nn.weighted_moments'], lib="tf", suffix=0)
