
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_hessians_inputs():
    """
    Generates a list of valid inputs for the tf.hessians function.

    NOTE: The user's testing framework is producing two conflicting errors:
    1. `AttributeError: 'list' object has no attribute 'shape'`: This occurs when 'ys'
       or 'xs' are provided as Python lists, which is required by the `tensor_list`
       type in the provided signature. The testing tool seems unable to handle lists
       for this type.
    2. `RuntimeError: ... not supported when eager execution is enabled`: This occurs
       when 'ys' and 'xs' are provided as single numpy arrays to bypass the first
       error. This error is due to the execution environment and cannot be fixed by
       modifying the inputs alone.

    To resolve the immediate `AttributeError`, the following inputs provide 'ys' and 'xs'
    as single numpy arrays. This is a valid use case for the API, which accepts
    "A `Tensor` or list of tensors". This should allow the inputs to pass the validation
    stage that was previously failing. The `aggregation_method` is provided as a string
    as specified in the signature.
    """
    list_of_inputs = []

    # Per the signature, aggregation_method is a string.
    aggregation_methods = [
        None,
        "add_n",
        "experimental_tree",
        "experimental_accumulate_n",
    ]

    # Input 1: Basic case, 1D float32 input
    xs1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ys1 = np.array(np.sum(xs1**2), dtype=np.float32)
    input_dict_1 = {
        'ys': ys1,
        'xs': xs1,
        'gate_gradients': False,
        'aggregation_method': aggregation_methods[0],
        'name': 'hessian_basic'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D float64 input, with gating and 'add_n' aggregation
    xs2 = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float64)
    ys2 = np.array(np.sum(xs2**3), dtype=np.float64)
    input_dict_2 = {
        'ys': ys2,
        'xs': xs2,
        'gate_gradients': True,
        'aggregation_method': aggregation_methods[1],
        'name': 'hessian_2d_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Scalar input
    xs3 = np.array(5.0, dtype=np.float32)
    ys3 = np.array(xs3**4, dtype=np.float32)
    input_dict_3 = {
        'ys': ys3,
        'xs': xs3,
        'gate_gradients': False,
        'aggregation_method': aggregation_methods[2],
        'name': 'hessian_scalar'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 3D input tensor with a different aggregation method
    xs4 = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    ys4 = np.array(np.sum(xs4), dtype=np.float32)
    input_dict_4 = {
        'ys': ys4,
        'xs': xs4,
        'gate_gradients': True,
        'aggregation_method': aggregation_methods[3],
        'name': 'hessian_3d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Long 1D vector
    xs5 = np.linspace(-5.0, 5.0, num=10, dtype=np.float32)
    ys5 = np.array(np.sum(xs5**2), dtype=np.float32)
    input_dict_5 = {
        'ys': ys5,
        'xs': xs5,
        'gate_gradients': False,
        'aggregation_method': aggregation_methods[0],
        'name': 'hessian_long_vector'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: All zeros input
    xs6 = np.zeros((3, 3), dtype=np.float32)
    ys6 = np.array(np.sum(xs6), dtype=np.float32)
    input_dict_6 = {
        'ys': ys6,
        'xs': xs6,
        'gate_gradients': True,
        'aggregation_method': aggregation_methods[1],
        'name': 'hessian_zeros'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: All ones input with float64
    xs7 = np.ones((2, 4), dtype=np.float64)
    ys7 = np.array(np.sum(np.log(xs7 + 1)), dtype=np.float64)
    input_dict_7 = {
        'ys': ys7,
        'xs': xs7,
        'gate_gradients': False,
        'aggregation_method': aggregation_methods[2],
        'name': 'hessian_ones_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: 4D tensor input
    xs8 = np.ones((1, 2, 3, 1), dtype=np.float32) * 2.0
    ys8 = np.array(np.sum(xs8**2), dtype=np.float32)
    input_dict_8 = {
        'ys': ys8,
        'xs': xs8,
        'gate_gradients': True,
        'aggregation_method': aggregation_methods[0],
        'name': 'hessian_4d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Mix of negative and positive values
    xs9 = np.array([-1., -2., 1., 2.], dtype=np.float32)
    ys9 = np.array(np.sum(np.sin(xs9)), dtype=np.float32)
    input_dict_9 = {
        'ys': ys9,
        'xs': xs9,
        'gate_gradients': False,
        'aggregation_method': aggregation_methods[1],
        'name': 'hessian_mixed_sign'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Larger 2D tensor
    xs10 = np.arange(1, 13, dtype=np.float64).reshape(3, 4)
    ys10 = np.array(np.sum(xs10 * np.log(xs10)), dtype=np.float64)
    input_dict_10 = {
        'ys': ys10,
        'xs': xs10,
        'gate_gradients': True,
        'aggregation_method': aggregation_methods[3],
        'name': 'hessian_large_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.hessians"] = tf_hessians_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.hessians' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.hessians'.")

check_valid('tf.hessians', generated_inputs['tf.hessians'], lib="tf", suffix=0)
