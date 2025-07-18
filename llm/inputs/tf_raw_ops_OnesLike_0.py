
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import ml_dtypes

def tf_raw_ops_oneslike_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.OnesLike.
    """
    list_of_inputs = []

    # Input 1: Simple 1D float32 tensor
    list_of_inputs.append({
        'x': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'name': 'float32_1d'
    })

    # Input 2: 2D int32 tensor with negative values
    list_of_inputs.append({
        'x': np.array([[-1, 2], [-3, 4]], dtype=np.int32),
        'name': 'int32_2d_neg'
    })

    # Input 3: 3D bool tensor
    list_of_inputs.append({
        'x': np.array([[[True, False], [False, True]]], dtype=np.bool_),
        'name': 'bool_3d'
    })

    # Input 4: 0D (scalar) float64 tensor
    list_of_inputs.append({
        'x': np.array(100.0, dtype=np.float64),
        'name': 'float64_scalar'
    })

    # Input 5: Tensor with a zero dimension
    list_of_inputs.append({
        'x': np.zeros((3, 0, 2), dtype=np.int8),
        'name': 'int8_zero_dim'
    })

    # Input 6: 1D uint32 tensor (replaces problematic uint64)
    list_of_inputs.append({
        'x': np.array([0, 10, 2**32 - 1], dtype=np.uint32),
        'name': 'uint32_1d'
    })

    # Input 7: 2D complex64 tensor
    list_of_inputs.append({
        'x': np.array([[1+2j, 3-4j], [5, 6j]], dtype=np.complex64),
        'name': 'complex64_2d'
    })

    # Input 8: 1D half (float16) tensor
    list_of_inputs.append({
        'x': np.array([-0.5, 0.5, 1.5, -2.5], dtype=np.float16),
        'name': 'half_float16_1d'
    })

    # Input 9: 2D bfloat16 tensor
    list_of_inputs.append({
        'x': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=ml_dtypes.bfloat16),
        'name': 'bfloat16_2d'
    })

    # Input 10: 1D complex128 tensor
    list_of_inputs.append({
        'x': np.array([1e10 + 1e10j, -1e10 - 1e10j], dtype=np.complex128),
        'name': 'complex128_1d'
    })

    # Input 11: 4D uint8 tensor
    list_of_inputs.append({
        'x': np.arange(16, dtype=np.uint8).reshape((1, 2, 2, 4)),
        'name': 'uint8_4d'
    })

    # Input 12: A large 1D int64 tensor
    list_of_inputs.append({
        'x': np.arange(-500, 500, dtype=np.int64),
        'name': 'int64_large_1d'
    })

    # Input 13: uint16 tensor
    list_of_inputs.append({
        'x': np.array([100, 200, 65535], dtype=np.uint16),
        'name': 'uint16_basic'
    })

    # Input 14: int16 tensor
    list_of_inputs.append({
        'x': np.array([-32768, 0, 32767], dtype=np.int16),
        'name': 'int16_range'
    })

    return [copy.deepcopy(i) for i in list_of_inputs]

generated_inputs["tf.raw_ops.OnesLike"] = tf_raw_ops_oneslike_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.OnesLike' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.OnesLike'.")

check_valid('tf.raw_ops.OnesLike', generated_inputs['tf.raw_ops.OnesLike'], lib="tf", suffix=0)
