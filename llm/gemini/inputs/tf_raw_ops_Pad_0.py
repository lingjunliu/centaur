
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Pad_inputs():
    list_of_inputs = []
    
    # Input 1: 1D input, int32, padding 1D with int32 paddings
    list_of_inputs.append({
        "name": "pad_1d_int32",
        "input": np.array([1, 2, 3], dtype=np.int32),
        "paddings": np.array([[1, 2]], dtype=np.int32)
    })

    # Input 2: 2D input, float32, padding 2D with int32 paddings
    list_of_inputs.append({
        "name": "pad_2d_float32",
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "paddings": np.array([[1, 1], [2, 2]], dtype=np.int32)
    })

    # Input 3: 3D input, float64, padding 3D with int64 paddings
    list_of_inputs.append({
        "name": "pad_3d_float64",
        "input": np.random.randn(2, 3, 4).astype(np.float64),
        "paddings": np.array([[0, 1], [1, 1], [2, 0]], dtype=np.int64)
    })

    # Input 4: 4D input, int64, padding 4D with int32 paddings
    list_of_inputs.append({
        "name": "pad_4d_int64",
        "input": np.random.randint(0, 10, size=(1, 2, 2, 1)).astype(np.int64),
        "paddings": np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.int32)
    })

    # Input 5: 1D input with zero padding
    list_of_inputs.append({
        "name": "pad_zero_padding",
        "input": np.array([5, 6, 7], dtype=np.float32),
        "paddings": np.array([[0, 0]], dtype=np.int32)
    })

    # Input 6: 2D input with large padding, int32 paddings
    list_of_inputs.append({
        "name": "pad_large_padding",
        "input": np.array([[1]], dtype=np.int32),
        "paddings": np.array([[5, 5], [5, 5]], dtype=np.int32)
    })

    # Input 7: 5D input, float32, int32 paddings
    list_of_inputs.append({
        "name": "pad_5d_float32",
        "input": np.ones((1, 1, 1, 1, 1), dtype=np.float32),
        "paddings": np.array([[1, 1], [0, 0], [2, 2], [0, 0], [1, 1]], dtype=np.int32)
    })

    # Input 8: 2D input, bool type, int32 paddings
    list_of_inputs.append({
        "name": "pad_bool",
        "input": np.array([[True, False], [False, True]], dtype=np.bool_),
        "paddings": np.array([[1, 0], [0, 1]], dtype=np.int32)
    })

    # Input 9: 3D input, complex64 type, int32 paddings
    list_of_inputs.append({
        "name": "pad_complex64",
        "input": (np.random.randn(2, 2, 2) + 1j * np.random.randn(2, 2, 2)).astype(np.complex64),
        "paddings": np.array([[0, 0], [1, 1], [0, 0]], dtype=np.int32)
    })

    # Input 10: 1D input, float16 type, int32 paddings
    list_of_inputs.append({
        "name": "pad_float16",
        "input": np.array([1.5, 2.5], dtype=np.float16),
        "paddings": np.array([[3, 3]], dtype=np.int32)
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.Pad"] = tf_raw_ops_Pad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Pad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Pad'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Pad', generated_inputs['tf.raw_ops.Pad'], lib="tf", suffix=0)
