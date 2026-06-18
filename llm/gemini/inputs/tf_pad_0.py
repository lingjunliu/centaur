
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_pad_inputs():
    list_of_inputs = []

    # Input 1: 1D CONSTANT pad with float32
    list_of_inputs.append({
        'tensor': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'paddings': np.array([[1, 2]], dtype=np.int32),
        'mode': "CONSTANT",
        'constant_values': np.array(0.0, dtype=np.float32),
        'name': "pad_1"
    })

    # Input 2: 2D CONSTANT pad with negative fill, int32
    list_of_inputs.append({
        'tensor': np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
        'paddings': np.array([[1, 1], [2, 2]], dtype=np.int32),
        'mode': "CONSTANT",
        'constant_values': np.array(-1, dtype=np.int32),
        'name': "pad_2"
    })

    # Input 3: 2D REFLECT pad
    list_of_inputs.append({
        'tensor': np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        'paddings': np.array([[1, 1], [2, 2]], dtype=np.int32),
        'mode': "REFLECT",
        'constant_values': np.array(0.0, dtype=np.float32),
        'name': "pad_3"
    })

    # Input 4: 2D SYMMETRIC pad, float64
    list_of_inputs.append({
        'tensor': np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64),
        'paddings': np.array([[2, 2], [3, 3]], dtype=np.int32),
        'mode': "SYMMETRIC",
        'constant_values': np.array(0.0, dtype=np.float64),
        'name': "pad_4"
    })

    # Input 5: 3D CONSTANT pad
    list_of_inputs.append({
        'tensor': np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        'paddings': np.array([[1, 0], [0, 2], [1, 1]], dtype=np.int32),
        'mode': "CONSTANT",
        'constant_values': np.array(0.5, dtype=np.float32),
        'name': "pad_5"
    })

    # Input 6: 4D CONSTANT pad, int64
    list_of_inputs.append({
        'tensor': np.ones((2, 2, 2, 2), dtype=np.int64),
        'paddings': np.array([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=np.int32),
        'mode': "CONSTANT",
        'constant_values': np.array(9, dtype=np.int64),
        'name': "pad_6"
    })

    # Input 7: 1D REFLECT pad
    list_of_inputs.append({
        'tensor': np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        'paddings': np.array([[3, 3]], dtype=np.int32),
        'mode': "REFLECT",
        'constant_values': np.array(0.0, dtype=np.float32),
        'name': "pad_7"
    })

    # Input 8: 3D SYMMETRIC pad
    list_of_inputs.append({
        'tensor': np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        'paddings': np.array([[1, 1], [2, 2], [3, 3]], dtype=np.int32),
        'mode': "SYMMETRIC",
        'constant_values': np.array(0.0, dtype=np.float32),
        'name': "pad_8"
    })

    # Input 9: 2D Zero pad
    list_of_inputs.append({
        'tensor': np.array([[1.5]], dtype=np.float32),
        'paddings': np.array([[0, 0], [0, 0]], dtype=np.int32),
        'mode': "CONSTANT",
        'constant_values': np.array(0.0, dtype=np.float32),
        'name': "pad_9"
    })

    # Input 10: Boolean tensor pad
    list_of_inputs.append({
        'tensor': np.array([True, False], dtype=bool),
        'paddings': np.array([[1, 1]], dtype=np.int32),
        'mode': "CONSTANT",
        'constant_values': np.array(True, dtype=bool),
        'name': "pad_10"
    })

    return list_of_inputs

generated_inputs["tf.pad"] = tf_pad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.pad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.pad'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.pad', generated_inputs['tf.pad'], lib="tf", suffix=0)
