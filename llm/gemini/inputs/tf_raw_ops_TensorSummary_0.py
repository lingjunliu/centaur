
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_TensorSummary_inputs():
    list_of_inputs = []

    # Input 1
    list_of_inputs.append({
        'tensor': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'description': "",
        'labels': [],
        'display_name': "",
        'name': "summary_1"
    })

    # Input 2
    list_of_inputs.append({
        'tensor': np.array([[1, 2], [3, 4]], dtype=np.int32),
        'description': "json-encoded-proto",
        'labels': [],
        'display_name': "my_tensor",
        'name': "summary_2"
    })

    # Input 3
    list_of_inputs.append({
        'tensor': np.array(42.0, dtype=np.float64),
        'description': "",
        'labels': [],
        'display_name': "scalar",
        'name': "summary_3"
    })

    # Input 4
    list_of_inputs.append({
        'tensor': np.array([True, False, True], dtype=np.bool_),
        'description': "{}",
        'labels': [],
        'display_name': "boolean_tensor",
        'name': "summary_4"
    })

    # Input 5
    list_of_inputs.append({
        'tensor': np.array([10, 20, 30], dtype=np.int16),
        'description': "",
        'labels': [],
        'display_name': "int16_tensor",
        'name': "summary_5"
    })

    # Input 6
    list_of_inputs.append({
        'tensor': np.random.randn(2, 3, 4).astype(np.float32),
        'description': "desc",
        'labels': [],
        'display_name': "3d_tensor",
        'name': "summary_6"
    })

    # Input 7
    list_of_inputs.append({
        'tensor': np.array([-1, -2, -3], dtype=np.int64),
        'description': "",
        'labels': [],
        'display_name': "negatives",
        'name': "summary_7"
    })

    # Input 8
    list_of_inputs.append({
        'tensor': np.array([1e-5, 1e-6], dtype=np.float16),
        'description': "{\"key\": \"val\"}",
        'labels': [],
        'display_name': "half_precision",
        'name': "summary_8"
    })

    # Input 9
    list_of_inputs.append({
        'tensor': np.zeros((1, 1, 1, 1), dtype=np.int32),
        'description': "",
        'labels': [],
        'display_name': "four_d",
        'name': "summary_9"
    })

    # Input 10
    list_of_inputs.append({
        'tensor': np.array([255, 128, 0], dtype=np.uint8),
        'description': "",
        'labels': [],
        'display_name': "image_bytes",
        'name': "summary_10"
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.TensorSummary"] = tf_raw_ops_TensorSummary_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.TensorSummary' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.TensorSummary'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.TensorSummary', generated_inputs['tf.raw_ops.TensorSummary'], lib="tf", suffix=0)
