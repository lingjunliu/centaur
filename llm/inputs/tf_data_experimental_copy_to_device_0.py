
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_data_experimental_copy_to_device_inputs():
    """
    Generates a list of valid inputs for tf.data.experimental.copy_to_device.
    The 'inner_values' key provides the data to create a tf.data.Dataset,
    to which the returned transformation function will be applied by the test harness.
    This special key is required by the test harness for APIs that return functions.
    Inputs are constrained to devices likely to exist in a standard test environment (CPU:0, GPU:0).
    """
    list_of_inputs = []

    # Case 1: Simple int32 array, CPU to GPU:0
    list_of_inputs.append({
        'inner_values': (np.arange(10, dtype=np.int32),),
        'target_device': '/gpu:0',
        'source_device': '/cpu:0'
    })

    # Case 2: Float32 array, CPU to GPU:0
    list_of_inputs.append({
        'inner_values': (np.random.rand(5, 2).astype(np.float32),),
        'target_device': '/gpu:0',
        'source_device': '/cpu:0'
    })

    # Case 3: Multi-component dataset with bool and int64, CPU to GPU:0
    list_of_inputs.append({
        'inner_values': (
            np.ones((4, 4), dtype=np.bool_),
            np.arange(4, dtype=np.int64)
        ),
        'target_device': '/gpu:0',
        'source_device': '/cpu:0'
    })

    # Case 4: GPU to CPU transfer
    list_of_inputs.append({
        'inner_values': (np.full((2, 2), -1.0, dtype=np.float32),),
        'target_device': '/cpu:0',
        'source_device': '/gpu:0'
    })

    # Case 5: Empty dataset
    list_of_inputs.append({
        'inner_values': (np.array([], dtype=np.float32),),
        'target_device': '/gpu:0',
        'source_device': '/cpu:0'
    })

    # Case 6: Case-insensitive device type strings
    list_of_inputs.append({
        'inner_values': (np.array([100, 200], dtype=np.uint32),),
        'target_device': '/GPU:0',
        'source_device': '/CPU:0'
    })

    # Case 7: Full device string specification for available devices
    list_of_inputs.append({
        'inner_values': (np.array([1, 8, 27], dtype=np.int16),),
        'target_device': '/job:localhost/replica:0/task:0/device:GPU:0',
        'source_device': '/job:localhost/replica:0/task:0/device:CPU:0'
    })

    # Case 8: Boolean data type
    list_of_inputs.append({
        'inner_values': (np.array([True, False, True, False], dtype=np.bool_),),
        'target_device': '/gpu:0',
        'source_device': '/cpu:0'
    })

    # Case 9: CPU to CPU (identity copy)
    list_of_inputs.append({
        'inner_values': (np.array([[1],[2],[3]], dtype=np.int8),),
        'target_device': '/cpu:0',
        'source_device': '/cpu:0'
    })

    # Case 10: GPU to GPU (identity copy)
    list_of_inputs.append({
        'inner_values': (np.arange(5, dtype=np.uint8),),
        'target_device': '/gpu:0',
        'source_device': '/gpu:0'
    })

    # Case 11: float16 data
    list_of_inputs.append({
        'inner_values': (np.random.rand(3,3).astype(np.float16),),
        'target_device': '/gpu:0',
        'source_device': '/cpu:0'
    })
    
    # Case 12: Default source_device to GPU
    list_of_inputs.append({
        'inner_values': (np.array([1.1, 2.2, 3.3], dtype=np.float32),),
        'target_device': '/gpu:0',
    })

    final_list = [copy.deepcopy(d) for d in list_of_inputs]
    return final_list

generated_inputs["tf.data.experimental.copy_to_device"] = get_tf_data_experimental_copy_to_device_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.copy_to_device' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.copy_to_device'.")

check_valid('tf.data.experimental.copy_to_device', generated_inputs['tf.data.experimental.copy_to_device'], lib="tf", suffix=0)
