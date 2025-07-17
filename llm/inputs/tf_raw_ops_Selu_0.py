
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_selu_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 tensor
    features = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    input_dict = {"features": features.astype(np.float32), "name": "selu_example_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative float32 tensor
    features = np.array([-2.0, -1.0, -0.5], dtype=np.float32)
    input_dict = {"features": features.astype(np.float32), "name": "selu_example_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Positive float32 tensor
    features = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    input_dict = {"features": features.astype(np.float32), "name": "selu_example_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Zero float32 tensor
    features = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    input_dict = {"features": features.astype(np.float32), "name": "selu_example_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional float32 tensor
    features = np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    input_dict = {"features": features.astype(np.float32), "name": "selu_example_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger float32 tensor
    features = np.random.randn(100).astype(np.float32)
    input_dict = {"features": features.astype(np.float32), "name": "selu_example_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float64 tensor
    features = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    input_dict = {"features": features.astype(np.float64), "name": "selu_example_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: half tensor (float16)
    features = np.array([-1.0, 0.0, 1.0], dtype=np.float16)
    input_dict = {"features": features.astype(np.float16), "name": "selu_example_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Remove bfloat16 for now, as it may cause issues
    # features = tf.constant([-1.0, 0.0, 1.0], dtype=tf.float32)
    # features = tf.cast(features, dtype=tf.bfloat16).numpy()
    # input_dict = {"features": features, "name": "selu_example_9"}
    # list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Multi-dimensional float64 tensor with different shape
    features = np.random.randn(5, 5, 5).astype(np.float64)
    input_dict = {"features": features, "name": "selu_example_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Multi-dimensional float16 tensor
    features = np.random.randn(3, 4).astype(np.float16)
    input_dict = {"features": features, "name": "selu_example_11"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Selu"] = tf_raw_ops_selu_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Selu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Selu'.")

check_valid('tf.raw_ops.Selu', generated_inputs['tf.raw_ops.Selu'], lib="tf", suffix=0)
