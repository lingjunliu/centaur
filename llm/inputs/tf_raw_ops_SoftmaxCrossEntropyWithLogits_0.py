
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SoftmaxCrossEntropyWithLogits_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 input
    features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    labels = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    name = "softmax_1"
    input_dict = {"features": tf.convert_to_tensor(features), "labels": tf.convert_to_tensor(labels), "name": str(name)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different values, float64
    features = np.array([[-1.0, 0.5, 2.0], [3.5, -2.0, 1.0]], dtype=np.float64)
    labels = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    name = "softmax_2"
    input_dict = {"features": tf.convert_to_tensor(features), "labels": tf.convert_to_tensor(labels), "name": str(name)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3:  half type
    features = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float16)
    labels = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16)
    name = "softmax_3"
    input_dict = {"features": tf.convert_to_tensor(features), "labels": tf.convert_to_tensor(labels), "name": str(name)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: bfloat16 type
    features = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=tf.dtypes.bfloat16.as_numpy_dtype)
    labels = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=tf.dtypes.bfloat16.as_numpy_dtype)
    name = "softmax_4"
    input_dict = {"features": tf.convert_to_tensor(features), "labels": tf.convert_to_tensor(labels), "name": str(name)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger batch size
    features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    labels = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    name = "softmax_5"
    input_dict = {"features": tf.convert_to_tensor(features), "labels": tf.convert_to_tensor(labels), "name": str(name)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6:  More classes
    features = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    labels = np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    name = "softmax_6"
    input_dict = {"features": tf.convert_to_tensor(features), "labels": tf.convert_to_tensor(labels), "name": str(name)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: All zero labels (should be a valid probability distribution).
    features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    labels = np.array([[0.5, 0.5], [0.2, 0.8]], dtype=np.float32)
    name = "softmax_7"
    input_dict = {"features": tf.convert_to_tensor(features), "labels": tf.convert_to_tensor(labels), "name": str(name)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative features and labels
    features = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    labels = np.array([[0.7, 0.3], [0.1, 0.9]], dtype=np.float32)
    name = "softmax_8"
    input_dict = {"features": tf.convert_to_tensor(features), "labels": tf.convert_to_tensor(labels), "name": str(name)}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9: Only one class
    features = np.array([[1.0], [2.0]], dtype=np.float32)
    labels = np.array([[1.0], [1.0]], dtype=np.float32)
    name = "softmax_9"
    input_dict = {"features": tf.convert_to_tensor(features), "labels": tf.convert_to_tensor(labels), "name": str(name)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different values, float64
    features = np.array([[1.0, -0.5, -2.0], [-3.5, 2.0, -1.0]], dtype=np.float64)
    labels = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    name = "softmax_10"
    input_dict = {"features": tf.convert_to_tensor(features), "labels": tf.convert_to_tensor(labels), "name": str(name)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SoftmaxCrossEntropyWithLogits"] = tf_raw_ops_SoftmaxCrossEntropyWithLogits_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SoftmaxCrossEntropyWithLogits' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SoftmaxCrossEntropyWithLogits'.")

check_valid('tf.raw_ops.SoftmaxCrossEntropyWithLogits', generated_inputs['tf.raw_ops.SoftmaxCrossEntropyWithLogits'], lib="tf", suffix=0)
