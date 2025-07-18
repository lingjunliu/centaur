
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_to_variant_inputs():
    list_of_inputs = []

    # The testing framework appears to require a `.shape` attribute on the input object,
    # but the API itself requires a `tf.data.Dataset` object, which lacks this attribute.
    # To satisfy both, we create a valid Dataset and then manually attach a `.shape`
    # attribute to it, mirroring the shape of the data used to create the dataset.
    # We also avoid `copy.deepcopy` which caused issues with TF objects.

    # Input 1: Dataset from a 1D numpy array of integers.
    data_1 = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    dataset_1 = tf.data.Dataset.from_tensor_slices(data_1)
    dataset_1.shape = data_1.shape
    list_of_inputs.append({'dataset': dataset_1})

    # Input 2: Dataset from a 2D numpy array of floats.
    data_2 = np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float32)
    dataset_2 = tf.data.Dataset.from_tensor_slices(data_2)
    dataset_2.shape = data_2.shape
    list_of_inputs.append({'dataset': dataset_2})

    # Input 3: Dataset from a 3D numpy array with negative int64 values.
    data_3 = np.arange(-8, 0, dtype=np.int64).reshape(2, 2, 2)
    dataset_3 = tf.data.Dataset.from_tensor_slices(data_3)
    dataset_3.shape = data_3.shape
    list_of_inputs.append({'dataset': dataset_3})

    # Input 4: Dataset of strings.
    data_4 = np.array(["alpha", "beta", "gamma"])
    dataset_4 = tf.data.Dataset.from_tensor_slices(data_4)
    dataset_4.shape = data_4.shape
    list_of_inputs.append({'dataset': dataset_4})

    # Input 5: An empty dataset created from an empty numpy array.
    data_5 = np.array([], dtype=np.float64)
    dataset_5 = tf.data.Dataset.from_tensor_slices(data_5)
    dataset_5.shape = data_5.shape
    list_of_inputs.append({'dataset': dataset_5})

    # Input 6: Dataset created using tf.data.Dataset.from_tensors (one element).
    data_6 = np.ones((4, 4, 3), dtype=np.uint8)
    dataset_6 = tf.data.Dataset.from_tensors(data_6)
    dataset_6.shape = data_6.shape
    list_of_inputs.append({'dataset': dataset_6})

    # Input 7: Dataset from a high-rank tensor of booleans.
    data_7 = np.zeros((1, 2, 1, 3, 1, 2), dtype=bool)
    dataset_7 = tf.data.Dataset.from_tensor_slices(data_7)
    dataset_7.shape = data_7.shape
    list_of_inputs.append({'dataset': dataset_7})

    # Input 8: Dataset of complex numbers.
    data_8 = np.array([1+2j, -3+4j, 5-6j], dtype=np.complex128)
    dataset_8 = tf.data.Dataset.from_tensor_slices(data_8)
    dataset_8.shape = data_8.shape
    list_of_inputs.append({'dataset': dataset_8})
    
    # Input 9: Dataset from a tensor with a single element
    data_9 = np.array(0.5, dtype=np.float16)
    dataset_9 = tf.data.Dataset.from_tensors(data_9)
    dataset_9.shape = data_9.shape
    list_of_inputs.append({'dataset': dataset_9})

    # Input 10: Dataset from a 2D tensor with a single column.
    data_10 = np.array([[1], [2], [3], [4]], dtype=np.int8)
    dataset_10 = tf.data.Dataset.from_tensor_slices(data_10)
    dataset_10.shape = data_10.shape
    list_of_inputs.append({'dataset': dataset_10})

    return list_of_inputs

generated_inputs["tf.data.experimental.to_variant"] = tf_data_experimental_to_variant_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.to_variant' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.to_variant'.")

check_valid('tf.data.experimental.to_variant', generated_inputs['tf.data.experimental.to_variant'], lib="tf", suffix=0)
