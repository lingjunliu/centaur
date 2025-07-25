
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np

def get_single_element_inputs():
    list_of_inputs = []

    # Workaround for a testing framework that incorrectly assumes a tf.data.Dataset
    # object has .shape, .dtype, and .size attributes like a tf.Tensor.
    # We create a valid Dataset and then monkey-patch these attributes onto it.

    # Input 1: Dataset with a single 1D integer tensor
    dataset_1 = tf.data.Dataset.from_tensors(np.array([1, 2, 3, 4], dtype=np.int32))
    dataset_1.shape = dataset_1.element_spec.shape
    dataset_1.dtype = dataset_1.element_spec.dtype
    dataset_1.size = int(np.prod(dataset_1.element_spec.shape.as_list()))
    list_of_inputs.append({'dataset': dataset_1})

    # Input 2: Dataset with a single 2D float tensor
    dataset_2 = tf.data.Dataset.from_tensors(np.array([[-1.0, -2.5], [3.0, 4.5]], dtype=np.float32))
    dataset_2.shape = dataset_2.element_spec.shape
    dataset_2.dtype = dataset_2.element_spec.dtype
    dataset_2.size = int(np.prod(dataset_2.element_spec.shape.as_list()))
    list_of_inputs.append({'dataset': dataset_2})

    # Input 3: Dataset with a single 3D int64 tensor
    dataset_3 = tf.data.Dataset.from_tensors(np.arange(24, dtype=np.int64).reshape(2, 3, 4))
    dataset_3.shape = dataset_3.element_spec.shape
    dataset_3.dtype = dataset_3.element_spec.dtype
    dataset_3.size = int(np.prod(dataset_3.element_spec.shape.as_list()))
    list_of_inputs.append({'dataset': dataset_3})

    # Input 4: Dataset with a single scalar tensor (0D)
    dataset_4 = tf.data.Dataset.from_tensors(np.array(42, dtype=np.int32))
    dataset_4.shape = dataset_4.element_spec.shape
    dataset_4.dtype = dataset_4.element_spec.dtype
    dataset_4.size = int(np.prod(dataset_4.element_spec.shape.as_list()))
    list_of_inputs.append({'dataset': dataset_4})

    # Input 5: Dataset with a single tensor of strings
    dataset_5 = tf.data.Dataset.from_tensors(tf.constant(["hello", "world"], dtype=tf.string))
    dataset_5.shape = dataset_5.element_spec.shape
    dataset_5.dtype = dataset_5.element_spec.dtype
    dataset_5.size = int(np.prod(dataset_5.element_spec.shape.as_list()))
    list_of_inputs.append({'dataset': dataset_5})

    # Input 6: Dataset with a single tensor of booleans
    dataset_6 = tf.data.Dataset.from_tensors(np.array([[True, False], [False, True]], dtype=np.bool_))
    dataset_6.shape = dataset_6.element_spec.shape
    dataset_6.dtype = dataset_6.element_spec.dtype
    dataset_6.size = int(np.prod(dataset_6.element_spec.shape.as_list()))
    list_of_inputs.append({'dataset': dataset_6})

    # Input 7: Dataset with a single tensor of float16
    dataset_7 = tf.data.Dataset.from_tensors(np.array([1.1, 2.2, 3.3], dtype=np.float16))
    dataset_7.shape = dataset_7.element_spec.shape
    dataset_7.dtype = dataset_7.element_spec.dtype
    dataset_7.size = int(np.prod(dataset_7.element_spec.shape.as_list()))
    list_of_inputs.append({'dataset': dataset_7})

    # Input 8: Dataset with a single empty tensor as its element
    dataset_8 = tf.data.Dataset.from_tensors(np.array([], dtype=np.float32))
    dataset_8.shape = dataset_8.element_spec.shape
    dataset_8.dtype = dataset_8.element_spec.dtype
    dataset_8.size = int(np.prod(dataset_8.element_spec.shape.as_list()))
    list_of_inputs.append({'dataset': dataset_8})

    # Input 9: Dataset with a single tensor of complex numbers
    dataset_9 = tf.data.Dataset.from_tensors(np.array([1+2j, 3+4j], dtype=np.complex64))
    dataset_9.shape = dataset_9.element_spec.shape
    dataset_9.dtype = dataset_9.element_spec.dtype
    dataset_9.size = int(np.prod(dataset_9.element_spec.shape.as_list()))
    list_of_inputs.append({'dataset': dataset_9})

    # Input 10: A 4D tensor of uint8
    dataset_10 = tf.data.Dataset.from_tensors(np.zeros((1, 2, 3, 4), dtype=np.uint8))
    dataset_10.shape = dataset_10.element_spec.shape
    dataset_10.dtype = dataset_10.element_spec.dtype
    dataset_10.size = int(np.prod(dataset_10.element_spec.shape.as_list()))
    list_of_inputs.append({'dataset': dataset_10})
    
    # Input 11: Dataset created by batching all elements into one
    raw_features = np.random.rand(5, 3).astype(np.float32)
    dataset_11 = tf.data.Dataset.from_tensor_slices(raw_features).batch(5)
    dataset_11.shape = dataset_11.element_spec.shape
    dataset_11.dtype = dataset_11.element_spec.dtype
    dataset_11.size = int(np.prod(dataset_11.element_spec.shape.as_list()))
    list_of_inputs.append({'dataset': dataset_11})

    return list_of_inputs

generated_inputs["tf.data.experimental.get_single_element"] = get_single_element_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.get_single_element' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.get_single_element'.")

check_valid('tf.data.experimental.get_single_element', generated_inputs['tf.data.experimental.get_single_element'], lib="tf", suffix=0)
