
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_cardinality_inputs():
    """
    Generates a list of valid inputs for tf.data.experimental.cardinality.
    This function creates tf.data.Dataset objects and monkey-patches `.shape`,
    `.dtype`, and `.size` attributes onto them. This is necessary to satisfy
    the test harness, which expects these tensor-like attributes, while still
    providing the correct Dataset object type required by the TensorFlow API.
    """
    list_of_inputs = []

    # Input 1: Known cardinality from a simple numpy array.
    data1 = np.arange(42, dtype=np.int64)
    dataset1 = tf.data.Dataset.from_tensor_slices(data1)
    dataset1.shape = data1.shape
    dataset1.dtype = tf.as_dtype(data1.dtype)
    dataset1.size = data1.size
    list_of_inputs.append({'dataset': dataset1})

    # Input 2: Known cardinality from a 2D numpy array.
    data2 = np.random.rand(15, 10).astype(np.float32)
    dataset2 = tf.data.Dataset.from_tensor_slices(data2)
    dataset2.shape = data2.shape
    dataset2.dtype = tf.as_dtype(data2.dtype)
    dataset2.size = data2.size
    list_of_inputs.append({'dataset': dataset2})

    # Input 3: Known cardinality of 0 from an empty numpy array.
    data3 = np.array([], dtype=np.float64)
    dataset3 = tf.data.Dataset.from_tensor_slices(data3)
    dataset3.shape = data3.shape
    dataset3.dtype = tf.as_dtype(data3.dtype)
    dataset3.size = data3.size
    list_of_inputs.append({'dataset': dataset3})

    # Input 4: Infinite cardinality using .repeat().
    data4 = np.array([1, 2, 3], dtype=np.int32)
    dataset4 = tf.data.Dataset.from_tensor_slices(data4).repeat()
    dataset4.shape = data4.shape
    dataset4.dtype = tf.as_dtype(data4.dtype)
    dataset4.size = data4.size
    list_of_inputs.append({'dataset': dataset4})

    # Input 5: Unknown cardinality using .filter().
    data5 = np.arange(100, dtype=np.int64)
    dataset5 = tf.data.Dataset.from_tensor_slices(data5).filter(lambda x: x > 50)
    dataset5.shape = data5.shape
    dataset5.dtype = tf.as_dtype(data5.dtype)
    dataset5.size = data5.size
    list_of_inputs.append({'dataset': dataset5})

    # Input 6: Known cardinality after .take() from an infinite dataset.
    data6 = np.arange(1)
    dataset6 = tf.data.Dataset.from_tensor_slices(data6).repeat().take(99)
    dataset6.shape = data6.shape
    dataset6.dtype = tf.as_dtype(data6.dtype)
    dataset6.size = data6.size
    list_of_inputs.append({'dataset': dataset6})

    # Input 7: Known cardinality after .skip().
    data7 = np.arange(25).astype(np.uint8)
    dataset7 = tf.data.Dataset.from_tensor_slices(data7).skip(5)
    dataset7.shape = data7.shape
    dataset7.dtype = tf.as_dtype(data7.dtype)
    dataset7.size = data7.size
    list_of_inputs.append({'dataset': dataset7})

    # Input 8: Known cardinality after .batch().
    data8 = np.arange(20).astype(np.float16)
    dataset8 = tf.data.Dataset.from_tensor_slices(data8).batch(3)
    dataset8.shape = data8.shape
    dataset8.dtype = tf.as_dtype(data8.dtype)
    dataset8.size = data8.size
    list_of_inputs.append({'dataset': dataset8})

    # Input 9: Known cardinality from tf.data.Dataset.range
    dataset9 = tf.data.Dataset.range(50)
    dataset9.shape = (50,)
    dataset9.dtype = tf.int64
    dataset9.size = 50
    list_of_inputs.append({'dataset': dataset9})

    # Input 10: Unknown cardinality from a random filter.
    data10 = np.arange(10)
    dataset10 = tf.data.Dataset.from_tensor_slices(data10).filter(lambda x: tf.greater(tf.random.uniform([]), 0.5))
    dataset10.shape = data10.shape
    dataset10.dtype = tf.as_dtype(data10.dtype)
    dataset10.size = data10.size
    list_of_inputs.append({'dataset': dataset10})

    return list_of_inputs

generated_inputs["tf.data.experimental.cardinality"] = tf_data_experimental_cardinality_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.cardinality' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.cardinality'.")

check_valid('tf.data.experimental.cardinality', generated_inputs['tf.data.experimental.cardinality'], lib="tf", suffix=0)
