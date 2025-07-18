
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_prefetch_to_device_inputs():
  """
  Generates a list of valid inputs for tf.data.experimental.prefetch_to_device.
  
  This function is designed to work with a test harness that understands that
  tf.data.experimental.prefetch_to_device returns a transformation function.
  The harness is expected to find a 'self' key in the input dictionary, which
  is a tf.data.Dataset object, and apply the transformation to it.
  i.e., self.apply(tf.data.experimental.prefetch_to_device(**kwargs))
  """
  list_of_inputs = []

  # Input 1: Basic CPU prefetching with a small buffer size.
  input_dict_1 = {
      'self': tf.data.Dataset.from_tensor_slices(np.arange(10, dtype=np.int32)),
      'device': '/cpu:0',
      'buffer_size': 2
  }
  list_of_inputs.append(input_dict_1)

  # Input 2: Basic GPU prefetching with a standard buffer size.
  input_dict_2 = {
      'self': tf.data.Dataset.from_tensor_slices(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)),
      'device': '/gpu:0',
      'buffer_size': 16
  }
  list_of_inputs.append(input_dict_2)

  # Input 3: Prefetch to another GPU device with a larger buffer size.
  input_dict_3 = {
      'self': tf.data.Dataset.from_tensor_slices(np.array([True, False, True, False])),
      'device': '/gpu:1',
      'buffer_size': 64
  }
  list_of_inputs.append(input_dict_3)

  # Input 4: Using AUTOTUNE for buffer_size on CPU.
  input_dict_4 = {
      'self': tf.data.Dataset.from_tensor_slices(np.zeros((5, 2), dtype=np.bool_)),
      'device': '/cpu:0',
      'buffer_size': tf.data.AUTOTUNE
  }
  list_of_inputs.append(input_dict_4)

  # Input 5: Minimal buffer size of 1 on GPU.
  input_dict_5 = {
      'self': tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3, 4, 5], dtype=np.int64)),
      'device': '/gpu:0',
      'buffer_size': 1
  }
  list_of_inputs.append(input_dict_5)

  # Input 6: Very large buffer size on CPU.
  input_dict_6 = {
      'self': tf.data.Dataset.from_tensor_slices(np.random.rand(20).astype(np.float64)),
      'device': '/cpu:0',
      'buffer_size': 1000
  }
  list_of_inputs.append(input_dict_6)

  # Input 7: Fully qualified CPU device name with a numpy.int32 buffer size.
  input_dict_7 = {
      'self': tf.data.Dataset.from_tensor_slices(np.arange(5, dtype=np.uint8)),
      'device': '/job:localhost/replica:0/task:0/device:CPU:0',
      'buffer_size': np.int32(10)
  }
  list_of_inputs.append(input_dict_7)
  
  # Input 8: Fully qualified GPU device name with a numpy.int64 buffer size.
  input_dict_8 = {
      'self': tf.data.Dataset.from_tensor_slices(np.ones(3, dtype=np.int16)),
      'device': '/job:localhost/replica:0/task:0/device:GPU:0',
      'buffer_size': np.int64(32)
  }
  list_of_inputs.append(input_dict_8)

  # Input 9: Device string with a different job/task.
  input_dict_9 = {
      'self': tf.data.Dataset.from_tensor_slices(np.ones((4, 4), dtype=np.float32)),
      'device': '/job:worker/replica:0/task:1/device:CPU:0',
      'buffer_size': 8
  }
  list_of_inputs.append(input_dict_9)

  # Input 10: Using AUTOTUNE for buffer_size on GPU.
  input_dict_10 = {
      'self': tf.data.Dataset.from_tensor_slices(np.linspace(0, 1, 10, dtype=np.float16)),
      'device': '/gpu:0',
      'buffer_size': tf.data.AUTOTUNE
  }
  list_of_inputs.append(input_dict_10)

  return list_of_inputs

generated_inputs["tf.data.experimental.prefetch_to_device"] = tf_data_experimental_prefetch_to_device_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.prefetch_to_device' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.prefetch_to_device'.")

check_valid('tf.data.experimental.prefetch_to_device', generated_inputs['tf.data.experimental.prefetch_to_device'], lib="tf", suffix=0)
