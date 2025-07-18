
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_data_experimental_copy_to_device_inputs():
  """
  Generates a list of valid inputs for the tf.data.experimental.copy_to_device function.
  The test harness for dataset transformations expects the input dictionary to contain:
  1. The arguments for the transformation function itself (e.g., 'target_device').
  2. A special key, 'tensors', holding the numpy data to create the initial tf.data.Dataset.
  """
  list_of_inputs = []

  dataset1_np = np.arange(12, dtype=np.int32).reshape((4, 3))
  dataset2_np = np.random.rand(5, 2).astype(np.float32)
  dataset3_np = {
      'features': np.random.rand(4, 2).astype(np.float64),
      'labels': np.random.randint(0, 5, size=4, dtype=np.int64)
  }
  dataset4_np = (
      np.array([1, 2, 3], dtype=np.int8),
      np.array([True, False, True], dtype=np.bool_)
  )
  dataset5_np = np.array(['alpha', 'beta', 'gamma', 'delta'], dtype=object)

  # Input 1: Standard CPU to GPU copy
  input_1 = {
      'tensors': dataset1_np,
      'target_device': '/gpu:0',
      'source_device': '/cpu:0'
  }
  list_of_inputs.append(copy.deepcopy(input_1))

  # Input 2: Copying a different data type (float32)
  input_2 = {
      'tensors': dataset2_np,
      'target_device': '/gpu:1',
      'source_device': '/cpu:0'
  }
  list_of_inputs.append(copy.deepcopy(input_2))

  # Input 3: Copying a structured dataset (dict) from GPU back to CPU
  input_3 = {
      'tensors': dataset3_np,
      'target_device': '/cpu:0',
      'source_device': '/gpu:0'
  }
  list_of_inputs.append(copy.deepcopy(input_3))

  # Input 4: Copying a tuple-based dataset between GPUs
  input_4 = {
      'tensors': dataset4_np,
      'target_device': '/gpu:1',
      'source_device': '/gpu:0'
  }
  list_of_inputs.append(copy.deepcopy(input_4))

  # Input 5: Copying a string dataset to TPU
  input_5 = {
      'tensors': dataset5_np,
      'target_device': '/tpu:0',
      'source_device': '/cpu:0'
  }
  list_of_inputs.append(copy.deepcopy(input_5))

  # Input 6: Copying from TPU back to CPU
  input_6 = {
      'tensors': dataset2_np,
      'target_device': '/cpu:0',
      'source_device': '/tpu:1'
  }
  list_of_inputs.append(copy.deepcopy(input_6))

  # Input 7: Using fully-specified device names for distributed training
  input_7 = {
      'tensors': dataset3_np,
      'target_device': '/job:worker/replica:0/task:0/device:GPU:0',
      'source_device': '/job:ps/replica:0/task:0/device:CPU:0'
  }
  list_of_inputs.append(copy.deepcopy(input_7))

  # Input 8: Another complex device name format
  input_8 = {
      'tensors': dataset4_np,
      'target_device': '/job:localhost/replica:0/task:0/device:GPU:1',
      'source_device': '/job:localhost/replica:0/task:0/device:CPU:0'
  }
  list_of_inputs.append(copy.deepcopy(input_8))

  # Input 9: Copy from GPU to TPU
  input_9 = {
      'tensors': dataset1_np,
      'target_device': '/tpu:0',
      'source_device': '/gpu:0'
  }
  list_of_inputs.append(copy.deepcopy(input_9))

  # Input 10: Copy between different CPU cores
  input_10 = {
      'tensors': dataset2_np,
      'target_device': '/cpu:1',
      'source_device': '/cpu:0'
  }
  list_of_inputs.append(copy.deepcopy(input_10))

  return list_of_inputs

generated_inputs["tf.data.experimental.copy_to_device"] = tf_data_experimental_copy_to_device_inputs()

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
