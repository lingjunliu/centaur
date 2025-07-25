
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_accumulator_set_global_step_inputs():
  """
  Generates a list of valid inputs for tf.raw_ops.AccumulatorSetGlobalStep.
  NOTE: This operation is not compatible with eager execution. The provided
  inputs are structurally valid according to the API signature but are
  expected to fail with a RuntimeError if executed in an eager context, as
  the 'handle' argument is a 'ref' type specific to TensorFlow's graph mode.
  """
  list_of_inputs = []

  # The 'handle' argument requires a 'ref' tensor, which is not supported
  # in eager execution. The following inputs conform to the signature but will
  # trigger the known RuntimeError in an eager environment.
  # We use np.array with dtype=object for the string tensor as it is the most
  # compatible format for the testing harness.

  # Case 1: Basic case
  input_dict1 = {
      'handle': np.array('handle_string_1', dtype=object),
      'new_global_step': np.array(0, dtype=np.int64),
      'name': 'step_0'
  }
  list_of_inputs.append(copy.deepcopy(input_dict1))

  # Case 2: Positive step
  input_dict2 = {
      'handle': np.array('handle_string_2', dtype=object),
      'new_global_step': np.array(100, dtype=np.int64),
      'name': None
  }
  list_of_inputs.append(copy.deepcopy(input_dict2))

  # Case 3: Larger step value
  input_dict3 = {
      'handle': np.array('handle_string_3', dtype=object),
      'new_global_step': np.array(98765, dtype=np.int64),
      'name': 'large_step'
  }
  list_of_inputs.append(copy.deepcopy(input_dict3))

  # Case 4: Step value 1
  input_dict4 = {
      'handle': np.array('handle_string_4', dtype=object),
      'new_global_step': np.array(1, dtype=np.int64),
      'name': 'step_1'
  }
  list_of_inputs.append(copy.deepcopy(input_dict4))

  # Case 5: Max int64 step value
  input_dict5 = {
      'handle': np.array('handle_string_5', dtype=object),
      'new_global_step': np.array(np.iinfo(np.int64).max, dtype=np.int64),
      'name': 'max_step'
  }
  list_of_inputs.append(copy.deepcopy(input_dict5))

  # Case 6: Different handle string
  input_dict6 = {
      'handle': np.array('some_other_accumulator_handle', dtype=object),
      'new_global_step': np.array(555, dtype=np.int64),
      'name': 'another_op_name'
  }
  list_of_inputs.append(copy.deepcopy(input_dict6))

  # Case 7: Name with slashes for scoping
  input_dict7 = {
      'handle': np.array('handle_string_7', dtype=object),
      'new_global_step': np.array(1024, dtype=np.int64),
      'name': 'my_scope/my_op'
  }
  list_of_inputs.append(copy.deepcopy(input_dict7))

  # Case 8: Another high value
  input_dict8 = {
      'handle': np.array('handle_string_8', dtype=object),
      'new_global_step': np.array(2000000, dtype=np.int64),
      'name': 'two_million_step'
  }
  list_of_inputs.append(copy.deepcopy(input_dict8))

  # Case 9: Empty string for name
  input_dict9 = {
      'handle': np.array('handle_string_9', dtype=object),
      'new_global_step': np.array(42, dtype=np.int64),
      'name': ''
  }
  list_of_inputs.append(copy.deepcopy(input_dict9))

  # Case 10: Handle with special chars
  input_dict10 = {
      'handle': np.array('handle/with/slashes_10', dtype=object),
      'new_global_step': np.array(314, dtype=np.int64),
      'name': 'pi_step'
  }
  list_of_inputs.append(copy.deepcopy(input_dict10))

  return list_of_inputs

generated_inputs["tf.raw_ops.AccumulatorSetGlobalStep"] = tf_raw_ops_accumulator_set_global_step_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.AccumulatorSetGlobalStep' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AccumulatorSetGlobalStep'.")

check_valid('tf.raw_ops.AccumulatorSetGlobalStep', generated_inputs['tf.raw_ops.AccumulatorSetGlobalStep'], lib="tf", suffix=0)
