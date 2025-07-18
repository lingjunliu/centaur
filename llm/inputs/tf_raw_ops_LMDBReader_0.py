
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def tf_raw_ops_lmdbreader_inputs():
  """
  Generates a list of valid inputs for tf.raw_ops.LMDBReader.
  """
  list_of_inputs = []

  # Input 1: Default parameters (empty strings for container and shared_name)
  input_dict_1 = {
      'container': '',
      'shared_name': '',
      'name': 'default_reader'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_1))

  # Input 2: A different name for the operation
  input_dict_2 = {
      'container': '',
      'shared_name': '',
      'name': 'another_reader_op'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_2))

  # Input 3: Non-empty container, default shared_name
  input_dict_3 = {
      'container': 'my_container',
      'shared_name': '',
      'name': 'reader_in_container'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_3))

  # Input 4: Non-empty shared_name, default container
  input_dict_4 = {
      'container': '',
      'shared_name': 'my_shared_reader',
      'name': 'shared_reader'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_4))

  # Input 5: Both container and shared_name are non-empty
  input_dict_5 = {
      'container': 'app_container',
      'shared_name': 'global_reader',
      'name': 'global_app_reader'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_5))

  # Input 6: Using strings with numbers
  input_dict_6 = {
      'container': 'container123',
      'shared_name': 'shared_reader_v2',
      'name': 'reader_op_789'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_6))

  # Input 7: Using strings with underscores and hyphens
  input_dict_7 = {
      'container': 'my-app-container',
      'shared_name': 'shared_reader_for_images',
      'name': 'lmdb_reader-alpha'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_7))

  # Input 8: Using long strings
  input_dict_8 = {
      'container': 'a_very_long_and_specific_container_name_for_testing',
      'shared_name': 'a_similarly_long_and_descriptive_shared_name',
      'name': 'LongOperationNameForReader'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_8))

  # Input 9: Using path-like strings
  input_dict_9 = {
      'container': '/my/app/container',
      'shared_name': 'reader/for/text_data',
      'name': 'TextReader'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_9))

  # Input 10: All parameters set to the same string
  input_dict_10 = {
      'container': 'common_name',
      'shared_name': 'common_name',
      'name': 'common_name'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_10))

  return list_of_inputs

generated_inputs["tf.raw_ops.LMDBReader"] = tf_raw_ops_lmdbreader_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.LMDBReader' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.LMDBReader'.")

check_valid('tf.raw_ops.LMDBReader', generated_inputs['tf.raw_ops.LMDBReader'], lib="tf", suffix=0)
