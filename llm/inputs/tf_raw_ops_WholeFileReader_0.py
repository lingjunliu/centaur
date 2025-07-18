
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_WholeFileReader_inputs():
  """
  Generates a list of valid inputs for the tf.raw_ops.WholeFileReader function.
  This op is known to work only in graph mode, so direct eager execution will fail.
  The provided inputs are syntactically valid according to the function signature.
  """
  list_of_inputs = []

  # Input 1: Default values for all parameters.
  input_dict_1 = {
      'container': '',
      'shared_name': '',
      'name': 'WFR_1'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_1))

  # Input 2: A simple container name.
  input_dict_2 = {
      'container': 'c1',
      'shared_name': '',
      'name': 'WFR_2'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_2))

  # Input 3: A simple shared name.
  input_dict_3 = {
      'container': '',
      'shared_name': 's1',
      'name': 'WFR_3'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_3))

  # Input 4: Both container and shared name specified.
  input_dict_4 = {
      'container': 'c2',
      'shared_name': 's2',
      'name': 'WFR_4'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_4))

  # Input 5: Names containing numbers.
  input_dict_5 = {
      'container': 'container_with_number_123',
      'shared_name': 'shared_name_456',
      'name': 'WFR_5'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_5))

  # Input 6: Names containing underscores.
  input_dict_6 = {
      'container': 'my_data_set',
      'shared_name': 'shared_for_all_users',
      'name': 'WFR_6'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_6))

  # Input 7: Mixed-case names.
  input_dict_7 = {
      'container': 'TestContainer',
      'shared_name': 'TestSharedName',
      'name': 'WFR_7_Test'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_7))

  # Input 8: Long string values for all parameters.
  input_dict_8 = {
      'container': 'this_is_a_very_long_string_to_be_used_as_a_container_name',
      'shared_name': 'this_is_an_even_longer_string_to_be_used_as_a_shared_name_for_this_reader',
      'name': 'WFR_8'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_8))
  
  # Input 9: All uppercase names.
  input_dict_9 = {
      'container': 'IMAGES',
      'shared_name': 'IMAGE_READER',
      'name': 'WFR_9'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_9))
  
  # Input 10: Using a name that is different from the default pattern.
  input_dict_10 = {
      'container': 'logs',
      'shared_name': 'log_reader',
      'name': 'MyCustomReaderName'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_10))

  return list_of_inputs

generated_inputs["tf.raw_ops.WholeFileReader"] = tf_raw_ops_WholeFileReader_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.WholeFileReader' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.WholeFileReader'.")

check_valid('tf.raw_ops.WholeFileReader', generated_inputs['tf.raw_ops.WholeFileReader'], lib="tf", suffix=0)
