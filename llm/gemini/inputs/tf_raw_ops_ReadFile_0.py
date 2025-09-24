
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import tempfile
import os

def tf_raw_ops_read_file_inputs():
  """
  Generates a list of valid inputs for the tf.raw_ops.ReadFile function.
  """
  list_of_inputs = []

  # Create temporary files that are guaranteed to exist during the test run.
  # The test runner is expected to clean up the temporary directory.
  # delete=False is crucial for the file to persist after the file handle is closed.
  temp_f_content = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix=".txt")
  temp_f_content.write("hello world")
  temp_f_content.close()

  temp_f_empty = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix=".empty")
  temp_f_empty.close()

  # Use these real, existing filenames for the inputs.
  # Using dtype=object for the numpy array to ensure compatibility.

  # Input 1: Read a file with content, with an operation name
  input_dict = {
      'filename': np.array(temp_f_content.name, dtype=object),
      'name': 'ReadFileWithContent'
  }
  list_of_inputs.append(copy.deepcopy(input_dict))

  # Input 2: Read an empty file, without an operation name
  input_dict = {
      'filename': np.array(temp_f_empty.name, dtype=object),
      'name': None
  }
  list_of_inputs.append(copy.deepcopy(input_dict))

  # Input 3: Read a file with content, without an operation name
  input_dict = {
      'filename': np.array(temp_f_content.name, dtype=object),
      'name': None
  }
  list_of_inputs.append(copy.deepcopy(input_dict))

  # Input 4: Read an empty file, with an operation name
  input_dict = {
      'filename': np.array(temp_f_empty.name, dtype=object),
      'name': 'ReadEmptyFile'
  }
  list_of_inputs.append(copy.deepcopy(input_dict))

  # Input 5: Long operation name
  input_dict = {
      'filename': np.array(temp_f_content.name, dtype=object),
      'name': 'a_long_and_specific_operation_name_for_reading_a_file_from_disk'
  }
  list_of_inputs.append(copy.deepcopy(input_dict))

  # Input 6: Operation name with characters allowed in TF names
  input_dict = {
      'filename': np.array(temp_f_content.name, dtype=object),
      'name': 'read_file/op_1'
  }
  list_of_inputs.append(copy.deepcopy(input_dict))

  # Input 7: Another valid operation name
  input_dict = {
      'filename': np.array(temp_f_empty.name, dtype=object),
      'name': 'my_file_reader'
  }
  list_of_inputs.append(copy.deepcopy(input_dict))

  # Input 8: Another valid operation name
  input_dict = {
      'filename': np.array(temp_f_content.name, dtype=object),
      'name': 'test.reader.op'
  }
  list_of_inputs.append(copy.deepcopy(input_dict))

  # Input 9: Using bytes literal for the filename path
  input_dict = {
      'filename': np.array(temp_f_content.name.encode('utf-8'), dtype=object),
      'name': 'ReadFileWithBytesFilename'
  }
  list_of_inputs.append(copy.deepcopy(input_dict))

  # Input 10: Another case with the empty file and no name
  input_dict = {
      'filename': np.array(temp_f_empty.name, dtype=object),
      'name': None
  }
  list_of_inputs.append(copy.deepcopy(input_dict))

  return list_of_inputs

generated_inputs["tf.raw_ops.ReadFile"] = tf_raw_ops_read_file_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ReadFile' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ReadFile'.")

check_valid('tf.raw_ops.ReadFile', generated_inputs['tf.raw_ops.ReadFile'], lib="tf", suffix=0)
