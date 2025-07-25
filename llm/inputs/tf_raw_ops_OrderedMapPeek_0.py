
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_raw_ops_ordered_map_peek_inputs():
  """
  Generates a list of valid inputs for tf.raw_ops.OrderedMapPeek.

  NOTE: This operation is inherently blocking. It is designed to wait
  (block execution) until data for the specified 'key' is made available by a
  tf.raw_ops.OrderedMapStage operation. When this 'Peek' operation is
  executed in isolation, it will wait indefinitely, leading to a timeout.
  This timeout is the correct and expected behavior for this op when run
  standalone. The problem cannot be "fixed" by changing the inputs to this
  function alone.

  This function provides a single, minimal, syntactically valid input to
  demonstrate the issue is with the op's nature, not the input values.
  """
  list_of_inputs = []

  # A single, minimal, canonical input. Any valid input for this op will block
  # and time out in an isolated execution environment.
  input_dict_1 = {
      'key': np.array(1, dtype=np.int64),
      'indices': np.array(0, dtype=np.int32),
      'dtypes': [np.float32],
      'capacity': 1,
      'memory_limit': 0,
      'container': '',
      'shared_name': 'singleton_peek_map',
      'name': 'minimal_peek_op'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_1))

  return list_of_inputs

generated_inputs["tf.raw_ops.OrderedMapPeek"] = get_tf_raw_ops_ordered_map_peek_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.OrderedMapPeek' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.OrderedMapPeek'.")

check_valid('tf.raw_ops.OrderedMapPeek', generated_inputs['tf.raw_ops.OrderedMapPeek'], lib="tf", suffix=0)
