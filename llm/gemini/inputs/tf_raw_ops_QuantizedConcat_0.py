
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_quantized_concat_inputs():
  """
  Generates a list of valid inputs for tf.raw_ops.QuantizedConcat.
  """
  list_of_inputs = []

  # Case 1: Simple 1D Concat (quint8)
  input_dict_1 = {
      'name': 'simple_1d_quint8',
      'concat_dim': np.array(0, dtype=np.int32),
      'values': [
          np.array([1, 2, 3], dtype=np.uint8),
          np.array([4, 5], dtype=np.uint8)
      ],
      'input_mins': [
          np.array(0.0, dtype=np.float32),
          np.array(0.0, dtype=np.float32)
      ],
      'input_maxes': [
          np.array(255.0, dtype=np.float32),
          np.array(255.0, dtype=np.float32)
      ]
  }
  list_of_inputs.append(copy.deepcopy(input_dict_1))

  # Case 2: 2D Concat along axis 0 (qint8)
  input_dict_2 = {
      'name': '2d_concat_axis0_qint8',
      'concat_dim': np.array(0, dtype=np.int32),
      'values': [
          np.array([[-1, -2, -3], [-4, -5, -6]], dtype=np.int8),
          np.array([[7, 8, 9]], dtype=np.int8)
      ],
      'input_mins': [
          np.array(-128.0, dtype=np.float32),
          np.array(-10.0, dtype=np.float32)
      ],
      'input_maxes': [
          np.array(127.0, dtype=np.float32),
          np.array(10.0, dtype=np.float32)
      ]
  }
  list_of_inputs.append(copy.deepcopy(input_dict_2))

  # Case 3: 2D Concat along axis 1 (qint8)
  input_dict_3 = {
      'name': '2d_concat_axis1_qint8',
      'concat_dim': np.array(1, dtype=np.int32),
      'values': [
          np.array([[1, 2], [4, 5], [7, 8]], dtype=np.int8),
          np.array([[3], [6], [9]], dtype=np.int8)
      ],
      'input_mins': [
          np.array(-50.0, dtype=np.float32),
          np.array(-60.0, dtype=np.float32)
      ],
      'input_maxes': [
          np.array(50.0, dtype=np.float32),
          np.array(60.0, dtype=np.float32)
      ]
  }
  list_of_inputs.append(copy.deepcopy(input_dict_3))

  # Case 4: Three tensors, 3D Concat along axis 0 (quint8)
  input_dict_4 = {
      'name': '3d_concat_axis0_three_tensors_quint8',
      'concat_dim': np.array(0, dtype=np.int32),
      'values': [
          np.random.randint(0, 255, size=(1, 2, 3), dtype=np.uint8),
          np.random.randint(0, 255, size=(2, 2, 3), dtype=np.uint8),
          np.random.randint(0, 255, size=(1, 2, 3), dtype=np.uint8)
      ],
      'input_mins': [
          np.array(0.0, dtype=np.float32),
          np.array(10.0, dtype=np.float32),
          np.array(20.0, dtype=np.float32)
      ],
      'input_maxes': [
          np.array(200.0, dtype=np.float32),
          np.array(210.0, dtype=np.float32),
          np.array(220.0, dtype=np.float32)
      ]
  }
  list_of_inputs.append(copy.deepcopy(input_dict_4))

  # Case 5: 3D Concat along axis 1 (qint32)
  input_dict_5 = {
      'name': '3d_concat_axis1_qint32',
      'concat_dim': np.array(1, dtype=np.int32),
      'values': [
          np.random.randint(-1000, 1000, size=(2, 1, 3), dtype=np.int32),
          np.random.randint(-1000, 1000, size=(2, 3, 3), dtype=np.int32)
      ],
      'input_mins': [
          np.array(-10000.0, dtype=np.float32),
          np.array(-5000.0, dtype=np.float32)
      ],
      'input_maxes': [
          np.array(10000.0, dtype=np.float32),
          np.array(5000.0, dtype=np.float32)
      ]
  }
  list_of_inputs.append(copy.deepcopy(input_dict_5))

  # Case 6: 3D Concat along axis 2 (qint8)
  input_dict_6 = {
      'name': '3d_concat_axis2_qint8',
      'concat_dim': np.array(2, dtype=np.int32),
      'values': [
          np.random.randint(-128, 127, size=(2, 3, 1), dtype=np.int8),
          np.random.randint(-128, 127, size=(2, 3, 4), dtype=np.int8)
      ],
      'input_mins': [
          np.array(-1.0, dtype=np.float32),
          np.array(-2.0, dtype=np.float32)
      ],
      'input_maxes': [
          np.array(1.0, dtype=np.float32),
          np.array(2.0, dtype=np.float32)
      ]
  }
  list_of_inputs.append(copy.deepcopy(input_dict_6))

  # Case 7: Four tensors, 1D Concat (quint8)
  input_dict_7 = {
      'name': 'four_tensors_1d_quint8',
      'concat_dim': np.array(0, dtype=np.int32),
      'values': [
          np.array([1, 2], dtype=np.uint8),
          np.array([3], dtype=np.uint8),
          np.array([4, 5, 6], dtype=np.uint8),
          np.array([7, 8], dtype=np.uint8)
      ],
      'input_mins': [
          np.array(0.0, dtype=np.float32),
          np.array(0.0, dtype=np.float32),
          np.array(0.0, dtype=np.float32),
          np.array(0.0, dtype=np.float32)
      ],
      'input_maxes': [
          np.array(10.0, dtype=np.float32),
          np.array(10.0, dtype=np.float32),
          np.array(10.0, dtype=np.float32),
          np.array(10.0, dtype=np.float32)
      ]
  }
  list_of_inputs.append(copy.deepcopy(input_dict_7))
  
  # Case 8: 4D Concat along axis 3 (qint8)
  input_dict_8 = {
      'name': '4d_concat_axis3_qint8',
      'concat_dim': np.array(3, dtype=np.int32),
      'values': [
          np.random.randint(-128, 127, size=(1, 2, 2, 3), dtype=np.int8),
          np.random.randint(-128, 127, size=(1, 2, 2, 5), dtype=np.int8)
      ],
      'input_mins': [
          np.array(-128.0, dtype=np.float32),
          np.array(-128.0, dtype=np.float32)
      ],
      'input_maxes': [
          np.array(127.0, dtype=np.float32),
          np.array(127.0, dtype=np.float32)
      ]
  }
  list_of_inputs.append(copy.deepcopy(input_dict_8))

  # Case 9: Concat with empty tensor along concat_dim
  input_dict_9 = {
      'name': 'concat_empty_tensor',
      'concat_dim': np.array(0, dtype=np.int32),
      'values': [
          np.array([], dtype=np.int8).reshape(0, 3),
          np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
      ],
      'input_mins': [
          np.array(-1.0, dtype=np.float32),
          np.array(-1.0, dtype=np.float32)
      ],
      'input_maxes': [
          np.array(1.0, dtype=np.float32),
          np.array(1.0, dtype=np.float32)
      ]
  }
  list_of_inputs.append(copy.deepcopy(input_dict_9))

  # Case 10: Widely different min/max ranges
  input_dict_10 = {
      'name': 'different_min_max_ranges',
      'concat_dim': np.array(0, dtype=np.int32),
      'values': [
          np.array([1, 2, 3], dtype=np.int8),
          np.array([-4, -5], dtype=np.int8)
      ],
      'input_mins': [
          np.array(-1.0, dtype=np.float32),
          np.array(-128.0, dtype=np.float32)
      ],
      'input_maxes': [
          np.array(1.0, dtype=np.float32),
          np.array(127.0, dtype=np.float32)
      ]
  }
  list_of_inputs.append(copy.deepcopy(input_dict_10))

  return list_of_inputs

generated_inputs["tf.raw_ops.QuantizedConcat"] = tf_raw_ops_quantized_concat_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QuantizedConcat' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedConcat'.")

check_valid('tf.raw_ops.QuantizedConcat', generated_inputs['tf.raw_ops.QuantizedConcat'], lib="tf", suffix=0)
