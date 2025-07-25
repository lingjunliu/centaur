
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

class ShapeableList(list):
    @property
    def shape(self):
        return (len(self),)
    
    @property
    def dtype(self):
        return np.dtype('object')

def tf_sparse_cross_inputs():
  """
  Generates a list of valid inputs for tf.sparse.cross.
  """
  list_of_inputs = []

  # Input 1: Basic case with two dense 2D string tensors and a custom separator
  input_dict_1 = {
      'inputs': ShapeableList([
          np.array([["a"], ["b"]], dtype=object),
          np.array([["c"], ["d"]], dtype=object)
      ]),
      'name': 'dense_2d_strings',
      'separator': '_Y_'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_1))

  # Input 2: Documentation example - mix of sparse and dense tensors
  input_dict_2 = {
      'inputs': ShapeableList([
          {
              'indices': np.array([[0, 0], [1, 0], [1, 1]], dtype=np.int64),
              'values': np.array(['a', 'b', 'c'], dtype=object),
              'dense_shape': np.array([2, 2], dtype=np.int64)
          },
          {
              'indices': np.array([[0, 0], [1, 0]], dtype=np.int64),
              'values': np.array(['d', 'e'], dtype=object),
              'dense_shape': np.array([2, 1], dtype=np.int64)
          },
          np.array([['f'], ['g']], dtype=object)
      ]),
      'name': 'doc_example_mix',
      'separator': '_X_'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_2))

  # Input 3: All sparse tensors with integer values, which will be stringified
  input_dict_3 = {
      'inputs': ShapeableList([
          {
              'indices': np.array([[0, 0], [1, 1]], dtype=np.int64),
              'values': np.array([10, 20], dtype=np.int32),
              'dense_shape': np.array([2, 2], dtype=np.int64)
          },
          {
              'indices': np.array([[0, 0], [1, 1]], dtype=np.int64),
              'values': np.array([30, 40], dtype=np.int32),
              'dense_shape': np.array([2, 2], dtype=np.int64)
          }
      ]),
      'name': 'all_sparse_int',
      'separator': '---'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_3))

  # Input 4: Single dense tensor input
  input_dict_4 = {
      'inputs': ShapeableList([
          np.array([['a', 'b'], ['c', 'd']], dtype=object)
      ]),
      'name': 'single_dense_tensor',
      'separator': '#'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_4))

  # Input 5: Single sparse tensor input
  input_dict_5 = {
      'inputs': ShapeableList([
          {
              'indices': np.array([[0, 1], [1, 0]], dtype=np.int64),
              'values': np.array(['x', 'y'], dtype=object),
              'dense_shape': np.array([2, 2], dtype=np.int64)
          }
      ]),
      'name': 'single_sparse_tensor',
      'separator': '@'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_5))

  # Input 6: Mixed dtypes (int, float, string)
  input_dict_6 = {
      'inputs': ShapeableList([
          np.array([[1], [2]], dtype=np.int32),
          np.array([[3.5], [4.5]], dtype=np.float32),
          np.array([["x"], ["y"]], dtype=object)
      ]),
      'name': 'mixed_dtypes',
      'separator': '<>'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_6))

  # Input 7: Broadcasting shapes
  input_dict_7 = {
      'inputs': ShapeableList([
          np.array([['a'], ['b']], dtype=object),
          np.array([['x', 'y']], dtype=object)
      ]),
      'name': 'broadcasting_cross',
      'separator': '->'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_7))

  # Input 8: Using default separator and None for name
  input_dict_8 = {
      'inputs': ShapeableList([
          np.array([[-10], [-20]], dtype=np.int64),
          np.array([[1.1], [2.2]], dtype=np.float64)
      ]),
      'name': None,
      'separator': None
  }
  list_of_inputs.append(copy.deepcopy(input_dict_8))

  # Input 9: Crossing with an empty sparse tensor
  input_dict_9 = {
      'inputs': ShapeableList([
          {
              'indices': np.array([[0, 0]], dtype=np.int64),
              'values': np.array(['val'], dtype=object),
              'dense_shape': np.array([2, 2], dtype=np.int64)
          },
          {
              'indices': np.array([], dtype=np.int64).reshape(0, 2),
              'values': np.array([], dtype=object),
              'dense_shape': np.array([2, 2], dtype=np.int64)
          }
      ]),
      'name': 'cross_with_empty',
      'separator': 'EMPTY'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_9))

  # Input 10: Four-way cross
  input_dict_10 = {
      'inputs': ShapeableList([
          np.array([['a']], dtype=object),
          np.array([['b']], dtype=object),
          np.array([['c']], dtype=object),
          np.array([['d']], dtype=object)
      ]),
      'name': 'four_way_cross',
      'separator': ':'
  }
  list_of_inputs.append(copy.deepcopy(input_dict_10))

  return list_of_inputs

generated_inputs["tf.sparse.cross"] = tf_sparse_cross_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.cross' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.cross'.")

check_valid('tf.sparse.cross', generated_inputs['tf.sparse.cross'], lib="tf", suffix=0)
