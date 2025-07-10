
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_concat_inputs():
    list_of_inputs = []

    # Input 1
    t1 = np.array([[1, 2, 3], [4, 5, 6]])
    t2 = np.array([[7, 8, 9], [10, 11, 12]])
    values = [t1, t2]
    axis = 0
    name = "concat_example_1"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    t1 = np.array([[1, 2, 3], [4, 5, 6]])
    t2 = np.array([[7, 8, 9], [10, 11, 12]])
    values = [t1, t2]
    axis = 1
    name = "concat_example_2"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    t1 = np.array([[[1, 2], [2, 3]], [[4, 4], [5, 3]]])
    t2 = np.array([[[7, 4], [8, 4]], [[2, 10], [15, 11]]])
    values = [t1, t2]
    axis = -1
    name = "concat_example_3"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    t1 = np.array([1, 2, 3])
    t2 = np.array([4, 5, 6])
    values = [t1, t2]
    axis = 0
    name = "concat_example_4"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    t1 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    t2 = np.array([[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]])
    values = [t1, t2]
    axis = 0
    name = "concat_example_5"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    t1 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    t2 = np.array([[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]])
    values = [t1, t2]
    axis = 1
    name = "concat_example_6"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    t1 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    t2 = np.array([[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]])
    values = [t1, t2]
    axis = 2
    name = "concat_example_7"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    t1 = np.array([1, 2, 3])
    t2 = np.array([4, 5, 6])
    t3 = np.array([7, 8, 9])
    values = [t1, t2, t3]
    axis = 0
    name = "concat_example_8"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    t1 = np.array([[1, 2], [3, 4]])
    t2 = np.array([[5, 6], [7, 8]])
    values = [t1, t2]
    axis = -2
    name = "concat_example_9"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    t1 = np.array([[[1], [2]], [[3], [4]]])
    t2 = np.array([[[5], [6]], [[7], [8]]])
    values = [t1, t2]
    axis = -1
    name = "concat_example_10"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}

def check_valid(api, input_list, lib="tf", suffix=0):
  from fuzzingbook.GrammarFuzzer import cheader
  from fuzzingbook.GrammarFuzzer import CppGrammarFuzzer
  from fuzzingbook.Parser import EarleyParser
  from fuzzingbook.GrammarFuzzer import GrammarFuzzer
  from fuzzingbook.Generator import Generator

  def get_signature(api, lib="torch", suffix=0):
    if lib == "tf":
      import tensorflow as tf
      return tf.compat.v1.flags.FLAGS.api_signatures[api]
    else:
      import torch
      return torch.FLAGS.api_signatures[api]

  def get_domain(type):
      if type == 'tensor':
          return 'TENSOR'
      elif type == 'tensor_list':
          return 'TENSOR_LIST'
      elif type == 'number':
          return 'NUMBER'
      elif type == 'string':
          return 'STRING'
      elif type == 'integer':
          return 'INTEGER'
      elif type == 'boolean':
          return 'BOOLEAN'
      elif type == 'float':
          return 'FLOAT'
      else:
          return "ANY"

  def get_abstract_input(concrete, signature):
      abstract = {}
      for arg in signature:
          type = signature[arg]
          domain = get_domain(type)
          if domain == "TENSOR" or domain == "NUMBER" or domain == "FLOAT" or domain == 'INTEGER':
              abstract[arg] = concrete[arg]
          elif domain == 'TENSOR_LIST':
              abstract[arg] = concrete[arg]
          else:
              abstract[arg] = concrete[arg]
      return abstract
  
  def get_ll(domain, value):
      if domain == 'TENSOR':
          list_val = list(value.shape)
          ll = "<{}>".format(",".join([str(x) for x in list_val]))
          return ll
      elif domain == 'TENSOR_LIST':
          if isinstance(value, list):
              ll = []
              for v in value:
                  list_val = list(v.shape)
                  ll.append("<{}>".format(",".join([str(x) for x in list_val])))
              return ll
          else:
              list_val = list(value.shape)
              ll = "<{}>".format(",".join([str(x) for x in list_val]))
              return ll
      elif domain == 'NUMBER':
          return "number"
      elif domain == 'STRING':
          return "string"
      elif domain == 'INTEGER':
          return "integer"
      elif domain == 'BOOLEAN':
          return "boolean"
      elif domain == 'FLOAT':
          return "float"
      else:
          return "ANY"
  
  for input_dict in input_list:
    _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))

generated_inputs["tf.concat"] = tf_concat_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.concat' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.concat'.")

check_valid('tf.concat', generated_inputs['tf.concat'], lib="tf", suffix=0)
