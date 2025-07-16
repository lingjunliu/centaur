
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_nondifferentiable_batch_function_inputs():
    list_of_inputs = []

    # Input 1, valid
    input_dict = {
        'num_batch_threads': 1,
        'max_batch_size': 2,
        'batch_timeout_micros': 3,
        'allowed_batch_sizes': [],
        'max_enqueued_batches': 10,
        'autograph': True,
        'enable_large_batch_splitting': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    input_dict = {
        'num_batch_threads': 2,
        'max_batch_size': 4,
        'batch_timeout_micros': 100,
        'allowed_batch_sizes': [2, 4],
        'max_enqueued_batches': 5,
        'autograph': False,
        'enable_large_batch_splitting': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    input_dict = {
        'num_batch_threads': 4,
        'max_batch_size': 8,
        'batch_timeout_micros': 500,
        'allowed_batch_sizes': [4, 8],
        'max_enqueued_batches': 15,
        'autograph': True,
        'enable_large_batch_splitting': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid
    input_dict = {
        'num_batch_threads': 1,
        'max_batch_size': 16,
        'batch_timeout_micros': 1000,
        'allowed_batch_sizes': [4, 8, 16],
        'max_enqueued_batches': 20,
        'autograph': False,
        'enable_large_batch_splitting': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid
    input_dict = {
        'num_batch_threads': 3,
        'max_batch_size': 32,
        'batch_timeout_micros': 2000,
        'allowed_batch_sizes': [8, 16, 32],
        'max_enqueued_batches': 25,
        'autograph': True,
        'enable_large_batch_splitting': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid
    input_dict = {
        'num_batch_threads': 5,
        'max_batch_size': 64,
        'batch_timeout_micros': 4000,
        'allowed_batch_sizes': [16, 32, 64],
        'max_enqueued_batches': 30,
        'autograph': False,
        'enable_large_batch_splitting': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7, valid
    input_dict = {
        'num_batch_threads': 10,
        'max_batch_size': 128,
        'batch_timeout_micros': 8000,
        'allowed_batch_sizes': [32, 64, 128],
        'max_enqueued_batches': 35,
        'autograph': True,
        'enable_large_batch_splitting': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid
    input_dict = {
        'num_batch_threads': 1,
        'max_batch_size': 5,
        'batch_timeout_micros': 1,
        'allowed_batch_sizes': [1,2,3,4,5],
        'max_enqueued_batches': 1,
        'autograph': True,
        'enable_large_batch_splitting': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid
    input_dict = {
        'num_batch_threads': 3,
        'max_batch_size': 10,
        'batch_timeout_micros': 50,
        'allowed_batch_sizes': [5, 10],
        'max_enqueued_batches': 8,
        'autograph': False,
        'enable_large_batch_splitting': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid
    input_dict = {
        'num_batch_threads': 2,
        'max_batch_size': 20,
        'batch_timeout_micros': 150,
        'allowed_batch_sizes': [5, 10, 15, 20],
        'max_enqueued_batches': 12,
        'autograph': True,
        'enable_large_batch_splitting': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


def call_api(num_batch_threads, max_batch_size, batch_timeout_micros, allowed_batch_sizes, max_enqueued_batches, autograph, enable_large_batch_splitting):
  @tf.function
  def func(x):
    return x * 2

  batched_func = tf.nondifferentiable_batch_function(num_batch_threads, max_batch_size, batch_timeout_micros, allowed_batch_sizes, max_enqueued_batches, autograph, enable_large_batch_splitting)(func)

  input_tensor = tf.constant([1.0, 2.0, 3.0])
  result = batched_func(input_tensor)

  return result

generated_inputs = {}
generated_inputs["tf.nondifferentiable_batch_function"] = []

for i in range(10):
  num_batch_threads = np.random.randint(1, 11)
  max_batch_size = np.random.randint(2, 65)
  batch_timeout_micros = np.random.randint(1, 1000)
  num_allowed_sizes = np.random.randint(0,4)
  allowed_batch_sizes = sorted(np.random.choice(np.arange(1,max_batch_size+1), size=num_allowed_sizes, replace=False).tolist())
  if len(allowed_batch_sizes) > 0 and allowed_batch_sizes[-1] != max_batch_size:
      if max_batch_size not in allowed_batch_sizes:
        allowed_batch_sizes.append(max_batch_size)
        allowed_batch_sizes.sort()


  max_enqueued_batches = np.random.randint(1, 21)
  autograph = np.random.choice([True, False])
  enable_large_batch_splitting = np.random.choice([True, False])

  input_dict = {
      'num_batch_threads': num_batch_threads,
      'max_batch_size': max_batch_size,
      'batch_timeout_micros': batch_timeout_micros,
      'allowed_batch_sizes': allowed_batch_sizes,
      'max_enqueued_batches': max_enqueued_batches,
      'autograph': autograph,
      'enable_large_batch_splitting': enable_large_batch_splitting,
  }

  generated_inputs["tf.nondifferentiable_batch_function"].append(input_dict)

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nondifferentiable_batch_function' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nondifferentiable_batch_function'.")

check_valid('tf.nondifferentiable_batch_function', generated_inputs['tf.nondifferentiable_batch_function'], lib="tf", suffix=0)
