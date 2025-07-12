
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_queue_QueueBase_inputs():
    list_of_inputs = []

    # Input 1
    dtypes = [np.int32]
    shapes = [[10]]
    names = ['data']
    queue = tf.queue.FIFOQueue(capacity=10, dtypes=[tf.int32], shapes=[[10]])
    queue_ref = queue.queue_ref

    input_dict = {
        "dtypes": dtypes,
        "shapes": shapes,
        "names": names,
        "queue_ref": queue_ref
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    dtypes = [np.float32, np.int64]
    shapes = [[], [5, 5]]
    names = ['value1', 'value2']
    queue = tf.queue.FIFOQueue(capacity=5, dtypes=[tf.float32, tf.int64], shapes=[[], [5, 5]])
    queue_ref = queue.queue_ref

    input_dict = {
        "dtypes": dtypes,
        "shapes": shapes,
        "names": names,
        "queue_ref": queue_ref
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    dtypes = [np.string_]
    shapes = [[1]]
    names = ['text']
    queue = tf.queue.FIFOQueue(capacity=5, dtypes=[tf.string], shapes=[[1]])
    queue_ref = queue.queue_ref

    input_dict = {
        "dtypes": dtypes,
        "shapes": shapes,
        "names": names,
        "queue_ref": queue_ref
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    dtypes = [np.bool_]
    shapes = [[2, 3, 4]]
    names = ['mask']
    queue = tf.queue.FIFOQueue(capacity=5, dtypes=[tf.bool], shapes=[[2,3,4]])
    queue_ref = queue.queue_ref

    input_dict = {
        "dtypes": dtypes,
        "shapes": shapes,
        "names": names,
        "queue_ref": queue_ref
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    dtypes = [np.int32, np.float64, np.string_]
    shapes = [[], [10, 10], [1]]
    names = ['id', 'features', 'label']
    queue = tf.queue.FIFOQueue(capacity=5, dtypes=[tf.int32, tf.float64, tf.string], shapes=[[], [10, 10], [1]])
    queue_ref = queue.queue_ref

    input_dict = {
        "dtypes": dtypes,
        "shapes": shapes,
        "names": names,
        "queue_ref": queue_ref
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    dtypes = [np.uint8]
    shapes = [[1, 28, 28, 1]]
    names = ['image']
    queue = tf.queue.FIFOQueue(capacity=5, dtypes=[tf.uint8], shapes=[[1,28,28,1]])
    queue_ref = queue.queue_ref

    input_dict = {
        "dtypes": dtypes,
        "shapes": shapes,
        "names": names,
        "queue_ref": queue_ref
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7
    dtypes = [np.int16, np.float16]
    shapes = [[4], []]
    names = ['short_data', 'half_precision']
    queue =  tf.queue.FIFOQueue(capacity=5, dtypes=[tf.int16, tf.float16], shapes=[[4], []])
    queue_ref = queue.queue_ref

    input_dict = {
        "dtypes": dtypes,
        "shapes": shapes,
        "names": names,
        "queue_ref": queue_ref
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    dtypes = [np.complex64]
    shapes = [[2, 2]]
    names = ['complex_matrix']
    queue = tf.queue.FIFOQueue(capacity=5, dtypes=[tf.complex64], shapes=[[2,2]])
    queue_ref = queue.queue_ref

    input_dict = {
        "dtypes": dtypes,
        "shapes": shapes,
        "names": names,
        "queue_ref": queue_ref
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    dtypes = [tf.variant]
    shapes = [[3, 3]]
    names = ['arbitrary_object']
    queue = tf.queue.FIFOQueue(capacity=5, dtypes=[tf.variant], shapes=[[3,3]])
    queue_ref = queue.queue_ref

    input_dict = {
        "dtypes": dtypes,
        "shapes": shapes,
        "names": names,
        "queue_ref": queue_ref
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    dtypes = [np.int8, np.uint16, np.float32, np.string_]
    shapes = [[], [1], [2, 3], [2]]
    names = ['sbyte', 'ushort', 'fp', 'text']
    queue = tf.queue.FIFOQueue(capacity=5, dtypes=[tf.int8, tf.uint16, tf.float32, tf.string], shapes=[[], [1], [2, 3], [2]])
    queue_ref = queue.queue_ref

    input_dict = {
        "dtypes": dtypes,
        "shapes": shapes,
        "names": names,
        "queue_ref": queue_ref
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.queue.QueueBase"] = tf_queue_QueueBase_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.queue.QueueBase' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.queue.QueueBase'.")

check_valid('tf.queue.QueueBase', generated_inputs['tf.queue.QueueBase'], lib="tf", suffix=0)
