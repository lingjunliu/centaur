
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_queue_queuebase_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    dtypes = [tf.float32]
    shapes = [tf.TensorShape([2, 2])]
    names = ['tensor1']
    queue_ref = tf.compat.v1.get_variable("queue_ref1", initializer=tf.zeros(shape=(0, 2, 2), dtype=tf.float32), trainable=False, use_resource=True).handle

    input_dict = {
        "dtypes": dtypes,
        "shapes": shapes,
        "names": names,
        "queue_ref": queue_ref
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Multiple dtypes and shapes (excluding string to avoid issues)
    dtypes = [tf.int32, tf.bool]
    shapes = [tf.TensorShape([3]), tf.TensorShape([1, 1])]
    names = ['int_tensor', 'bool_tensor']
    queue_ref = tf.compat.v1.get_variable("queue_ref2", initializer=tf.zeros(shape=(0, 3), dtype=tf.int32), trainable=False, use_resource=True).handle

    input_dict = {
        "dtypes": dtypes,
        "shapes": shapes,
        "names": names,
        "queue_ref": queue_ref
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty names
    dtypes = [tf.float64]
    shapes = [tf.TensorShape([5])]
    names = []
    queue_ref =  tf.compat.v1.get_variable("queue_ref3", initializer=tf.zeros(shape=(0, 5), dtype=tf.float64), trainable=False, use_resource=True).handle

    input_dict = {
        "dtypes": dtypes,
        "shapes": shapes,
        "names": names,
        "queue_ref": queue_ref
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: None shape
    dtypes = [tf.complex64]
    shapes = [tf.TensorShape(None)]
    names = ['complex_tensor']
    queue_ref = tf.compat.v1.get_variable("queue_ref4", initializer=tf.zeros(shape=(0, 1), dtype=tf.complex64), trainable=False, use_resource=True).handle

    input_dict = {
        "dtypes": dtypes,
        "shapes": shapes,
        "names": names,
        "queue_ref": queue_ref
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different shapes dimensions
    dtypes = [tf.uint8, tf.int16]
    shapes = [tf.TensorShape([2, 3, 4]), tf.TensorShape([1])]
    names = ['uint_tensor', 'int16_tensor']
    queue_ref = tf.compat.v1.get_variable("queue_ref5", initializer=tf.zeros(shape=(0, 2, 3, 4), dtype=tf.uint8), trainable=False, use_resource=True).handle
    input_dict = {
        "dtypes": dtypes,
        "shapes": shapes,
        "names": names,
        "queue_ref": queue_ref
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty names, fixed shape for easier processing.
    dtypes = [tf.qint8]
    shapes = [tf.TensorShape([1])]
    names = []
    queue_ref = tf.compat.v1.get_variable("queue_ref6", initializer=tf.zeros(shape=(0, 1), dtype=tf.qint8), trainable=False, use_resource=True).handle
    input_dict = {
        "dtypes": dtypes,
        "shapes": shapes,
        "names": names,
        "queue_ref": queue_ref
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Only one element in the dtypes
    dtypes = [tf.quint8]
    shapes = [tf.TensorShape([4, 4])]
    names = ['tensor2']
    queue_ref = tf.compat.v1.get_variable("queue_ref7", initializer=tf.zeros(shape=(0, 4, 4), dtype=tf.quint8), trainable=False, use_resource=True).handle
    input_dict = {
        "dtypes": dtypes,
        "shapes": shapes,
        "names": names,
        "queue_ref": queue_ref
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: rank 0 shape, excluding problematic types
    dtypes = [tf.int64]
    shapes = [tf.TensorShape([])]
    names = ['scalar_int']
    queue_ref = tf.compat.v1.get_variable("queue_ref8", initializer=tf.zeros(shape=(0,), dtype=tf.int64), trainable=False, use_resource=True).handle
    input_dict = {
        "dtypes": dtypes,
        "shapes": shapes,
        "names": names,
        "queue_ref": queue_ref
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: empty list of names, one dtype and fixed shape
    dtypes = [tf.float32]
    shapes = [tf.TensorShape([2])]
    names = []
    queue_ref = tf.compat.v1.get_variable("queue_ref9", initializer=tf.zeros(shape=(0, 2), dtype=tf.float32), trainable=False, use_resource=True).handle
    input_dict = {
        "dtypes": dtypes,
        "shapes": shapes,
        "names": names,
        "queue_ref": queue_ref
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: different types of queue_ref (excluding float16 for potential errors in the environment)
    dtypes = [tf.int8]
    shapes = [tf.TensorShape([1, 2])]
    names = ['int8_example']
    queue_ref = tf.compat.v1.get_variable("queue_ref10", initializer=tf.zeros(shape=(0, 1, 2), dtype=tf.int8), trainable=False, use_resource=True).handle
    input_dict = {
        "dtypes": dtypes,
        "shapes": shapes,
        "names": names,
        "queue_ref": queue_ref
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
        
    # Input 11: different types of queue_ref with scalar
    dtypes = [tf.int8]
    shapes = [tf.TensorShape([])]
    names = ['int8_scalar']
    queue_ref = tf.compat.v1.get_variable("queue_ref11", initializer=tf.zeros(shape=(0,), dtype=tf.int8), trainable=False, use_resource=True).handle
    input_dict = {
        "dtypes": dtypes,
        "shapes": shapes,
        "names": names,
        "queue_ref": queue_ref
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.queue.QueueBase"] = tf_queue_queuebase_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.queue.QueueBase' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.queue.QueueBase'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.queue.QueueBase', generated_inputs['tf.queue.QueueBase'], lib="tf", suffix=0)
