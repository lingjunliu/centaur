
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_queue_queuebase_inputs():
    """
    Generates a list of valid inputs for the tf.queue.QueueBase constructor.
    """
    list_of_inputs = []

    # Helper function to create a queue_ref tensor.
    # tf.raw_ops.*QueueV2 requires fully defined shapes.
    def create_queue_ref(tf_dtypes, shapes, capacity=-1, container='', shared_name=''):
        return tf.raw_ops.FIFOQueueV2(
            component_types=tf_dtypes,
            shapes=shapes,
            capacity=capacity,
            container=container,
            shared_name=shared_name
        )

    # Input 1: Basic case with a single float32 tensor
    dtypes1 = [np.float32]
    shapes1 = [(10,)]
    names1 = ['a']
    queue_ref1 = create_queue_ref([tf.float32], shapes1)
    list_of_inputs.append(copy.deepcopy({
        'dtypes': dtypes1,
        'shapes': shapes1,
        'names': names1,
        'queue_ref': queue_ref1
    }))

    # Input 2: Two components, int32 matrix and bool scalar
    dtypes2 = [np.int32, np.bool_]
    shapes2 = [(3, 4), ()]
    names2 = ['b', 'c']
    queue_ref2 = create_queue_ref([tf.int32, tf.bool], shapes2)
    list_of_inputs.append(copy.deepcopy({
        'dtypes': dtypes2,
        'shapes': shapes2,
        'names': names2,
        'queue_ref': queue_ref2
    }))

    # Input 3: Fully defined shape
    dtypes3 = [np.float64]
    shapes3 = [(1, 128)]
    names3 = ['d']
    queue_ref3 = create_queue_ref([tf.float64], shapes3)
    list_of_inputs.append(copy.deepcopy({
        'dtypes': dtypes3,
        'shapes': shapes3,
        'names': names3,
        'queue_ref': queue_ref3
    }))

    # Input 4: Complex number type
    dtypes4 = [np.complex64]
    shapes4 = [(2, 2)]
    names4 = ['e']
    queue_ref4 = create_queue_ref([tf.complex64], shapes4)
    list_of_inputs.append(copy.deepcopy({
        'dtypes': dtypes4,
        'shapes': shapes4,
        'names': names4,
        'queue_ref': queue_ref4
    }))

    # Input 5: Mixed precision types
    dtypes5 = [np.float16, np.float32]
    shapes5 = [(100,), (100,)]
    names5 = ['f', 'g']
    queue_ref5 = create_queue_ref([tf.float16, tf.float32], shapes5)
    list_of_inputs.append(copy.deepcopy({
        'dtypes': dtypes5,
        'shapes': shapes5,
        'names': names5,
        'queue_ref': queue_ref5
    }))

    # Input 6: High-dimensional tensor
    dtypes6 = [np.uint8]
    shapes6 = [(32, 32, 3, 1)]
    names6 = ['h']
    queue_ref6 = create_queue_ref([tf.uint8], shapes6)
    list_of_inputs.append(copy.deepcopy({
        'dtypes': dtypes6,
        'shapes': shapes6,
        'names': names6,
        'queue_ref': queue_ref6
    }))

    # Input 7: Single scalar component
    dtypes7 = [np.int64]
    shapes7 = [()]
    names7 = ['i']
    queue_ref7 = create_queue_ref([tf.int64], shapes7)
    list_of_inputs.append(copy.deepcopy({
        'dtypes': dtypes7,
        'shapes': shapes7,
        'names': names7,
        'queue_ref': queue_ref7
    }))

    # Input 8: Three components including a string
    dtypes8 = [np.int16, np.float32, np.string_]
    shapes8 = [(64, 64), (10,), ()]
    names8 = ['j', 'k', 'l']
    queue_ref8 = create_queue_ref([tf.int16, tf.float32, tf.string], shapes8)
    list_of_inputs.append(copy.deepcopy({
        'dtypes': dtypes8,
        'shapes': shapes8,
        'names': names8,
        'queue_ref': queue_ref8
    }))

    # Input 9: Fully defined shapes for image batch and labels
    dtypes9 = [np.float32, np.int64]
    shapes9 = [(4, 224, 224, 3), (4,)]
    names9 = ['m', 'n']
    queue_ref9 = create_queue_ref([tf.float32, tf.int64], shapes9)
    list_of_inputs.append(copy.deepcopy({
        'dtypes': dtypes9,
        'shapes': shapes9,
        'names': names9,
        'queue_ref': queue_ref9
    }))

    # Input 10: Using complex128
    dtypes10 = [np.complex128, np.int8]
    shapes10 = [(4, 4), (4,)]
    names10 = ['o', 'p']
    queue_ref10 = create_queue_ref([tf.complex128, tf.int8], shapes10)
    list_of_inputs.append(copy.deepcopy({
        'dtypes': dtypes10,
        'shapes': shapes10,
        'names': names10,
        'queue_ref': queue_ref10
    }))

    return list_of_inputs

generated_inputs["tf.queue.QueueBase"] = get_tf_queue_queuebase_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.queue.QueueBase' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.queue.QueueBase'.")

check_valid('tf.queue.QueueBase', generated_inputs['tf.queue.QueueBase'], lib="tf", suffix=0)
