
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_queue_ref(component_types, shapes, capacity=10, container='', shared_name=''):
    """Helper to create a queue reference tensor."""
    # In a graph context, shapes can be tf.TensorShape objects.
    # For compatibility, we'll convert lists/tuples to tf.TensorShape.
    shapes_as_tensorshape = [tf.TensorShape(s) for s in shapes]
    # TensorFlow maps numpy dtypes to its own DType objects.
    component_types_tf = [tf.as_dtype(d) for d in component_types]
    return tf.raw_ops.FIFOQueueV2(
        component_types=component_types_tf,
        shapes=shapes_as_tensorshape,
        capacity=capacity,
        container=container,
        shared_name=shared_name
    )

def tf_queue_QueueBase_inputs():
    list_of_inputs = []

    # Input 1: Basic case with int32 and float32.
    dtypes1 = [np.int32, np.float32]
    shapes1 = [(2, 3), ()]
    names1 = ['int_matrix', 'float_scalar']
    input_dict1 = {
        'dtypes': dtypes1,
        'shapes': shapes1,
        'names': names1,
        'queue_ref': get_queue_ref(dtypes1, shapes1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: String and boolean types with fully defined shapes.
    dtypes2 = [np.string_, np.bool_]
    shapes2 = [(4, 5), (10,)]
    names2 = ['string_tensor', 'bool_vector']
    input_dict2 = {
        'dtypes': dtypes2,
        'shapes': shapes2,
        'names': names2,
        'queue_ref': get_queue_ref(dtypes2, shapes2, capacity=20)
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Single component with a high-rank tensor.
    dtypes3 = [np.float64]
    shapes3 = [(1, 2, 3, 4)]
    names3 = ['high_rank_tensor']
    input_dict3 = {
        'dtypes': dtypes3,
        'shapes': shapes3,
        'names': names3,
        'queue_ref': get_queue_ref(dtypes3, shapes3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Multiple scalar components.
    dtypes4 = [np.int8, np.uint16, np.complex64]
    shapes4 = [(), (), ()]
    names4 = ['int8_val', 'uint16_val', 'complex64_val']
    input_dict4 = {
        'dtypes': dtypes4,
        'shapes': shapes4,
        'names': names4,
        'queue_ref': get_queue_ref(dtypes4, shapes4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: A single component with a defined vector shape.
    dtypes5 = [np.int64]
    shapes5 = [(10,)]
    names5 = ['defined_shape_tensor']
    input_dict5 = {
        'dtypes': dtypes5,
        'shapes': shapes5,
        'names': names5,
        'queue_ref': get_queue_ref(dtypes5, shapes5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Complex numbers.
    dtypes6 = [np.complex64, np.complex128]
    shapes6 = [(4,), (2, 2)]
    names6 = ['c64_vec', 'c128_mat']
    input_dict6 = {
        'dtypes': dtypes6,
        'shapes': shapes6,
        'names': names6,
        'queue_ref': get_queue_ref(dtypes6, shapes6, capacity=5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Empty names list.
    dtypes7 = [np.float16, np.int32]
    shapes7 = [(1,), (1,)]
    names7 = []
    input_dict7 = {
        'dtypes': dtypes7,
        'shapes': shapes7,
        'names': names7,
        'queue_ref': get_queue_ref(dtypes7, shapes7)
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Larger number of components.
    dtypes8 = [np.int32, np.float32, np.string_, np.bool_, np.int64]
    shapes8 = [(1,), (2, 2), (3,), (4, 4), (5,)]
    names8 = ['c1', 'c2', 'c3', 'c4', 'c5']
    input_dict8 = {
        'dtypes': dtypes8,
        'shapes': shapes8,
        'names': names8,
        'queue_ref': get_queue_ref(dtypes8, shapes8)
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Using different integer types with defined shapes.
    dtypes9 = [np.uint8, np.int16, np.uint32, np.int64]
    shapes9 = [(), (10,), (3, 1), (5, 5)]
    names9 = ['uint8_s', 'int16_v', 'uint32_m', 'int64_m']
    input_dict9 = {
        'dtypes': dtypes9,
        'shapes': shapes9,
        'names': names9,
        'queue_ref': get_queue_ref(dtypes9, shapes9)
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    # Input 10: Fully defined matrix shapes.
    dtypes10 = [np.float32, np.float32]
    shapes10 = [[2, 3], [3, 4]]
    names10 = ['matrix_a', 'matrix_b']
    input_dict10 = {
        'dtypes': dtypes10,
        'shapes': shapes10,
        'names': names10,
        'queue_ref': get_queue_ref(dtypes10, shapes10)
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.queue.QueueBase"] = tf_queue_QueueBase_inputs()

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
