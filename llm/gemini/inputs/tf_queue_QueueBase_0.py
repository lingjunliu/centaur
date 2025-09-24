
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_queue_queuebase_inputs():
    """
    Generates a list of valid inputs for the tf.queue.QueueBase constructor.
    Note: tf.queue.QueueBase is an abstract base class and is not intended
    to be instantiated directly by users. Concrete implementations like
    tf.queue.FIFOQueue should be used. To generate a valid 'queue_ref' tensor,
    which is a resource handle, we create a dummy queue from a concrete
    class and extract its handle.
    """
    list_of_inputs = []

    def _create_queue_ref(dtypes, shapes):
        """
        Helper to create a valid queue_ref tensor.
        The created queue's names are auto-generated and not used, as we
        provide the names for the QueueBase constructor separately.
        """
        tf_dtypes = [tf.as_dtype(dt) for dt in dtypes]

        is_fully_defined = True
        if shapes is None:
            is_fully_defined = False
        else:
            for s in shapes:
                if s is None or any(d is None for d in s):
                    is_fully_defined = False
                    break
        
        if is_fully_defined:
            dummy_queue = tf.queue.FIFOQueue(capacity=10, dtypes=tf_dtypes, shapes=shapes)
        else:
            dummy_queue = tf.queue.FIFOQueue(capacity=10, dtypes=tf_dtypes)
        
        return dummy_queue.queue_ref

    # Input 1: Single float component
    dtypes1 = [np.float32]
    shapes1 = [[1]]
    names1 = ["vector_float"]
    input_dict1 = {
        'dtypes': dtypes1,
        'shapes': shapes1,
        'names': names1,
        'queue_ref': _create_queue_ref(dtypes1, shapes1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Two components: int vector and float matrix
    dtypes2 = [np.int32, np.float64]
    shapes2 = [[3], [2, 2]]
    names2 = ["int_vec", "float_mat"]
    input_dict2 = {
        'dtypes': dtypes2,
        'shapes': shapes2,
        'names': names2,
        'queue_ref': _create_queue_ref(dtypes2, shapes2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Multiple components with various types
    dtypes3 = [np.float32, np.int64, np.bool_, tf.string]
    shapes3 = [[1], [5], [1], [2]]
    names3 = ["float_val", "long_int_array", "bool_flag", "string_pair"]
    input_dict3 = {
        'dtypes': dtypes3,
        'shapes': shapes3,
        'names': names3,
        'queue_ref': _create_queue_ref(dtypes3, shapes3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Shapes with unknown dimensions (using None)
    dtypes4 = [np.float16, np.int8]
    shapes4 = [[None, 3], [None]]
    names4 = ["partially_known_shape", "unknown_vector"]
    input_dict4 = {
        'dtypes': dtypes4,
        'shapes': shapes4,
        'names': names4,
        'queue_ref': _create_queue_ref(dtypes4, shapes4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Fully unknown shapes
    dtypes5 = [np.complex64, np.uint8]
    shapes5 = [None, None]
    names5 = ["unknown_complex", "unknown_uint"]
    input_dict5 = {
        'dtypes': dtypes5,
        'shapes': shapes5,
        'names': names5,
        'queue_ref': _create_queue_ref(dtypes5, shapes5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: High-dimensional tensor
    dtypes6 = [np.float32]
    shapes6 = [[2, 3, 4, 5]]
    names6 = ["4d_tensor"]
    input_dict6 = {
        'dtypes': dtypes6,
        'shapes': shapes6,
        'names': names6,
        'queue_ref': _create_queue_ref(dtypes6, shapes6)
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Mix of known, unknown, and scalar-like shapes
    dtypes7 = [np.int16, np.float32, tf.string]
    shapes7 = [[10], [1], [None, None]]
    names7 = ["fixed_vec", "scalar_val", "unknown_matrix"]
    input_dict7 = {
        'dtypes': dtypes7,
        'shapes': shapes7,
        'names': names7,
        'queue_ref': _create_queue_ref(dtypes7, shapes7)
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Complex numbers
    dtypes8 = [np.complex64, np.complex128]
    shapes8 = [[2], [2]]
    names8 = ["complex64_vec", "complex128_vec"]
    input_dict8 = {
        'dtypes': dtypes8,
        'shapes': shapes8,
        'names': names8,
        'queue_ref': _create_queue_ref(dtypes8, shapes8)
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Using bfloat16 (as tf.bfloat16 since it has no numpy equivalent)
    dtypes9 = [tf.bfloat16]
    shapes9 = [[4, 4]]
    names9 = ["bfloat_matrix"]
    input_dict9 = {
        'dtypes': dtypes9,
        'shapes': shapes9,
        'names': names9,
        'queue_ref': _create_queue_ref(dtypes9, shapes9)
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Long list of components
    dtypes10 = [np.float32, np.int32, tf.string, np.bool_, np.float64, np.int64]
    shapes10 = [[1], [1], [2], [3], [4], [5]]
    names10 = ["c1", "c2", "c3", "c4", "c5", "c6"]
    input_dict10 = {
        'dtypes': dtypes10,
        'shapes': shapes10,
        'names': names10,
        'queue_ref': _create_queue_ref(dtypes10, shapes10)
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

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

check_valid('tf.queue.QueueBase', generated_inputs['tf.queue.QueueBase'], lib="tf", suffix=0)
