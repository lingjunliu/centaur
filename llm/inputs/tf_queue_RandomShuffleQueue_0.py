
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_queue_randomshufflequeue_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "capacity": 10,
        "min_after_dequeue": 5,
        "dtypes": [np.int32],
        "shapes": [(2,)],
        "names": [b'input1'],
        "seed": 123,
        "shared_name": b'queue1'.decode('utf-8'),
        "name": 'random_queue1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "capacity": 20,
        "min_after_dequeue": 10,
        "dtypes": [np.float32, np.int64],
        "shapes": [(1, 2), ()],
        "names": [b'input2_1', b'input2_2'],
        "seed": 456,
        "shared_name": b'queue2'.decode('utf-8'),
        "name": 'random_queue2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "capacity": 5,
        "min_after_dequeue": 2,
        "dtypes": [np.bool_],
        "shapes": [(3, 3, 3)],
        "names": [b'input3'],
        "seed": 789,
        "shared_name": b'queue3'.decode('utf-8'),
        "name": 'random_queue3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "capacity": 15,
        "min_after_dequeue": 7,
        "dtypes": [np.string_],
        "shapes": [()],
        "names": [b'input4'],
        "seed": 101,
        "shared_name": b'queue4'.decode('utf-8'),
        "name": 'random_queue4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "capacity": 8,
        "min_after_dequeue": 4,
        "dtypes": [np.int32, np.float64, np.string_],
        "shapes": [(), (2, 2), (1,)],
        "names": [b'in5_1', b'in5_2', b'in5_3'],
        "seed": 112,
        "shared_name": b'queue5'.decode('utf-8'),
        "name": 'random_queue5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "capacity": 12,
        "min_after_dequeue": 6,
        "dtypes": [np.uint8],
        "shapes": [(4,)],
        "names": [b'input6'],
        "seed": 131,
        "shared_name": b'queue6'.decode('utf-8'),
        "name": 'random_queue6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "capacity": 3,
        "min_after_dequeue": 1,
        "dtypes": [np.int16],
        "shapes": [(1, 1, 1)],
        "names": [b'input7'],
        "seed": 141,
        "shared_name": b'queue7'.decode('utf-8'),
        "name": 'random_queue7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "capacity": 25,
        "min_after_dequeue": 12,
        "dtypes": [np.complex64],
        "shapes": [(5,)],
        "names": [b'input8'],
        "seed": 151,
        "shared_name": b'queue8'.decode('utf-8'),
        "name": 'random_queue8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "capacity": 7,
        "min_after_dequeue": 3,
        "dtypes": [np.float16],
        "shapes": [(2, 3)],
        "names": [b'input9'],
        "seed": 161,
        "shared_name": b'queue9'.decode('utf-8'),
        "name": 'random_queue9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "capacity": 30,
        "min_after_dequeue": 15,
        "dtypes": [np.int8, np.uint16],
        "shapes": [(), ()],
        "names": [b'in10_1', b'in10_2'],
        "seed": 171,
        "shared_name": b'queue10'.decode('utf-8'),
        "name": 'random_queue10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.queue.RandomShuffleQueue"] = tf_queue_randomshufflequeue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.queue.RandomShuffleQueue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.queue.RandomShuffleQueue'.")

check_valid('tf.queue.RandomShuffleQueue', generated_inputs['tf.queue.RandomShuffleQueue'], lib="tf", suffix=0)
