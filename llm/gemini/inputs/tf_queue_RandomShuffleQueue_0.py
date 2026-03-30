
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_queue_RandomShuffleQueue_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "capacity": 10,
        "min_after_dequeue": 5,
        "dtypes": [np.int32],
        "shapes": [(2, 2)],
        "names": ['a'],
        "seed": 123,
        "shared_name": "queue1",
        "name": "random_queue_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "capacity": 20,
        "min_after_dequeue": 10,
        "dtypes": [np.float32, np.int64],
        "shapes": [(3, 3), ()],
        "names": ['b', 'c'],
        "seed": 456,
        "shared_name": "queue2",
        "name": "random_queue_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "capacity": 5,
        "min_after_dequeue": 2,
        "dtypes": [tf.string],
        "shapes": [(1,)],
        "names": ['d'],
        "seed": 789,
        "shared_name": "queue3",
        "name": "random_queue_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "capacity": 15,
        "min_after_dequeue": 7,
        "dtypes": [np.bool_],
        "shapes": [(4, 4, 4)],
        "names": ['e'],
        "seed": 101,
        "shared_name": "queue4",
        "name": "random_queue_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "capacity": 8,
        "min_after_dequeue": 3,
        "dtypes": [np.int8, np.float64, tf.string],
        "shapes": [(), (2,), (1, 1)],
        "names": ['f', 'g', 'h'],
        "seed": 202,
        "shared_name": "queue5",
        "name": "random_queue_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "capacity": 12,
        "min_after_dequeue": 6,
        "dtypes": [np.uint16],
        "shapes": [(5,)],
        "names": ['i'],
        "seed": 303,
        "shared_name": "queue6",
        "name": "random_queue_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "capacity": 30,
        "min_after_dequeue": 15,
        "dtypes": [np.complex64],
        "shapes": [(6, 6)],
        "names": ['j'],
        "seed": 404,
        "shared_name": "queue7",
        "name": "random_queue_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "capacity": 7,
        "min_after_dequeue": 1,
        "dtypes": [np.int16],
        "shapes": [(7, 7, 7, 7)],
        "names": ['k'],
        "seed": 505,
        "shared_name": "queue8",
        "name": "random_queue_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "capacity": 25,
        "min_after_dequeue": 12,
        "dtypes": [np.float16, np.int32],
        "shapes": [(), ()],
        "names": ['l', 'm'],
        "seed": 606,
        "shared_name": "queue9",
        "name": "random_queue_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "capacity": 3,
        "min_after_dequeue": 1,
        "dtypes": [np.uint32],
        "shapes": [(8,)],
        "names": ['n'],
        "seed": 707,
        "shared_name": "queue10",
        "name": "random_queue_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.queue.RandomShuffleQueue"] = tf_queue_RandomShuffleQueue_inputs()

def check_valid(api, inputs, lib="tf", suffix=0):
    import inspect
    api_func = eval(api)
    sig = inspect.signature(api_func)
    parameters = sig.parameters
    anno = {}
    for param in parameters:
        anno[param] = str(parameters[param].annotation)
    return anno

    def get_ll(domain, value):
        import numpy as np
        if type(value) == str:
            return f'"{value}"'
        if type(value) == bool:
            return "True" if value else "False"
        if type(value) == int or type(value) == np.int32 or type(value) == np.int64:
            return str(value)
        if type(value) == float or type(value) == np.float32 or type(value) == np.float64:
            return str(value)
        if value is tf.string:
            return "tf.string"
        if type(value) == list:
            return "[" + ",".join([get_ll(domain, i) for i in value]) + "]"
        if type(value) == tuple:
            return "[" + ",".join([get_ll(domain, i) for i in value]) + "]"
        if type(value) == np.dtype:
            return "np." + str(value)
        return "None"

    def get_abstract_input(concrete, signature):
        abstract = {}
        for arg in signature:
            domain = signature[arg]
            if arg not in concrete:
                continue
            value = concrete[arg]
            abstract[arg] = get_ll(domain, value)
        return abstract

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.queue.RandomShuffleQueue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.queue.RandomShuffleQueue'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.queue.RandomShuffleQueue', generated_inputs['tf.queue.RandomShuffleQueue'], lib="tf", suffix=0)
