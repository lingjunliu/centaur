
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_RequantizationRange_inputs():
    list_of_inputs = []

    # Input 1: qint8
    input1 = np.array([-10, -5, 0, 5, 10], dtype=np.int8)
    input_min1 = np.array([-10.0], dtype=np.float32)
    input_max1 = np.array([10.0], dtype=np.float32)
    input_dict1 = {"input": tf.constant(input1, dtype=tf.int8), "input_min": tf.constant(input_min1, dtype=tf.float32), "input_max": tf.constant(input_max1, dtype=tf.float32), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: quint8
    input2 = np.array([0, 5, 10, 15, 20], dtype=np.uint8)
    input_min2 = np.array([0.0], dtype=np.float32)
    input_max2 = np.array([20.0], dtype=np.float32)
    input_dict2 = {"input": tf.constant(input2, dtype=tf.uint8), "input_min": tf.constant(input_min2, dtype=tf.float32), "input_max": tf.constant(input_max2, dtype=tf.float32), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: qint32
    input3 = np.array([-1000, -500, 0, 500, 1000], dtype=np.int32)
    input_min3 = np.array([-1000.0], dtype=np.float32)
    input_max3 = np.array([1000.0], dtype=np.float32)
    input_dict3 = {"input": tf.constant(input3, dtype=tf.int32), "input_min": tf.constant(input_min3, dtype=tf.float32), "input_max": tf.constant(input_max3, dtype=tf.float32), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: qint16
    input4 = np.array([-100, -50, 0, 50, 100], dtype=np.int16)
    input_min4 = np.array([-100.0], dtype=np.float32)
    input_max4 = np.array([100.0], dtype=np.float32)
    input_dict4 = {"input": tf.constant(input4, dtype=tf.int16), "input_min": tf.constant(input_min4, dtype=tf.float32), "input_max": tf.constant(input_max4, dtype=tf.float32), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: quint16
    input5 = np.array([0, 50, 100, 150, 200], dtype=np.uint16)
    input_min5 = np.array([0.0], dtype=np.float32)
    input_max5 = np.array([200.0], dtype=np.float32)
    input_dict5 = {"input": tf.constant(input5, dtype=tf.uint16), "input_min": tf.constant(input_min5, dtype=tf.float32), "input_max": tf.constant(input_max5, dtype=tf.float32), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Multi-dimensional qint8
    input6 = np.array([[-10, -5], [0, 5], [10, 15]], dtype=np.int8)
    input_min6 = np.array([-15.0], dtype=np.float32)
    input_max6 = np.array([15.0], dtype=np.float32)
    input_dict6 = {"input": tf.constant(input6, dtype=tf.int8), "input_min": tf.constant(input_min6, dtype=tf.float32), "input_max": tf.constant(input_max6, dtype=tf.float32), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Multi-dimensional quint8
    input7 = np.array([[0, 5], [10, 15], [20, 25]], dtype=np.uint8)
    input_min7 = np.array([0.0], dtype=np.float32)
    input_max7 = np.array([25.0], dtype=np.float32)
    input_dict7 = {"input": tf.constant(input7, dtype=tf.uint8), "input_min": tf.constant(input_min7, dtype=tf.float32), "input_max": tf.constant(input_max7, dtype=tf.float32), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: qint32 with different min/max
    input8 = np.array([-5000, -2500, 0, 2500, 5000], dtype=np.int32)
    input_min8 = np.array([-6000.0], dtype=np.float32)
    input_max8 = np.array([6000.0], dtype=np.float32)
    input_dict8 = {"input": tf.constant(input8, dtype=tf.int32), "input_min": tf.constant(input_min8, dtype=tf.float32), "input_max": tf.constant(input_max8, dtype=tf.float32), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: quint16 with large values
    input9 = np.array([0, 10000, 20000, 30000, 40000], dtype=np.uint16)
    input_min9 = np.array([0.0], dtype=np.float32)
    input_max9 = np.array([40000.0], dtype=np.float32)
    input_dict9 = {"input": tf.constant(input9, dtype=tf.uint16), "input_min": tf.constant(input_min9, dtype=tf.float32), "input_max": tf.constant(input_max9, dtype=tf.float32), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: qint16 with different min/max and name
    input10 = np.array([-500, -250, 0, 250, 500], dtype=np.int16)
    input_min10 = np.array([-600.0], dtype=np.float32)
    input_max10 = np.array([600.0], dtype=np.float32)
    input_dict10 = {"input": tf.constant(input10, dtype=tf.int16), "input_min": tf.constant(input_min10, dtype=tf.float32), "input_max": tf.constant(input_max10, dtype=tf.float32), "name": "my_requant_range"}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.RequantizationRange"] = tf_raw_ops_RequantizationRange_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.RequantizationRange' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RequantizationRange'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.RequantizationRange', generated_inputs['tf.raw_ops.RequantizationRange'], lib="tf", suffix=0)
