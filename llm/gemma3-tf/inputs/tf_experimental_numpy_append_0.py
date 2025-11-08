
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_append_inputs():
    list_of_inputs = []

    arr1 = np.array([1, 2, 3])
    values1 = np.array([4, 5, 6])
    axis1 = None
    input_dict1 = {"arr": arr1, "values": values1, "axis": axis1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    arr2 = np.array([[1, 2], [3, 4]])
    values2 = np.array([[5, 6]])
    axis2 = 0
    input_dict2 = {"arr": arr2, "values": values2, "axis": axis2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    arr3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    values3 = np.array([[[9, 10], [11, 12]]])
    axis3 = 0
    input_dict3 = {"arr": arr3, "values": values3, "axis": axis3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    arr4 = np.array([1, 2, 3])
    values4 = np.array([4])
    axis4 = None
    input_dict4 = {"arr": arr4, "values": values4, "axis": axis4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    arr5 = np.array([[1, 2], [3, 4]])
    values5 = np.array([[5], [6]])
    axis5 = 1
    input_dict5 = {"arr": arr5, "values": values5, "axis": axis5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    arr6 = np.array([[[1, 2], [3, 4]]])
    values6 = np.array([[[5, 6], [7, 8]]])
    axis6 = 0
    input_dict6 = {"arr": arr6, "values": values6, "axis": axis6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    arr7 = np.array([1, 2, 3, 4])
    values7 = np.array([5, 6])
    axis7 = None
    input_dict7 = {"arr": arr7, "values": values7, "axis": axis7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    arr8 = np.array([[1, 2, 3], [4, 5, 6]])
    values8 = np.array([[7, 8, 9]])
    axis8 = 0
    input_dict8 = {"arr": arr8, "values": values8, "axis": axis8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    arr9 = np.array([1, 2, 3])
    values9 = np.array([4, 5, 6])
    axis9 = -1
    input_dict9 = {"arr": arr9, "values": values9, "axis": axis9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    arr10 = np.array([[[1, 2]]])
    values10 = np.array([[[3, 4]]])
    axis10 = 0
    input_dict10 = {"arr": arr10, "values": values10, "axis": axis10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.append"] = tf_experimental_numpy_append_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.append' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.append'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.append', generated_inputs['tf.experimental.numpy.append'], lib="tf", suffix=0)
