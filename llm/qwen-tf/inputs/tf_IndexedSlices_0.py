
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf

def tf_IndexedSlices_inputs():
    list_of_inputs = []
    
    # Input 1: Basic case with 2D values and 1D indices
    values = np.array([[1., 2., 3.], [4., 5., 6.]])
    indices = np.array([0, 1])
    dense_shape = np.array([3, 3])
    
    input_dict = {
        "values": values,
        "indices": indices,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 2: 3D values with 1D indices
    values = np.array([[[1., 2], [3., 4]], [[5., 6], [7., 8]]])
    indices = np.array([0, 1])
    dense_shape = np.array([2, 2, 2])
    
    input_dict = {
        "values": values,
        "indices": indices,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 3: Negative values in values
    values = np.array([[-1., -2.], [-3., -4.]])
    indices = np.array([0, 1])
    dense_shape = np.array([2, 2])
    
    input_dict = {
        "values": values,
        "indices": indices,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 4: Float values with different shape
    values = np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]])
    indices = np.array([0, 1, 2])
    dense_shape = np.array([3, 2])
    
    input_dict = {
        "values": values,
        "indices": indices,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 5: Mixed integer and float values
    values = np.array([[1, 2.5], [3, 4.5]])
    indices = np.array([0, 1])
    dense_shape = np.array([2, 2])
    
    input_dict = {
        "values": values,
        "indices": indices,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 6: 1D values with 1D indices and 1D dense_shape
    values = np.array([1, 2, 3])
    indices = np.array([0, 1, 2])
    dense_shape = np.array([3])
    
    input_dict = {
        "values": values,
        "indices": indices,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 7: Large number of indices
    values = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]])
    indices = np.array([0, 1, 2])
    dense_shape = np.array([3, 4])
    
    input_dict = {
        "values": values,
        "indices": indices,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 8: 4D values with 1D indices
    values = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]])
    indices = np.array([0, 1])
    dense_shape = np.array([3, 4, 2, 2])
    
    input_dict = {
        "values": values,
        "indices": indices,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 9: Single element values and indices
    values = np.array([[[1.5]])
    indices = np.array([0])
    dense_shape = np.array([1, 1, 1])
    
    input_dict = {
        "values": values,
        "indices": indices,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 10: Zero values with different shape
    values = np.array([[0., 0.], [0., 0.]])
    indices = np.array([0, 1])
    dense_shape = np.array([2, 2])
    
    input_dict = {
        "values": values,
        "indices": indices,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs["tf.IndexedSlices"] = tf_IndexedSlices_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.IndexedSlices' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.IndexedSlices'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.IndexedSlices', generated_inputs['tf.IndexedSlices'], lib="tf", suffix=0)
