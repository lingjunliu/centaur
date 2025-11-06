
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def polar_inputs():
    list_of_inputs = []

    # Input 1: 1D float32
    abs_arr = np.array([0.0, 1.5, 2.2], dtype=np.float32)
    angle = np.array([0.0, 0.5, -1.0], dtype=np.float32)
    out = np.empty(abs_arr.shape, dtype=np.complex64)
    input_dict = {"abs": abs_arr, "angle": angle, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float64
    abs_arr = np.array([1.0, 2.0], dtype=np.float64)
    angle = np.array([np.pi / 2, 5 * np.pi / 4], dtype=np.float64)
    out = np.empty(abs_arr.shape, dtype=np.complex128)
    input_dict = {"abs": abs_arr, "angle": angle, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D with broadcasting (float32)
    abs_arr = np.array([[1.0, 0.0, 3.0],
                        [4.0, 5.0, 6.0]], dtype=np.float32)
    angle = np.array([[0.0, -np.pi/3, np.pi/2]], dtype=np.float32)
    out = np.empty((2, 3), dtype=np.complex64)
    input_dict = {"abs": abs_arr, "angle": angle, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D same shape (float64)
    abs_arr = np.array([[2.5, 1.0],
                        [3.5, 4.5]], dtype=np.float64)
    angle = np.array([[np.pi, -np.pi],
                      [np.pi/4, -np.pi/2]], dtype=np.float64)
    out = np.empty((2, 2), dtype=np.complex128)
    input_dict = {"abs": abs_arr, "angle": angle, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D broadcasting (float32)
    abs_arr = np.array([[[1.0, 2.0, 3.0]],
                        [[4.0, 5.0, 6.0]]], dtype=np.float32)  # (2,1,3)
    angle = np.array([[[0.0],
                       [np.pi/2],
                       [np.pi],
                       [-np.pi/2]]], dtype=np.float32)  # (1,4,1)
    out = np.empty((2, 4, 3), dtype=np.complex64)
    input_dict = {"abs": abs_arr, "angle": angle, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 0-D scalars (float64)
    abs_arr = np.array(3.0, dtype=np.float64)
    angle = np.array(-np.pi/3, dtype=np.float64)
    out = np.empty((), dtype=np.complex128)
    input_dict = {"abs": abs_arr, "angle": angle, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty 1D (float32)
    abs_arr = np.array([], dtype=np.float32)
    angle = np.array([], dtype=np.float32)
    out = np.empty((0,), dtype=np.complex64)
    input_dict = {"abs": abs_arr, "angle": angle, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D broadcasting (float64)
    abs_arr = np.array([0.1, 0.5, 1.0, 2.0, 10.0], dtype=np.float64)
    angle = np.array([np.pi/6], dtype=np.float64)
    out = np.empty((5,), dtype=np.complex128)
    input_dict = {"abs": abs_arr, "angle": angle, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D with broadcasting (float32)
    abs_arr = np.arange(1, 1 + 1*4*1*2, dtype=np.float32).reshape(1, 4, 1, 2)
    angle = np.array([[[0.0, -np.pi/4]]], dtype=np.float32)  # (1,1,2)
    out = np.empty((1, 4, 1, 2), dtype=np.complex64)
    input_dict = {"abs": abs_arr, "angle": angle, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty last dimension (float64)
    abs_arr = np.empty((2, 2, 0), dtype=np.float64)
    angle = np.empty((2, 2, 0), dtype=np.float64)
    out = np.empty((2, 2, 0), dtype=np.complex128)
    input_dict = {"abs": abs_arr, "angle": angle, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D broadcasting (float64)
    abs_arr = np.arange(1, 1 + 3*1*4, dtype=np.float64).reshape(3, 1, 4)
    angle = np.array([[[-np.pi/2],
                       [np.pi/3]]], dtype=np.float64)  # (1,2,1)
    out = np.empty((3, 2, 4), dtype=np.complex128)
    input_dict = {"abs": abs_arr, "angle": angle, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Row/column broadcasting (float32)
    abs_arr = np.array([[0.0],
                        [1.0],
                        [2.0]], dtype=np.float32)  # (3,1)
    angle = np.array([[0.0, np.pi/2, np.pi, -np.pi/2]], dtype=np.float32)  # (1,4)
    out = np.empty((3, 4), dtype=np.complex64)
    input_dict = {"abs": abs_arr, "angle": angle, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.polar"] = polar_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.polar' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.polar'.")


check_valid('torch.polar', generated_inputs['torch.polar'], lib="torch", suffix=0)
