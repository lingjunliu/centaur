
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_complex_inputs():
    list_of_inputs = []

    real1 = torch.tensor([1.0, 2.0], dtype=torch.float32).numpy()
    imag1 = torch.tensor([3.0, 4.0], dtype=torch.float32).numpy()
    out1 = np.empty((2,), dtype=np.complex64)
    input_dict1 = {"real": real1, "imag": imag1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    real2 = torch.tensor([-1.0, 2.0], dtype=torch.float64).numpy()
    imag2 = torch.tensor([3.0, -4.0], dtype=torch.float64).numpy()
    out2 = np.empty((2,), dtype=np.complex128)
    input_dict2 = {"real": real2, "imag": imag2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    real3 = torch.randn(3, 2, dtype=torch.float32).numpy()
    imag3 = torch.randn(3, 2, dtype=torch.float32).numpy()
    out3 = np.empty((3, 2), dtype=np.complex64)
    input_dict3 = {"real": real3, "imag": imag3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    real4 = torch.tensor([0.0], dtype=torch.float32).numpy()
    imag4 = torch.tensor([0.0], dtype=torch.float32).numpy()
    out4 = np.empty((1,), dtype=np.complex64)
    input_dict4 = {"real": real4, "imag": imag4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    real5 = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32).numpy()
    imag5 = torch.tensor([-0.5, -1.5, -2.5], dtype=torch.float32).numpy()
    out5 = np.empty((3,), dtype=np.complex64)
    input_dict5 = {"real": real5, "imag": imag5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    real6 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64).numpy()
    imag6 = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64).numpy()
    out6 = np.empty((3,), dtype=np.complex128)
    input_dict6 = {"real": real6, "imag": imag6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    real7 = torch.zeros(2, 2, dtype=torch.float32).numpy()
    imag7 = torch.ones(2, 2, dtype=torch.float32).numpy()
    out7 = np.empty((2, 2), dtype=np.complex64)
    input_dict7 = {"real": real7, "imag": imag7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    real8 = torch.arange(1, 5, dtype=torch.float32).numpy()
    imag8 = torch.arange(5, 9, dtype=torch.float32).numpy()
    out8 = np.empty((4,), dtype=np.complex64)
    input_dict8 = {"real": real8, "imag": imag8, "out": out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    real9 = torch.tensor([1e-6, 2e-6], dtype=torch.float32).numpy()
    imag9 = torch.tensor([3e-6, 4e-6], dtype=torch.float32).numpy()
    out9 = np.empty((2,), dtype=np.complex64)
    input_dict9 = {"real": real9, "imag": imag9, "out": out9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    real10 = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64).numpy()
    imag10 = torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float64).numpy()
    out10 = np.empty((4,), dtype=np.complex128)
    input_dict10 = {"real": real10, "imag": imag10, "out": out10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.complex"] = torch_complex_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.complex' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.complex'.")


check_valid('torch.complex', generated_inputs['torch.complex'], lib="torch", suffix=0)
