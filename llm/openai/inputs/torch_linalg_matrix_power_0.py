
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def matrix_power_inputs():
    list_of_inputs = []

    A = torch.tensor([[2.0]], dtype=torch.float32).numpy()
    n = 3
    list_of_inputs.append(copy.deepcopy({"A": A, "n": n}))

    A = torch.tensor([[3.0, 0.0], [0.0, 4.0]], dtype=torch.float64).numpy()
    n = 0
    list_of_inputs.append(copy.deepcopy({"A": A, "n": n}))

    theta = np.pi / 4
    A = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]], dtype=np.float64)
    n = -1
    list_of_inputs.append(copy.deepcopy({"A": A, "n": n}))

    A = (0.5 * np.eye(3, dtype=np.float32))
    n = 5
    list_of_inputs.append(copy.deepcopy({"A": A, "n": n}))

    A = torch.stack([
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        torch.tensor([[-1.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    ], dim=0).numpy()
    n = 2
    list_of_inputs.append(copy.deepcopy({"A": A, "n": n}))

    A = torch.tensor([[1+1j, 2-1j],
                      [0.5+0.3j, 3-2j]], dtype=torch.complex64).numpy()
    n = 3
    list_of_inputs.append(copy.deepcopy({"A": A, "n": n}))

    A = torch.diag(torch.tensor([2+1j, -1+0.5j, 0.25-0.25j], dtype=torch.complex128)).numpy()
    n = -2
    list_of_inputs.append(copy.deepcopy({"A": A, "n": n}))

    A = torch.tensor([
        [1.0, 1.0, 0.5, -0.2],
        [0.0, 1.0, 1.5, 0.3],
        [0.0, 0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=torch.float64).numpy()
    n = 3
    list_of_inputs.append(copy.deepcopy({"A": A, "n": n}))

    A = np.stack([
        np.eye(3, dtype=np.float64),
        np.diag(np.array([2.0, -1.0, 0.5], dtype=np.float64)),
        np.array([[1.0, -1.0, 0.0],
                  [2.0,  0.0, 0.0],
                  [0.0,  0.0, 1.0]], dtype=np.float64)
    ], axis=0)
    n = 2
    list_of_inputs.append(copy.deepcopy({"A": A, "n": n}))

    A = torch.tensor([[-1.0 + 2.0j]], dtype=torch.complex64).numpy()
    n = -5
    list_of_inputs.append(copy.deepcopy({"A": A, "n": n}))

    scalars = np.array([[1.0, 0.5, -2.0],
                        [3.0, -1.5, 0.25]], dtype=np.float32)
    A = scalars[..., None, None] * np.eye(2, dtype=np.float32)
    n = 4
    list_of_inputs.append(copy.deepcopy({"A": A, "n": n}))

    A = np.diag(np.array([1.5, -2.0, 0.7, 3.3, -0.9], dtype=np.float64))
    n = 2
    list_of_inputs.append(copy.deepcopy({"A": A, "n": n}))

    return list_of_inputs

generated_inputs["torch.linalg.matrix_power"] = matrix_power_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.linalg.matrix_power' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.matrix_power'.")


check_valid('torch.linalg.matrix_power', generated_inputs['torch.linalg.matrix_power'], lib="torch", suffix=0)
