
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def tensorinv_inputs():
    list_of_inputs = []

    # Input 1: 2D float32, invertible with negatives
    A = torch.tensor([[2.0, -1.0, 0.0],
                      [0.0, 3.0, 1.0],
                      [1.0, 0.0, 1.0]], dtype=torch.float32).numpy()
    ind = 1
    out_shape = A.shape[ind:] + A.shape[:ind]
    out = np.zeros(out_shape, dtype=A.dtype)
    list_of_inputs.append(copy.deepcopy({"A": A, "ind": ind, "out": out}))

    # Input 2: 2D float64 identity
    A = torch.eye(4, dtype=torch.float64).numpy()
    ind = np.int64(1)
    out_shape = A.shape[ind:] + A.shape[:ind]
    out = np.zeros(out_shape, dtype=A.dtype)
    list_of_inputs.append(copy.deepcopy({"A": A, "ind": ind, "out": out}))

    # Input 3: 4D float32 from identity, shape (4,6,8,3), ind=2
    A = torch.eye(4 * 6, dtype=torch.float32).reshape(4, 6, 8, 3).numpy()
    ind = 2
    out_shape = A.shape[ind:] + A.shape[:ind]
    out = np.zeros(out_shape, dtype=A.dtype)
    list_of_inputs.append(copy.deepcopy({"A": A, "ind": ind, "out": out}))

    # Input 4: 4D float64, shape (2,3,3,2), ind=2
    A = torch.eye(2 * 3, dtype=torch.float64).reshape(2, 3, 3, 2).numpy()
    ind = np.int64(2)
    out_shape = A.shape[ind:] + A.shape[:ind]
    out = np.zeros(out_shape, dtype=A.dtype)
    list_of_inputs.append(copy.deepcopy({"A": A, "ind": ind, "out": out}))

    # Input 5: 4D complex64, shape (2,2,2,2), ind=2
    A = torch.eye(2 * 2, dtype=torch.complex64).reshape(2, 2, 2, 2).numpy()
    ind = 2
    out_shape = A.shape[ind:] + A.shape[:ind]
    out = np.zeros(out_shape, dtype=A.dtype)
    list_of_inputs.append(copy.deepcopy({"A": A, "ind": ind, "out": out}))

    # Input 6: 6D float32, shape (2,1,3,3,1,2), ind=3
    A = torch.eye(2 * 1 * 3, dtype=torch.float32).reshape(2, 1, 3, 3, 1, 2).numpy()
    ind = 3
    out_shape = A.shape[ind:] + A.shape[:ind]
    out = np.zeros(out_shape, dtype=A.dtype)
    list_of_inputs.append(copy.deepcopy({"A": A, "ind": ind, "out": out}))

    # Input 7: 2D float64 negative-scaled identity
    A = (-2.0 * torch.eye(5, dtype=torch.float64)).numpy()
    ind = np.int64(1)
    out_shape = A.shape[ind:] + A.shape[:ind]
    out = np.zeros(out_shape, dtype=A.dtype)
    list_of_inputs.append(copy.deepcopy({"A": A, "ind": ind, "out": out}))

    # Input 8: 2D complex128 near-identity
    re = torch.randn(4, 4, dtype=torch.float64)
    im = torch.randn(4, 4, dtype=torch.float64)
    noise = torch.complex(re, im)
    A = (torch.eye(4, dtype=torch.complex128) * (1.0 + 1.0j) + 0.01 * noise).numpy()
    ind = 1
    out_shape = A.shape[ind:] + A.shape[:ind]
    out = np.zeros(out_shape, dtype=A.dtype)
    list_of_inputs.append(copy.deepcopy({"A": A, "ind": ind, "out": out}))

    # Input 9: 6D float32, shape (2,2,3,3,2,2), ind=3
    A = torch.eye(2 * 2 * 3, dtype=torch.float32).reshape(2, 2, 3, 3, 2, 2).numpy()
    ind = np.int64(3)
    out_shape = A.shape[ind:] + A.shape[:ind]
    out = np.zeros(out_shape, dtype=A.dtype)
    list_of_inputs.append(copy.deepcopy({"A": A, "ind": ind, "out": out}))

    # Input 10: 3D float64, shape (6,2,3), ind=1
    A = torch.eye(6, dtype=torch.float64).reshape(6, 2, 3).numpy()
    ind = 1
    out_shape = A.shape[ind:] + A.shape[:ind]
    out = np.zeros(out_shape, dtype=A.dtype)
    list_of_inputs.append(copy.deepcopy({"A": A, "ind": ind, "out": out}))

    # Input 11: 8D float64, shape (2,2,3,1,1,3,2,2), ind=4
    A = torch.eye(2 * 2 * 3 * 1, dtype=torch.float64).reshape(2, 2, 3, 1, 1, 3, 2, 2).numpy()
    ind = np.int64(4)
    out_shape = A.shape[ind:] + A.shape[:ind]
    out = np.zeros(out_shape, dtype=A.dtype)
    list_of_inputs.append(copy.deepcopy({"A": A, "ind": ind, "out": out}))

    # Input 12: 6D complex64, shape (3,1,2,1,3,2), ind=3
    A = torch.eye(3 * 1 * 2, dtype=torch.complex64).reshape(3, 1, 2, 1, 3, 2).numpy()
    ind = 3
    out_shape = A.shape[ind:] + A.shape[:ind]
    out = np.zeros(out_shape, dtype=A.dtype)
    list_of_inputs.append(copy.deepcopy({"A": A, "ind": ind, "out": out}))

    return list_of_inputs

generated_inputs["torch.linalg.tensorinv"] = tensorinv_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.linalg.tensorinv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.tensorinv'.")


check_valid('torch.linalg.tensorinv', generated_inputs['torch.linalg.tensorinv'], lib="torch", suffix=0)
