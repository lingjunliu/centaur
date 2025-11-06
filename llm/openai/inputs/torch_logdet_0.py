
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def logdet_inputs():
    list_of_inputs = []

    # 1: 1x1 positive
    input = np.array([[2.5]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 2: 1x1 negative
    input = np.array([[-3.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 3: 2x2 invertible, positive det
    input = np.array([[4.0, 2.0],
                      [1.0, 3.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 4: 2x2 invertible, negative det
    input = np.array([[0.0, 1.0],
                      [1.0, 0.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 5: 2x2 singular
    input = np.array([[1.0, 2.0],
                      [2.0, 4.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 6: 3x3 batch of SPD matrices
    A0 = np.array([[1.0, 2.0, -1.0],
                   [0.0, 3.0, 1.0],
                   [2.0, -2.0, 1.0]], dtype=np.float64)
    A1 = np.array([[2.0, -1.0, 0.5],
                   [1.0, 0.0, 4.0],
                   [-3.0, 1.0, 2.0]], dtype=np.float64)
    A2 = np.array([[0.5, 1.0, 1.5],
                   [1.5, -2.0, 0.0],
                   [2.0, 2.0, 3.0]], dtype=np.float64)
    I3 = np.eye(3, dtype=np.float64) * 0.1
    SPD_batch = np.stack([A0 @ A0.T + I3, A1 @ A1.T + I3, A2 @ A2.T + I3], axis=0)
    input = SPD_batch
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 7: 3x3 singular with zero row
    input = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [0.0, 0.0, 0.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 8: batch (2,1,1) with positive and negative
    input = np.array([[[3.0]],
                      [[-1.0]]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 9: 3x3 diagonal with large values
    input = np.diag(np.array([1e20, 1e10, 1e5], dtype=np.float64))
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 10: 2x2 near-singular matrix
    input = np.array([[1.0, 1.0],
                      [1.0, 1.0 + 1e-12]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 11: batched (4,2,2) various matrices
    input = np.array([
        [[2.0, 1.0],
         [0.0, 3.0]],
        [[-1.0, 2.0],
         [3.0, -2.0]],
        [[1.0, 4.0],
         [2.0, 9.0]],
        [[5.0, 7.0],
         [6.0, 8.0]]
    ], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 12: multi-batch (2,3,2,2)
    input = np.array([
        [[[1.0, 2.0],
          [3.0, 5.0]],
         [[2.0, 1.0],
          [0.5, 4.0]],
         [[-1.0, 2.0],
          [3.0, -2.0]]],
        [[[1.5, -0.5],
          [2.0, 3.0]],
         [[0.1, 5.0],
          [6.0, 0.2]],
         [[7.0, 8.0],
          [9.0, 10.0]]]
    ], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    return list_of_inputs

generated_inputs["torch.logdet"] = logdet_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.logdet' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logdet'.")


check_valid('torch.logdet', generated_inputs['torch.logdet'], lib="torch", suffix=0)
