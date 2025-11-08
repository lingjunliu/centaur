
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def chain_matmul_inputs():
    class TensorList(list):
        def __init__(self, seq, dtype):
            super().__init__(seq)
            self.shape = (len(seq),)
            self.size = 0
            self.dtype = np.dtype(dtype)

    list_of_inputs = []

    A1 = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]], dtype=torch.float32).numpy()
    B1 = torch.tensor([[0.5, -1.0, 0.0, 2.0],
                       [1.5,  2.0, -0.5, -1.0],
                       [0.0,  3.0,  1.0,  0.5]], dtype=torch.float32).numpy()
    list_of_inputs.append({"matrices": TensorList([A1, B1], dtype=np.float32)})

    A2 = torch.randn((4, 5), dtype=torch.float64).numpy()
    B2 = torch.randn((5, 6), dtype=torch.float64).numpy()
    C2 = torch.randn((6, 2), dtype=torch.float64).numpy()
    list_of_inputs.append({"matrices": TensorList([A2, B2, C2], dtype=np.float64)})

    A3 = torch.randint(-5, 6, (2, 2), dtype=torch.int64).numpy()
    B3 = torch.randint(-5, 6, (2, 2), dtype=torch.int64).numpy()
    C3 = torch.randint(-5, 6, (2, 2), dtype=torch.int64).numpy()
    D3 = torch.randint(-5, 6, (2, 2), dtype=torch.int64).numpy()
    list_of_inputs.append({"matrices": TensorList([A3, B3, C3, D3], dtype=np.int64)})

    A4r = torch.tensor([[1.0, -2.0],
                        [0.5,  3.0]], dtype=torch.float32).numpy()
    A4i = torch.tensor([[0.0,  1.0],
                        [-1.5, 2.0]], dtype=torch.float32).numpy()
    A4 = (A4r + 1j * A4i).astype(np.complex64)
    B4r = torch.tensor([[2.0,  0.0],
                        [-3.0, 1.0]], dtype=torch.float32).numpy()
    B4i = torch.tensor([[1.0, -2.0],
                        [0.0,  0.5]], dtype=torch.float32).numpy()
    B4 = (B4r + 1j * B4i).astype(np.complex64)
    list_of_inputs.append({"matrices": TensorList([A4, B4], dtype=np.complex64)})

    A5 = torch.randn((6, 4), dtype=torch.float16).numpy()
    B5 = torch.randn((4, 3), dtype=torch.float16).numpy()
    C5 = torch.randn((3, 2), dtype=torch.float16).numpy()
    list_of_inputs.append({"matrices": TensorList([A5, B5, C5], dtype=np.float16)})

    I6 = torch.eye(3, dtype=torch.int64).numpy()
    M6 = torch.tensor([[2, -1, 0],
                       [0, 3, 1],
                       [-2, 4, -3]], dtype=torch.int64).numpy()
    v6 = torch.tensor([[1],
                       [0],
                       [-1]], dtype=torch.int64).numpy()
    list_of_inputs.append({"matrices": TensorList([I6, M6, v6], dtype=np.int64)})

    mats7 = [torch.randn((2, 2), dtype=torch.float32).numpy() for _ in range(5)]
    list_of_inputs.append({"matrices": TensorList(mats7, dtype=np.float32)})

    A8 = (torch.tensor([[1.0, -2.0, 0.0]], dtype=torch.float64).numpy().astype(np.complex128)
          + 1j * torch.tensor([[0.5, 1.5, -1.0]], dtype=torch.float64).numpy().astype(np.complex128))
    B8 = (torch.tensor([[2.0, 0.0, -1.0],
                        [1.0, 1.0,  0.0],
                        [0.0, 3.0,  2.0]], dtype=torch.float64).numpy().astype(np.complex128)
          + 1j * torch.tensor([[0.0, -1.0, 2.0],
                                [2.0,  0.0, 1.0],
                                [-1.0, 0.5, 0.0]], dtype=torch.float64).numpy().astype(np.complex128))
    C8 = (torch.tensor([[1.0],
                        [-2.0],
                        [3.0]], dtype=torch.float64).numpy().astype(np.complex128)
          + 1j * torch.tensor([[0.0],
                                [1.0],
                                [-1.0]], dtype=torch.float64).numpy().astype(np.complex128))
    list_of_inputs.append({"matrices": TensorList([A8, B8, C8], dtype=np.complex128)})

    A9 = torch.tensor([[ -1.0,  2.0, -3.0,  4.0],
                       [  5.0, -6.0,  7.0, -8.0],
                       [ -9.0, 10.0, -11.0, 12.0]], dtype=torch.float32).numpy()
    B9 = torch.tensor([[ 1.0, -2.0,  3.0],
                       [-4.0,  5.0, -6.0],
                       [ 7.0, -8.0,  9.0],
                       [-10.0, 11.0, -12.0]], dtype=torch.float32).numpy()
    list_of_inputs.append({"matrices": TensorList([A9, B9], dtype=np.float32)})

    A10 = torch.randn((2, 7), dtype=torch.float64).numpy()
    B10 = torch.randn((7, 5), dtype=torch.float64).numpy()
    C10 = torch.randn((5, 1), dtype=torch.float64).numpy()
    list_of_inputs.append({"matrices": TensorList([A10, B10, C10], dtype=np.float64)})

    return list_of_inputs

generated_inputs["torch.chain_matmul"] = chain_matmul_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.chain_matmul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.chain_matmul'.")


check_valid('torch.chain_matmul', generated_inputs['torch.chain_matmul'], lib="torch", suffix=0)
