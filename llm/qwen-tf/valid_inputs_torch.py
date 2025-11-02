generated_inputs = {}

import torch, copy
import numpy as np

def acos__inputs():
    list_of_inputs = []

    input = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = (torch.rand((2, 3), dtype=torch.float64) * 2 - 1).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor(0.3, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[[-1.0, -0.25], [0.25, 1.0]]], dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.linspace(-1, 1, num=10, dtype=np.float32)[::2]
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.asfortranarray(np.array([[-0.8, -0.3, 0.0], [0.2, 0.7, 1.0]], dtype=np.float64))
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([1 + 1j, -0.2 + 0.3j, 0.5 - 0.5j], dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[0.1 + 2j, -1.0 + 0.0j], [0.0 - 1.5j, 0.8 + 0.9j]], dtype=torch.complex128).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = (torch.rand((2, 1, 2, 3), dtype=torch.float32) * 2 - 1).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.array([-1.5, -1.0001, 1.0001, 2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.array([np.nan, np.inf, -np.inf, 0.5, -0.5], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    base = np.linspace(-1, 1, num=7, dtype=np.float64)
    input = base[::-1]
    list_of_inputs.append(copy.deepcopy({"input": input}))

    return list_of_inputs

generated_inputs["torch.acos_"] = acos__inputs()

def arccosh_inputs():
    list_of_inputs = []

    # 1: 0-D float32 scalar
    input = torch.tensor(1.0, dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 2: 1-D float64 vector
    input = torch.tensor([1.0, 1.0001, 2.0, 3.5], dtype=torch.float64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 3: 2-D float16 matrix
    input = torch.tensor([[1.0, 2.0], [10.0, 100.0]], dtype=torch.float16).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 4: 3-D float32 tensor
    input = torch.tensor([[[1.0, 1.5, 2.0]], [[3.0, 4.0, 5.0]]], dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 5: 1-D complex64 vector
    input = torch.tensor([1+0j, 0.5+0.5j, -2+3j, 4-1j], dtype=torch.complex64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 6: 2-D complex128 matrix
    input = torch.tensor([[1+0j, -1+2j], [3-4j, 0.1+0.2j]], dtype=torch.complex128).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 7: Empty 1-D float32
    input = torch.empty((0,), dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 8: Empty 3-D float64
    input = torch.empty((2, 0, 3), dtype=torch.float64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 9: Strided 1-D float32 (slice)
    base = torch.linspace(1, 10, steps=10, dtype=torch.float32).numpy()
    input = base[::2]
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 10: Includes inf and nan, float64
    input = torch.tensor([float('inf'), 1.0, float('nan'), 2.0], dtype=torch.float64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 11: Fortran-ordered 2-D float32
    arr = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3).numpy()
    input = np.asfortranarray(arr)
    out = np.empty_like(input, order='F')
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 12: 4-D float32 with varied magnitudes
    vals = torch.tensor([1.0, 1.5, 100.0, 1e10, 2.0, 1.000001, 3.0, 5.0, 10.0, 1.0001, 7.5, 2.5], dtype=torch.float32).numpy()
    input = vals.reshape(2, 2, 1, 3)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 13: Real values including <1 to produce NaNs, float32
    input = torch.tensor([0.5, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.arccosh"] = arccosh_inputs()



def floor__inputs():
    list_of_inputs = []

    input = torch.tensor([1.2, -3.7, 0.0, 5.999, -0.0001], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[1.5, 2.0, -2.1],
                          [3.7, -4.2, 0.9]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor(-123.456, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.randn(2, 3, 4, dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.empty((0,), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([-10, 0, 1, 2, 255], dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[0, 255],
                          [128, 64]], dtype=torch.uint8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.arange(-6, 6, dtype=torch.int32).reshape(2, 2, 3).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([float('nan'), float('inf'), float('-inf'), -1.001, 2.999], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.linspace(-5, 5, steps=24, dtype=torch.float32).reshape(2, 1, 3, 4).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.empty((0, 3), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([1e-7, -1e-7, 3.0000001, -3.0000001], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[1.9, -2.9],
                          [3.2, -4.8]], dtype=torch.float32).T.numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    return list_of_inputs

generated_inputs["torch.floor_"] = floor__inputs()

def get_rng_state_inputs():
    list_of_inputs = []
    input_dict = {}
    for _ in range(12):
        list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs["torch.get_rng_state"] = get_rng_state_inputs()

def isposinf_inputs():
    list_of_inputs = []

    # Input 1
    input = np.array([-np.inf, np.inf, 0.0], dtype=np.float32)
    out = np.empty(input.shape, dtype=np.bool_)
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input = np.array([[np.inf, -np.inf, np.nan],
                      [1.0, 2.5, 3.0]], dtype=np.float64)
    out = np.empty(input.shape, dtype=np.bool_)
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input = np.array([[[np.inf, 0.0, -1.0, 5.5]],
                      [[-np.inf, np.inf, 2.0, -3.0]]], dtype=np.float16)
    out = np.empty(input.shape, dtype=np.bool_)
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4 (scalar)
    input = np.array(np.inf, dtype=np.float64)
    out = np.empty((), dtype=np.bool_)
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 (empty 1D)
    input = np.array([], dtype=np.float32)
    out = np.empty(input.shape, dtype=np.bool_)
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 (non-contiguous via transpose)
    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    base[0, 0] = np.inf
    base[1, 2] = np.inf
    input = base.T
    out = np.empty(input.shape, dtype=np.bool_)
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (Fortran-ordered)
    input = np.asfortranarray(np.array([[0.0, np.inf, -np.inf],
                                        [np.nan, 7.0, 8.0]], dtype=np.float64))
    out = np.empty(input.shape, dtype=np.bool_)
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (large finite and infinities)
    input = np.array([3.4e38, -3.4e38, np.inf, -np.inf, 1.0], dtype=np.float32)
    out = np.empty(input.shape, dtype=np.bool_)
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (zeros, nans, inf)
    input = np.array([[0.0, -0.0, np.nan],
                      [np.inf, 1e-45, -1e-45]], dtype=np.float32)
    out = np.empty(input.shape, dtype=np.bool_)
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (3D float64)
    input = np.array([[[1.0, np.inf],
                       [-np.inf, 0.0]],
                      [[np.nan, 5.0],
                       [np.inf, -10.0]]], dtype=np.float64)
    out = np.empty(input.shape, dtype=np.bool_)
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 (float16 finite only)
    input = np.array([1.0, -2.0, 3.0, 0.0], dtype=np.float16)
    out = np.empty(input.shape, dtype=np.bool_)
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12 (4D with a single +inf)
    input = np.zeros((1, 2, 1, 3), dtype=np.float32)
    input[0, 1, 0, 2] = np.inf
    out = np.empty(input.shape, dtype=np.bool_)
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.isposinf"] = isposinf_inputs()

def multiply_inputs():
    list_of_inputs = []
    
    # 1
    input = torch.tensor([1.0, -2.0, 3.5, 0.0, -1.25], dtype=torch.float32).numpy()
    other = torch.tensor([0.5, 2.0, -4.0, 1.0, 3.0], dtype=torch.float32).numpy()
    out = torch.empty((5,), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # 2
    input = torch.tensor([[-1, 2, -3], [4, -5, 6]], dtype=torch.int64).numpy()
    other = torch.tensor([[7, -8, 9], [-10, 11, -12]], dtype=torch.int64).numpy()
    out = torch.empty((2, 3), dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # 3
    input = torch.tensor([[[1.0, -1.0], [2.0, -2.0]],
                          [[0.5, -0.5], [1.5, -1.5]]], dtype=torch.float16).numpy()
    other = torch.tensor([[[2.0, 3.0], [-1.0, 0.5]],
                          [[4.0, -2.0], [0.0, 1.0]]], dtype=torch.float16).numpy()
    out = torch.empty((2, 2, 2), dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # 4
    input = torch.tensor(3, dtype=torch.int32).numpy()
    other = torch.tensor(-2, dtype=torch.int32).numpy()
    out = torch.empty((), dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # 5
    input = torch.empty((0,), dtype=torch.float64).numpy()
    other = torch.empty((0,), dtype=torch.float64).numpy()
    out = torch.empty((0,), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # 6
    input = torch.tensor([[1.0], [-2.0], [3.5]], dtype=torch.float32).numpy()
    other = torch.tensor([[0.5, -1.0, 2.0, -3.0]], dtype=torch.float32).numpy()
    out = torch.empty((3, 4), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # 7
    input = torch.tensor([[[1], [-2]]], dtype=torch.int16).numpy()          # (1,2,1)
    other = torch.tensor([[[3, -1, 2]],
                          [[-4, 5, 0]],
                          [[1, 1, 1]],
                          [[2, -2, -2]]], dtype=torch.int16).numpy()        # (4,1,3)
    out = torch.empty((4, 2, 3), dtype=torch.int16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # 8
    input = torch.tensor([[1+2j, -3+0.5j, 2-1j],
                          [-0.5-0.5j, 0+1j, -1-2j]], dtype=torch.complex64).numpy()
    other = torch.tensor([[2-1j, -1+3j, 0.5+0.5j],
                          [-2-2j, 1-1j, 3+0j]], dtype=torch.complex64).numpy()
    out = torch.empty((2, 3), dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # 9
    input = torch.randn((1, 2, 3, 4), dtype=torch.float32).numpy()
    other = torch.randn((1, 2, 3, 4), dtype=torch.float32).numpy()
    out = torch.empty((1, 2, 3, 4), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    # 10
    input = torch.tensor([[1, 255], [128, 64]], dtype=torch.uint8).numpy()
    other = torch.tensor([[2, 2], [3, 4]], dtype=torch.uint8).numpy()
    out = torch.empty((2, 2), dtype=torch.uint8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    return list_of_inputs

generated_inputs["torch.multiply_1"] = multiply_inputs()

def torch_multiply_2_inputs():
    list_of_inputs = []

    # 1
    input = np.array([1.0, -2.5, 3.0], dtype=np.float32)
    other = 2.0
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 2
    input = np.array(5.0, dtype=np.float64)
    other = -1.5
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 3
    input = np.arange(6, dtype=np.int32).reshape(2, 3)
    other = np.float32(0.5)
    out = np.empty(input.shape, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 4
    input = (np.ones((2, 2, 3), dtype=np.float16) * -3)
    other = 4.0
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 5
    input = np.array([], dtype=np.float32)
    other = 3.14
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 6
    input = np.empty((2, 0, 3), dtype=np.float64)
    other = -2.0
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 7
    input = np.array([10, 20, 30, 255], dtype=np.uint8)
    other = 10.0
    out = np.empty(input.shape, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 8
    input = np.array([1+2j, -3+0.5j, 0-1j], dtype=np.complex64)
    other = 1.25
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 9
    input = np.asfortranarray(np.arange(9, dtype=np.float64).reshape(3, 3))
    other = np.float64(0.0)
    out = np.empty_like(input, order='F')
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 10
    base = np.arange(24, dtype=np.float32).reshape(4, 6)
    input = base[:, ::2]
    other = -0.75
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 11
    input = np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float64)
    other = 2.5
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 12
    input = np.array([10**12, -10**12, 42], dtype=np.int64)
    other = 1e-3
    out = np.empty(input.shape, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    return list_of_inputs

generated_inputs["torch.multiply_2"] = torch_multiply_2_inputs()



def multiply_inputs():
    list_of_inputs = []

    # Input 1
    input = torch.tensor([1.0, -2.5, 3.0], dtype=torch.float32).numpy()
    other = np.int64(2)
    out = np.empty_like(input)
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input = torch.tensor([[1.0, 2.0], [-3.5, 4.0]], dtype=torch.float64).numpy()
    other = np.int32(-3)
    out = np.zeros_like(input)
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input = torch.tensor(
        [[[1, -2, 3], [4, 5, -6]],
         [[-7, 8, 9], [10, -11, 12]]],
        dtype=torch.int16
    ).numpy()
    other = np.int8(5)
    out = np.empty_like(input)
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input = torch.arange(24, dtype=torch.int64).reshape(1, 2, 3, 4).numpy()
    other = np.int64(0)
    out = np.empty_like(input)
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input = torch.tensor(3.14, dtype=torch.float32).numpy()
    other = np.int16(4)
    out = np.empty_like(input)
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input = torch.empty((0, 5), dtype=torch.float64).numpy()
    other = np.int32(7)
    out = np.zeros_like(input)
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    base = torch.arange(10, dtype=torch.float32).numpy()
    input = base[::2]
    other = np.int8(-1)
    out = np.empty_like(input)
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input = torch.tensor([[0, 128, 255], [10, 20, 30]], dtype=torch.uint8).numpy()
    other = np.int8(2)
    out = np.empty_like(input)
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input = torch.tensor([1 + 2j, -3 + 4j, 0 - 1j], dtype=torch.complex64).numpy()
    other = np.int16(3)
    out = np.empty_like(input)
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input = torch.randn((1, 2, 1, 3, 2), dtype=torch.float16).numpy()
    other = np.int64(11)
    out = np.zeros_like(input)
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input = torch.linspace(-5, 5, steps=6, dtype=torch.float32).reshape(2, 3).numpy()
    other = np.int32(-7)
    out = np.empty_like(input)
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input = torch.tensor([[[1, -1], [2, -2]], [[3, -3], [4, -4]]], dtype=torch.int32).numpy()
    other = np.int64(9)
    out = np.zeros_like(input)
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.multiply_3"] = multiply_inputs()

def nansum_inputs():
    list_of_inputs = []

    input = np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float64)
    dtype = None
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype}))

    input = np.array([[1.0, -2.5, np.nan],
                      [3.0, np.nan, 6.5]], dtype=np.float32)
    dtype = torch.float64
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype}))

    input = np.array([[[1.0, np.nan], [-np.inf, 2.0]],
                      [[np.nan, 3.0], [4.0, 5.0]]], dtype=np.float16)
    dtype = torch.float32
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype}))

    input = np.array(np.nan, dtype=np.float32)
    dtype = None
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype}))

    input = np.array([], dtype=np.float64)
    dtype = None
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype}))

    input = np.empty((2, 0, 3), dtype=np.float32)
    dtype = None
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype}))

    input = np.array([-5, 0, 7, -3], dtype=np.int32)
    dtype = torch.int64
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype}))

    input = np.array([[255, 0, 1],
                      [2, 3, 4]], dtype=np.uint8)
    dtype = torch.int64
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype}))

    input = np.array([[[[1.0, np.nan, -1.0],
                        [2.0, 3.0, np.nan]]],
                      [[[np.nan, 0.0, 5.0],
                        [6.0, -7.5, 8.0]]]], dtype=np.float64)
    dtype = None
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype}))

    input = np.array([1e20, 1e20, np.nan, -1e20], dtype=np.float32)
    dtype = torch.float64
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype}))

    input = np.array([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=np.float64)
    dtype = None
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype}))

    arr = np.arange(24, dtype=np.float64).reshape(2, 3, 4).swapaxes(1, 2)
    arr[0, 1, 2] = np.nan
    input = arr
    dtype = None
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype}))

    return list_of_inputs

generated_inputs["torch.nansum_1"] = nansum_inputs()



def nansum_inputs():
    list_of_inputs = []

    input_arr = torch.tensor([1.0, 2.0, float('nan'), 4.0]).numpy()
    input_dict = {"input": input_arr, "dim": 0, "keepdim": False, "dtype": torch.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[1.0, float('nan')], [3.0, 4.0]]).numpy()
    input_dict = {"input": input_arr, "dim": 1, "keepdim": True, "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[1, 2, -3], [4, -5, 6]], dtype=torch.int64).numpy()
    input_dict = {"input": input_arr, "dim": 0, "keepdim": False, "dtype": torch.int64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([
        [[1.0, float('nan')], [-2.0, -3.0]],
        [[float('nan'), 5.0], [6.0, float('nan')]]
    ]).numpy()
    input_dict = {"input": input_arr, "dim": -1, "keepdim": False, "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor(
        [[[[1.0, float('nan')], [float('inf'), -1.0], [2.0, 3.0]],
          [[-float('inf'), 0.0], [4.0, float('nan')], [-2.0, 5.0]]],
         [[[float('nan'), -3.0], [7.0, 8.0], [float('nan'), float('inf')]],
          [[9.0, -10.0], [float('-inf'), float('nan')], [0.0, 1.0]]]]
    ).numpy()
    input_dict = {"input": input_arr, "dim": 2, "keepdim": True, "dtype": torch.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.empty(0, dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "dim": 0, "keepdim": False, "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([
        [[1, -2, 3, -4],
         [5, -6, 7, -8],
         [9, -10, 11, -12]],
        [[-1, 2, -3, 4],
         [-5, 6, -7, 8],
         [-9, 10, -11, 12]]
    ], dtype=torch.int32).numpy()
    input_dict = {"input": input_arr, "dim": 1, "keepdim": True, "dtype": torch.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[True, False, True],
                              [False, False, True]], dtype=torch.bool).numpy()
    input_dict = {"input": input_arr, "dim": -2, "keepdim": True, "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.randn(2, 1, 3, 1, 4, dtype=torch.float32)
    t[0, 0, 1, 0, 2] = float('nan')
    t[1, 0, 2, 0, 3] = float('nan')
    input_arr = t.numpy()
    input_dict = {"input": input_arr, "dim": 3, "keepdim": False, "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    u = torch.tensor(np.arange(2*3*1*2, dtype=np.float32).reshape(2, 3, 1, 2))
    u[0, 2, 0, 1] = float('nan')
    input_arr = u.numpy()
    input_dict = {"input": input_arr, "dim": -3, "keepdim": True, "dtype": torch.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([1e4, -2e4, float('nan'), 3e4, -4e4], dtype=torch.float16).numpy()
    input_dict = {"input": input_arr, "dim": 0, "keepdim": True, "dtype": torch.float16}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[-30000, 20000, -10000, 5000],
                              [4000, -3000, 2000, -1000]], dtype=torch.int16).numpy()
    input_dict = {"input": input_arr, "dim": 0, "keepdim": False, "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nansum_2"] = nansum_inputs()



def nansum_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = torch.tensor([1.0, float('nan'), -3.5, 2.5], dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "dim": (0,),
        "keepdim": False,
        "dtype": torch.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = torch.tensor([[1.0, 2.0], [float('nan'), 4.0]], dtype=torch.float64).numpy()
    input_dict = {
        "input": input_arr,
        "dim": (0,),
        "keepdim": True,
        "dtype": torch.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = torch.tensor([[1, -2, 3], [4, 5, -6]], dtype=torch.int64).numpy()
    input_dict = {
        "input": input_arr,
        "dim": (1,),
        "keepdim": False,
        "dtype": torch.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_arr = torch.tensor([[[1.0, float('nan')], [3.0, 4.0]],
                              [[-1.0, 2.0], [float('nan'), 5.0]]], dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "dim": (1, 2),
        "keepdim": False,
        "dtype": torch.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_arr = torch.tensor([[[1.0, 2.0, float('nan'), -4.0],
                               [5.0, -6.0, 7.0, float('nan')],
                               [9.0, 10.0, -11.0, 12.0]],
                              [[float('nan'), -2.5, 3.0, 4.0],
                               [5.5, float('nan'), -7.5, 8.0],
                               [9.5, -10.0, 11.0, float('nan')]]], dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "dim": (-1,),
        "keepdim": True,
        "dtype": torch.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_arr = torch.tensor([[[[1.0, float('nan'), -3.0],
                                [4.0, 5.0, float('nan')]],
                               [[-1.0, 2.0, 3.0],
                                [float('nan'), -5.0, 6.0]]],
                              [[[7.0, 8.0, float('nan')],
                                [-9.0, 10.0, 11.0]],
                               [[float('nan'), -12.0, 13.0],
                                [14.0, 15.0, float('nan')]]],
                              [[[1.5, -2.5, 3.5],
                                [float('nan'), -4.5, 5.5]],
                               [[6.5, float('nan'), -7.5],
                                [8.5, 9.5, 10.5]]],
                              [[[float('nan'), 0.0, -1.0],
                                [2.0, 3.0, 4.0]],
                               [[-5.0, float('nan'), 6.0],
                                [7.0, 8.0, 9.0]]]], dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "dim": (0, 3),
        "keepdim": False,
        "dtype": torch.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_arr = torch.tensor([[1.0, -2.0, float('nan')],
                              [4.5, float('nan'), -6.5],
                              [7.0, 8.0, 9.0],
                              [float('nan'), -10.0, 11.0],
                              [12.0, -13.0, 14.0]], dtype=torch.float64).numpy()
    input_dict = {
        "input": input_arr,
        "dim": (0, 1),
        "keepdim": True,
        "dtype": torch.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_arr = torch.tensor([[[-1, 2],
                               [3, -4],
                               [5, -6],
                               [7, 8]],
                              [[-9, 10],
                               [11, -12],
                               [13, -14],
                               [15, 16]],
                              [[-17, 18],
                               [19, -20],
                               [21, -22],
                               [23, 24]]], dtype=torch.int16).numpy()
    input_dict = {
        "input": input_arr,
        "dim": (0,),
        "keepdim": False,
        "dtype": torch.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_arr = torch.tensor([[[[1.0, float('nan')], [3.0, 4.0]],
                               [[-5.0, 6.0], [float('nan'), 8.0]]],
                              [[[9.0, -10.0], [11.0, float('nan')]],
                               [[13.0, 14.0], [-15.0, 16.0]]]], dtype=torch.float16).numpy()
    input_dict = {
        "input": input_arr,
        "dim": (1, 2),
        "keepdim": True,
        "dtype": torch.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_arr = torch.tensor([[True, False, True],
                              [False, True, True],
                              [True, True, False]], dtype=torch.bool).numpy()
    input_dict = {
        "input": input_arr,
        "dim": (1,),
        "keepdim": True,
        "dtype": torch.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_arr = torch.tensor(np.array([[[[1.0, float('nan'), 3.0, -4.0, 5.0],
                                         [6.0, -7.0, float('nan'), 9.0, -10.0],
                                         [11.0, 12.0, -13.0, 14.0, float('nan')],
                                         [16.0, -17.0, 18.0, 19.0, -20.0]],
                                        [[-1.5, 2.5, float('nan'), -4.5, 5.5],
                                         [6.5, float('nan'), 8.5, -9.5, 10.5],
                                         [11.5, -12.5, 13.5, float('nan'), 15.5],
                                         [16.5, 17.5, -18.5, 19.5, 20.5]],
                                        [[float('nan'), -2.0, 3.0, 4.0, -5.0],
                                         [6.0, 7.0, float('nan'), -9.0, 10.0],
                                         [11.0, -12.0, 13.0, 14.0, float('nan')],
                                         [16.0, 17.0, -18.0, 19.0, 20.0]]],
                                      [[[-1.0, 2.0, -3.0, 4.0, float('nan')],
                                        [6.0, -7.0, 8.0, float('nan'), 10.0],
                                        [11.0, float('nan'), 13.0, -14.0, 15.0],
                                        [16.0, -17.0, 18.0, 19.0, -20.0]],
                                       [[1.25, -2.25, 3.25, float('nan'), -5.25],
                                        [6.25, 7.25, -8.25, 9.25, float('nan')],
                                        [11.25, -12.25, 13.25, 14.25, -15.25],
                                        [float('nan'), 17.25, 18.25, -19.25, 20.25]],
                                       [[-1.75, float('nan'), 3.75, -4.75, 5.75],
                                        [6.75, 7.75, -8.75, 9.75, 10.75],
                                        [float('nan'), -12.75, 13.75, 14.75, -15.75],
                                        [16.75, 17.75, float('nan'), -19.75, 20.75]]]], dtype=np.float32)).numpy()
    input_dict = {
        "input": input_arr,
        "dim": (-4, -2),
        "keepdim": False,
        "dtype": torch.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_arr = torch.tensor([float('nan')], dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "dim": (0,),
        "keepdim": True,
        "dtype": torch.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nansum_3"] = nansum_inputs()



def nansum_inputs():
    list_of_inputs = []

    input_arr = torch.tensor([1.0, 2.0, float('nan'), -3.0], dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "dim": [0], "keepdim": False, "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[1.0, 2.0], [3.0, float('nan')]], dtype=torch.float64).numpy()
    input_dict = {"input": input_arr, "dim": [0], "keepdim": True, "dtype": torch.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[float('nan'), -5.5, 2.0], [1.0, float('nan'), 3.5]], dtype=torch.float16).numpy()
    input_dict = {"input": input_arr, "dim": [1], "keepdim": False, "dtype": torch.float16}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[[1.0, float('nan')], [2.0, 3.0]], [[float('nan'), 4.0], [5.0, float('nan')]]], dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "dim": [0, 2], "keepdim": True, "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.arange(24., dtype=torch.float64).reshape(2, 3, 4).numpy()
    input_dict = {"input": input_arr, "dim": [-1], "keepdim": False, "dtype": torch.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5).numpy()
    input_arr[0, 1, 2, 3] = np.nan
    input_arr[1, 2, 0, 4] = np.nan
    input_dict = {"input": input_arr, "dim": [1, 3], "keepdim": True, "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[-1, 2, -3], [4, -5, 6]], dtype=torch.int32).numpy()
    input_dict = {"input": input_arr, "dim": [0], "keepdim": False, "dtype": torch.int64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[1.0, float('inf')], [float('nan'), -float('inf')]], dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "dim": [0, 1], "keepdim": True, "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[[-1.5, 2.5], [float('nan'), -3.0]], [[4.0, -2.0], [1.0, float('nan')]]], dtype=torch.float64).numpy()
    input_dict = {"input": input_arr, "dim": [1], "keepdim": False, "dtype": torch.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 1, 3, 1, 4, dtype=torch.float32).numpy()
    input_arr[0, 0, 1, 0, 2] = np.nan
    input_arr[1, 0, 2, 0, 3] = np.nan
    input_dict = {"input": input_arr, "dim": [0, 2, 4], "keepdim": True, "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.arange(12., dtype=torch.float32).reshape(3, 4).t().numpy()
    input_dict = {"input": input_arr, "dim": [1], "keepdim": False, "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[-10, 20], [30, -40]], dtype=torch.int16).numpy()
    input_dict = {"input": input_arr, "dim": [0], "keepdim": True, "dtype": torch.int32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nansum_4"] = nansum_inputs()



def maxpool3d_inputs():
    list_of_inputs = []

    inp = torch.randn(2, 3, 8, 10, 12, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(1, 1, 7, 9, 11, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(3, 6, 7, 5, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(2, 8, 8, 8, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 4,
        "stride": 4,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(1, 4, 6, 6, 6, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "dilation": 2,
        "return_indices": False,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(5, 2, 5, 6, 4, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 3,
        "padding": 1,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(2, 3, 4, 5, 6, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 3,
        "padding": 1,
        "dilation": 2,
        "return_indices": True,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(2, 4, 9, 9, 9, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": 5,
        "stride": 2,
        "padding": 2,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(3, 1, 9, 9, 7, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "dilation": 3,
        "return_indices": False,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(7, 5, 5, 5, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 1,
        "padding": 0,
        "dilation": 2,
        "return_indices": False,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(1, 2, 3, 2, 4, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": True,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(2, 3, 11, 7, 9, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 5,
        "padding": 1,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool3d_1"] = maxpool3d_inputs()



def maxpool3d_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = torch.randn(2, 3, 10, 12, 14).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": (3, 3, 3),
        "stride": 2,
        "padding": 0,
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }))

    # Input 2
    input_arr = torch.randn(3, 8, 8, 8).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": (2, 2, 2),
        "stride": 2,
        "padding": 0,
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }))

    # Input 3
    input_arr = torch.randn(1, 1, 5, 7, 9).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": (3, 2, 2),
        "stride": 1,
        "padding": 0,
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": input_arr
    }))

    # Input 4
    input_arr = torch.randn(4, 2, 6, 6, 6).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": (3, 3, 3),
        "stride": 1,
        "padding": 1,
        "dilation": (2, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }))

    # Input 5
    input_arr = torch.randn(2, 1, 5, 5, 5).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": (4, 4, 4),
        "stride": 3,
        "padding": 1,
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": input_arr
    }))

    # Input 6
    input_arr = torch.randn(1, 3, 7, 8, 10).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": (2, 2, 2),
        "stride": 2,
        "padding": 0,
        "dilation": (1, 2, 3),
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }))

    # Input 7
    input_arr = torch.randn(2, 1, 20, 3, 7, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": (5, 1, 3),
        "stride": 1,
        "padding": 0,
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }))

    # Input 8
    input_arr = torch.randn(4, 9, 9, 9).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": (1, 3, 3),
        "stride": 2,
        "padding": 0,
        "dilation": (1, 2, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": input_arr
    }))

    # Input 9
    input_arr = torch.randn(8, 4, 15, 17, 19).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": (2, 3, 2),
        "stride": 3,
        "padding": 0,
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": True,
        "input": input_arr
    }))

    # Input 10
    input_arr = torch.randn(1, 5, 16, 16, 16).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": (4, 2, 2),
        "stride": 4,
        "padding": 0,
        "dilation": (2, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }))

    # Input 11
    input_arr = torch.randn(3, 2, 6, 7, 5).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": (2, 3, 2),
        "stride": 2,
        "padding": 1,
        "dilation": (3, 2, 2),
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }))

    # Input 12
    input_arr = torch.randn(2, 12, 11, 10).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": (3, 4, 5),
        "stride": 2,
        "padding": 1,
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool3d_10"] = maxpool3d_inputs()



def maxpool3d_inputs():
    list_of_inputs = []

    # 1
    input_arr = torch.randn(1, 1, 8, 8, 8).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": 2,
        "padding": (0, 0, 0),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2
    input_arr = torch.randn(2, 3, 10, 12, 14).numpy()
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": 3,
        "padding": (1, 1, 1),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3 (4D input: C, D, H, W)
    input_arr = torch.randn(4, 7, 9, 11).numpy()
    input_dict = {
        "kernel_size": (2, 3, 2),
        "stride": 2,
        "padding": (0, 1, 0),
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4 (dilation > 1)
    input_arr = torch.randn(4, 2, 5, 5, 5).numpy()
    input_dict = {
        "kernel_size": (3, 2, 2),
        "stride": 1,
        "padding": (0, 0, 0),
        "dilation": 2,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5 (mixed padding)
    input_arr = torch.randn(1, 1, 6, 7, 8).numpy()
    input_dict = {
        "kernel_size": (4, 3, 2),
        "stride": 2,
        "padding": (1, 0, 1),
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6 (ceil_mode and return_indices True)
    input_arr = torch.randn(3, 5, 9, 9, 9).numpy()
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": 2,
        "padding": (0, 0, 0),
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7 (larger padding on depth)
    input_arr = torch.randn(2, 1, 15, 13, 11).numpy()
    input_dict = {
        "kernel_size": (5, 3, 3),
        "stride": 3,
        "padding": (2, 1, 1),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8 (4D input with ceil_mode True)
    input_arr = torch.randn(2, 4, 6, 8).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": 2,
        "padding": (0, 0, 0),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9 (dilation 2 with padding and larger stride)
    input_arr = torch.randn(1, 7, 16, 12, 9).numpy()
    input_dict = {
        "kernel_size": (4, 4, 3),
        "stride": 4,
        "padding": (1, 2, 1),
        "dilation": 2,
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10 (wide kernels in H and W)
    input_arr = torch.randn(5, 3, 20, 10, 5).numpy()
    input_dict = {
        "kernel_size": (2, 5, 5),
        "stride": 5,
        "padding": (0, 0, 0),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 11 (small input with padding to allow pooling)
    input_arr = torch.randn(2, 2, 3, 3, 3).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": 1,
        "padding": (1, 1, 1),
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 12 (very small input, ceil_mode True)
    input_arr = torch.tensor([[[[[1.0, -1.0],
                                 [2.0, -2.0]],
                                [[-3.0, 3.0],
                                 [4.0, -4.0]]]]], dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": 2,
        "padding": (0, 0, 0),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool3d_11"] = maxpool3d_inputs()



def maxpool3d_inputs():
    list_of_inputs = []
    
    # 1
    input = torch.randn(2, 3, 8, 10, 12, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": 2,
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 2
    input = torch.randn(4, 7, 9, 11, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": 1,
        "padding": (1, 1, 1),
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 3 (fixed padding)
    input = torch.randn(1, 1, 10, 10, 10, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": 1,
        "padding": (1, 1, 1),
        "dilation": (2, 2, 2),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 4
    input = torch.randn(1, 2, 11, 9, 7, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": 3,
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 5
    input = torch.randn(3, 3, 5, 4, 3, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 3, 2),
        "stride": 2,
        "padding": (1, 0, 1),
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 6
    input = torch.randn(2, 1, 20, 15, 10, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (4, 3, 2),
        "stride": 2,
        "padding": (0, 0, 0),
        "dilation": (2, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 7 (4D input)
    input = torch.linspace(-10, 10, steps=2*5*5*5, dtype=torch.float32).reshape(2, 5, 5, 5).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": 2,
        "padding": (1, 1, 1),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 8 (avoid float16 on CPU)
    input = torch.randn(4, 4, 6, 8, 10, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 2, 4),
        "stride": 1,
        "padding": (0, 1, 2),
        "dilation": (1, 2, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 9
    input = torch.randn(1, 3, 9, 9, 9, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": (5, 5, 5),
        "stride": 4,
        "padding": (1, 1, 1),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 10
    input = torch.randn(2, 2, 4, 6, 8, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (4, 6, 2),
        "stride": 2,
        "padding": (2, 3, 0),
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 11 (4D input)
    input = torch.randn(1, 8, 7, 6, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 2, 2),
        "stride": 3,
        "padding": (0, 1, 1),
        "dilation": (1, 2, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 12
    input = torch.randn(1, 4, 3, 3, 20, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2, 7),
        "stride": 5,
        "padding": (0, 0, 3),
        "dilation": (1, 1, 2),
        "return_indices": True,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.MaxPool3d_12"] = maxpool3d_inputs()



def maxpool3d_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = torch.randn(1, 1, 4, 4, 4, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": (2, 2, 2),
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = torch.randn(2, 3, 5, 6, 8, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 2, 2),
        "stride": (1, 2, 2),
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = torch.randn(1, 2, 5, 5, 5, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": (2, 2, 2),
        "padding": 1,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_arr = torch.randn(1, 1, 5, 5, 5, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": (1, 1, 1),
        "padding": 0,
        "dilation": 2,
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_arr = torch.randn(1, 1, 6, 6, 6, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": (2, 2, 2),
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 (4D input: C, D, H, W)
    input_arr = torch.randn(3, 10, 12, 14, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (4, 5, 6),
        "stride": (4, 4, 4),
        "padding": 1,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (float64)
    input_arr = torch.randn(2, 4, 7, 8, 9, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": (2, 3, 2),
        "stride": (2, 3, 2),
        "padding": 1,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (dilation > 1 with padding adjusted to be valid)
    input_arr = torch.randn(1, 2, 6, 6, 6, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": (1, 1, 1),
        "padding": 1,
        "dilation": 3,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (negative values)
    input_arr = torch.full((1, 1, 3, 3, 3), -5.0, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 1, 1),
        "stride": (1, 1, 1),
        "padding": 0,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (small input, padding adjusted to be valid)
    input_arr = torch.randn(1, 1, 2, 2, 2, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (1, 1, 1),
        "stride": (2, 2, 2),
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 (larger multi-channel with dilation)
    input_arr = torch.randn(4, 8, 15, 13, 11, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": (1, 2, 2),
        "padding": 1,
        "dilation": 2,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12 (4D input with ceil_mode)
    input_arr = torch.randn(5, 9, 10, 10, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 4, 5),
        "stride": (3, 4, 5),
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool3d_13"] = maxpool3d_inputs()



def maxpool3d_inputs_14():
    list_of_inputs = []

    # Input 1
    input_arr = torch.randn(2, 3, 8, 10, 12, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": (2, 2, 2),
        "padding": 0,
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2 (4D input)
    input_arr = torch.randn(1, 7, 9, 11, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": (1, 2, 2),
        "padding": 1,
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = torch.randn(1, 2, 5, 5, 5, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 1, 1),
        "stride": (1, 1, 1),
        "padding": 0,
        "dilation": (2, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_arr = torch.randn(3, 4, 9, 9, 9, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 3, 4),
        "stride": (1, 2, 3),
        "padding": 0,
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 (dilation > 1, padding adjusted to be valid)
    input_arr = torch.randn(1, 1, 10, 12, 14, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": (2, 2, 2),
        "padding": 1,
        "dilation": (2, 2, 2),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 (ceil_mode=True)
    input_arr = torch.randn(2, 2, 7, 8, 9, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": (3, 3, 3),
        "padding": 0,
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (padding > 0)
    input_arr = torch.randn(1, 3, 4, 5, 6, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": (2, 2, 1),
        "padding": 1,
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (4D input, non-cubic kernel)
    input_arr = torch.randn(2, 8, 6, 10, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 4, 3),
        "stride": (2, 3, 2),
        "padding": 0,
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (4D input, ceil_mode=True, padding > 0)
    input_arr = torch.randn(1, 6, 6, 6, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": (2, 2, 2),
        "padding": 1,
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (mixed dilation)
    input_arr = torch.randn(2, 5, 5, 10, 10, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (1, 2, 3),
        "stride": (1, 2, 1),
        "padding": 0,
        "dilation": (1, 2, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 (dilation with stride=1)
    input_arr = torch.randn(1, 1, 15, 15, 15, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": (1, 1, 1),
        "padding": 0,
        "dilation": (3, 1, 2),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12 (minimal spatial dims with kernel=1)
    input_arr = torch.randn(4, 1, 1, 1, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (1, 1, 1),
        "stride": (1, 1, 1),
        "padding": 0,
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool3d_14"] = maxpool3d_inputs_14()



def maxpool3d_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = torch.randn(2, 4, 10, 12, 14).numpy()
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": (2, 2, 2),
        "padding": (0, 0, 0),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = torch.randn(20, 16, 50, 44, 31).numpy()
    input_dict = {
        "kernel_size": (3, 2, 2),
        "stride": (2, 1, 2),
        "padding": (0, 0, 0),
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = torch.randn(1, 1, 3, 3, 3).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": (2, 2, 2),
        "padding": (1, 1, 1),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4 (fixed padding to satisfy pad <= floor(kernel/2))
    input_arr = torch.randn(1, 2, 8, 9, 10).numpy()
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": (1, 1, 1),
        "padding": (1, 1, 1),
        "dilation": 2,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_arr = torch.randn(3, 5, 9, 9, 9).numpy()
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": (2, 2, 2),
        "padding": (1, 1, 1),
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 (4D input without batch)
    input_arr = torch.randn(4, 12, 13, 14).numpy()
    input_dict = {
        "kernel_size": (2, 3, 4),
        "stride": (2, 3, 4),
        "padding": (0, 1, 2),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (use float32 to ensure broad CPU support)
    input_arr = torch.randn(2, 3, 16, 16, 16, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (1, 1, 1),
        "stride": (1, 1, 1),
        "padding": (0, 0, 0),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_arr = torch.randn(1, 2, 7, 8, 9).numpy()
    input_dict = {
        "kernel_size": (4, 4, 4),
        "stride": (3, 3, 3),
        "padding": (2, 1, 0),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (zeros)
    input_arr = torch.zeros(1, 1, 5, 5, 5).numpy()
    input_dict = {
        "kernel_size": (5, 5, 5),
        "stride": (5, 5, 5),
        "padding": (0, 0, 0),
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (dilation 3)
    input_arr = torch.randn(2, 2, 10, 10, 10).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": (2, 2, 2),
        "padding": (0, 0, 0),
        "dilation": 3,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 (4D input, negative-heavy)
    input_arr = (torch.randn(1, 7, 7, 7) * 10.0).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": (1, 2, 3),
        "padding": (0, 0, 1),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12 (prime sizes with dilation 2)
    input_arr = torch.randn(4, 11, 17, 19).numpy()
    input_dict = {
        "kernel_size": (1, 4, 2),
        "stride": (2, 1, 2),
        "padding": (0, 1, 0),
        "dilation": 2,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool3d_15"] = maxpool3d_inputs()



def maxpool3d_inputs():
    list_of_inputs = []

    inp = torch.randn(1, 1, 4, 4, 4).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": (2, 2, 2),
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(2, 3, 5, 6, 7).numpy()
    input_dict = {
        "kernel_size": (3, 2, 2),
        "stride": (2, 1, 2),
        "padding": (1, 0, 1),
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(1, 4, 8, 9, 10).numpy()
    input_dict = {
        "kernel_size": (2, 2, 3),
        "stride": (1, 2, 1),
        "padding": (1, 1, 1),
        "dilation": (2, 2, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(3, 6, 7, 8).numpy()
    input_dict = {
        "kernel_size": (1, 3, 1),
        "stride": (1, 2, 1),
        "padding": (0, 1, 0),
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(2, 5, 10, 9, 8).numpy()
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": (2, 2, 2),
        "padding": (1, 1, 1),
        "dilation": (2, 2, 2),
        "return_indices": False,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(4, 2, 11, 5, 13).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": (3, 2, 4),
        "padding": (0, 1, 0),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(1, 1, 3, 3, 4).numpy()
    input_dict = {
        "kernel_size": (4, 1, 5),
        "stride": (1, 1, 1),
        "padding": (2, 0, 2),
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = (torch.randn(2, 3, 2, 2, 2) * -2.0).numpy()
    input_dict = {
        "kernel_size": (1, 1, 1),
        "stride": (1, 1, 1),
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(1, 2, 6, 10, 7).numpy()
    input_dict = {
        "kernel_size": (2, 3, 4),
        "stride": (2, 3, 1),
        "padding": (1, 1, 2),
        "dilation": (1, 2, 1),
        "return_indices": True,
        "ceil_mode": True,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(1, 1, 5, 5, 5).numpy()
    input_dict = {
        "kernel_size": (5, 5, 5),
        "stride": (3, 3, 3),
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(3, 2, 4, 3, 3).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": (1, 1, 1),
        "padding": (1, 1, 1),
        "dilation": (2, 1, 3),
        "return_indices": False,
        "ceil_mode": True,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(2, 7, 8, 9).numpy()
    input_dict = {
        "kernel_size": (1, 2, 2),
        "stride": (1, 2, 2),
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool3d_16"] = maxpool3d_inputs()



def maxpool3d_inputs():
    list_of_inputs = []

    # 1
    input_arr = np.random.randn(1, 1, 5, 6, 7).astype(np.float32)
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 0,
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2 (4D input)
    input_arr = np.random.randn(3, 10, 12, 14).astype(np.float32)
    input_dict = {
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3
    input_arr = np.random.randn(2, 4, 9, 9, 9).astype(np.float64)
    input_dict = {
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4 (fixed padding to 1 to satisfy constraint)
    input_arr = np.random.randn(1, 2, 8, 6, 6).astype(np.float32)
    input_dict = {
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "dilation": (2, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5
    input_arr = np.random.randn(1, 1, 7, 5, 4).astype(np.float32)
    input_dict = {
        "kernel_size": 2,
        "stride": 3,
        "padding": 1,
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6
    input_arr = np.random.randn(2, 3, 15, 15, 15).astype(np.float32)
    input_dict = {
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "dilation": (2, 2, 2),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7 (fixed padding to 0 because k=1)
    input_arr = np.random.randn(4, 2, 3, 3, 3).astype(np.float32)
    input_dict = {
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "dilation": (3, 3, 3),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8 (4D input)
    input_arr = np.random.randn(1, 9, 8, 7).astype(np.float32)
    input_dict = {
        "kernel_size": 3,
        "stride": 3,
        "padding": 0,
        "dilation": (1, 2, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9
    input_arr = np.random.randn(1, 5, 10, 12, 20).astype(np.float32)
    input_dict = {
        "kernel_size": 4,
        "stride": 4,
        "padding": 1,
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10
    input_arr = np.random.randn(8, 3, 16, 16, 16).astype(np.float64)
    input_dict = {
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 11
    input_arr = np.random.randn(1, 1, 2, 2, 2).astype(np.float32)
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 12 (fixed padding to 1 to satisfy constraint for dilation=1 dim)
    input_arr = np.random.randn(2, 1, 20, 10, 12).astype(np.float32)
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "dilation": (1, 3, 2),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool3d_2"] = maxpool3d_inputs()



def maxpool3d_inputs():
    list_of_inputs = []

    # 1
    input_arr = torch.randn(2, 3, 6, 8, 10, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 2,
        "padding": (0, 0, 0),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2
    input_arr = torch.randn(3, 5, 7, 9, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 1,
        "padding": (1, 1, 1),
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3
    input_arr = torch.randn(1, 1, 10, 12, 14, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 4,
        "stride": 2,
        "padding": (1, 0, 2),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4
    input_arr = torch.randn(4, 2, 15, 17, 19, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 5,
        "stride": 3,
        "padding": (2, 2, 2),
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5
    input_arr = torch.randn(2, 4, 5, 8, 10, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 1,
        "padding": (0, 1, 1),
        "dilation": 2,
        "return_indices": False,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6
    input_arr = torch.randn(3, 3, 7, 9, 11, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": (0, 0, 0),
        "dilation": 2,
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7
    vals = torch.arange(1*2*6*6*6, dtype=torch.float64) - 500.0
    input_arr = vals.view(1, 2, 6, 6, 6).numpy()
    input_dict = {
        "kernel_size": 1,
        "stride": 1,
        "padding": (0, 0, 0),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8
    input_arr = torch.randn(1, 5, 4, 5, 6, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 3,
        "padding": (1, 1, 1),
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9
    input_arr = torch.randn(1, 1, 20, 10, 16, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 7,
        "stride": 1,
        "padding": (3, 0, 2),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10 (adjusted padding to satisfy constraints)
    input_arr = torch.randn(2, 2, 15, 14, 13, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 3,
        "padding": (0, 1, 0),
        "dilation": 3,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 11
    input_arr = torch.randn(3, 1, 4, 3, 2, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 2,
        "padding": (1, 1, 1),
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 12
    vals = torch.linspace(-10.0, 10.0, steps=8*3*3*3, dtype=torch.float32)
    input_arr = vals.view(8, 3, 3, 3).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 1,
        "padding": (0, 0, 0),
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool3d_3"] = maxpool3d_inputs()



def maxpool3d_inputs():
    list_of_inputs = []

    input_arr = torch.randn(2, 3, 10, 10, 10, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": 2,
        "stride": 2,
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }))

    input_arr = torch.randn(1, 2, 7, 7, 7, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": 3,
        "stride": 2,
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }))

    input_arr = torch.randn(1, 1, 5, 6, 7, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": 3,
        "stride": 2,
        "padding": (1, 1, 0),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }))

    input_arr = torch.randn(3, 2, 5, 5, 5, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": 2,
        "stride": 1,
        "padding": (0, 0, 0),
        "dilation": (2, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }))

    input_arr = torch.randn(2, 4, 9, 8, 7, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": 3,
        "stride": 4,
        "padding": (1, 1, 1),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": input_arr
    }))

    input_arr = torch.randn(3, 8, 8, 8, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": 2,
        "stride": 2,
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }))

    input_arr = torch.randn(1, 4, 5, 6, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": 3,
        "stride": 1,
        "padding": (1, 0, 1),
        "dilation": (2, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }))

    input_arr = (-torch.rand(1, 5, 3, 4, 5, dtype=torch.float32)).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": 1,
        "stride": 1,
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }))

    input_arr = torch.randn(2, 2, 5, 5, 5, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": 2,
        "stride": 1,
        "padding": (1, 1, 1),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }))

    input_arr = torch.randn(1, 1, 15, 16, 17, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": 5,
        "stride": 2,
        "padding": (2, 2, 2),
        "dilation": (2, 2, 2),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }))

    input_arr = torch.randn(1, 1, 1, 1, 1, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": 1,
        "stride": 2,
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }))

    input_arr = torch.randn(4, 3, 12, 10, 8, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": 4,
        "stride": 3,
        "padding": (2, 1, 0),
        "dilation": (1, 2, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool3d_4"] = maxpool3d_inputs()



def maxpool3d_inputs():
    list_of_inputs = []

    input = torch.randn(1, 1, 8, 8, 8, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (2, 2, 2),
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 3, 10, 12, 14, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (2, 1, 2),
        "padding": 0,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(4, 2, 5, 5, 5, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (2, 2, 2),
        "padding": 1,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 1, 10, 10, 10, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (1, 1, 1),
        "padding": 1,
        "dilation": 2,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 1, 7, 7, 7, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (3, 3, 3),
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(3, 9, 9, 9, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (3, 3, 3),
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 5, 20, 15, 12, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 4,
        "stride": (4, 3, 2),
        "padding": 1,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(3, 3, 18, 18, 18, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (2, 2, 2),
        "padding": 1,
        "dilation": 2,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 4, 6, 7, 8, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 1,
        "stride": (1, 2, 3),
        "padding": 0,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 2, 3, 3, 3, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (1, 1, 1),
        "padding": 1,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 10, 8, 10, 12, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 5,
        "stride": (2, 5, 4),
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 3, 11, 13, 17, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (2, 4, 5),
        "padding": 1,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool3d_5"] = maxpool3d_inputs()



def maxpool3d_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = torch.randn(1, 1, 4, 4, 4).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (2, 2, 2),
        "padding": 0,
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2 (4D input)
    input_arr = torch.randn(3, 5, 6, 7).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (1, 2, 2),
        "padding": 1,
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3 (dilation > 1, float64)
    input_arr = torch.randn(2, 2, 5, 5, 5, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (1, 1, 1),
        "padding": 0,
        "dilation": (2, 2, 2),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4 (non-uniform stride, return_indices True)
    input_arr = torch.randn(4, 8, 9, 10, 11).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (3, 2, 4),
        "padding": 1,
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 (ceil_mode True)
    input_arr = torch.randn(1, 1, 5, 5, 5).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (2, 2, 2),
        "padding": 0,
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 (valid padding)
    input_arr = torch.randn(1, 1, 1, 3, 3).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (1, 1, 1),
        "padding": 1,
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (return_indices True, mixed stride)
    input_arr = torch.randn(2, 3, 7, 9, 11).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (2, 3, 4),
        "padding": 1,
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (mixed dilation, ceil_mode True, adjusted padding)
    input_arr = torch.randn(3, 1, 6, 6, 6).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (1, 2, 2),
        "padding": 1,
        "dilation": (2, 1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (4D input, float64)
    input_arr = torch.randn(4, 5, 8, 9, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (2, 2, 2),
        "padding": 0,
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (k=1 with dilation > 1)
    input_arr = torch.ones(5, 6, 7, 8, 9).numpy()
    input_dict = {
        "kernel_size": 1,
        "stride": (1, 1, 1),
        "padding": 0,
        "dilation": (3, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool3d_6"] = maxpool3d_inputs()



def maxpool3d_inputs():
    list_of_inputs = []

    input_arr = torch.randn(1, 1, 8, 8, 8).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (2, 2, 2),
        "padding": (0, 0, 0),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(3, 10, 12, 14).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (2, 2, 2),
        "padding": (1, 1, 1),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 4, 20, 15, 15).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (1, 2, 2),
        "padding": (1, 1, 1),
        "dilation": 2,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(1, 2, 7, 7, 7).numpy()
    input_dict = {
        "kernel_size": 4,
        "stride": (3, 3, 3),
        "padding": (1, 1, 1),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(4, 6, 16, 17, 9).numpy()
    input_dict = {
        "kernel_size": 5,
        "stride": (2, 3, 2),
        "padding": (2, 1, 0),
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(5, 3, 3, 3).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (1, 1, 1),
        "padding": (1, 1, 1),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(3, 2, 25, 18, 33).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (3, 2, 4),
        "padding": (0, 1, 1),
        "dilation": 3,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(8, 1, 9, 20, 11, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (1, 2, 1),
        "padding": (0, 1, 1),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 3, 7, 9, 11).numpy()
    input_dict = {
        "kernel_size": 1,
        "stride": (2, 2, 2),
        "padding": (0, 0, 0),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(5, 7, 10, 13, 20).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (2, 3, 4),
        "padding": (1, 0, 1),
        "dilation": 2,
        "return_indices": True,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(1, 2, 60, 33, 25, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": 7,
        "stride": (6, 5, 4),
        "padding": (3, 2, 1),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 4, 4, 4).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (2, 2, 2),
        "padding": (1, 1, 1),
        "dilation": 2,
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool3d_7"] = maxpool3d_inputs()



def maxpool3d_inputs():
    list_of_inputs = []

    inp = torch.randn(2, 3, 8, 8, 8, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (2, 2, 2),
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(1, 4, 7, 5, 9, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (2, 1, 2),
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(3, 2, 10, 10, 10, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 4,
        "stride": (3, 2, 2),
        "padding": (1, 1, 1),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(1, 1, 9, 12, 8, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (2, 2, 2),
        "padding": (1, 1, 0),
        "dilation": (2, 3, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(5, 3, 3, 3, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (2, 2, 2),
        "padding": (1, 1, 1),
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": True,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(2, 2, 4, 5, 6, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 1,
        "stride": (1, 1, 1),
        "padding": (0, 0, 0),
        "dilation": (3, 2, 2),
        "return_indices": False,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(1, 1, 2, 2, 2, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (2, 2, 2),
        "padding": (1, 1, 0),
        "dilation": (3, 1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(2, 3, 15, 20, 16, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 5,
        "stride": (3, 4, 2),
        "padding": (2, 0, 1),
        "dilation": (1, 2, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(4, 1, 10, 11, 12, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (3, 2, 4),
        "padding": (1, 1, 1),
        "dilation": (6, 6, 6),
        "return_indices": True,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(16, 18, 22, 30, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (2, 3, 5),
        "padding": (0, 1, 1),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randint(-10, 10, (1, 2, 6, 6, 6), dtype=torch.int32).to(torch.float32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (1, 1, 1),
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(1, 7, 6, 9, 10, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (1, 2, 3),
        "padding": (1, 1, 1),
        "dilation": (3, 2, 5),
        "return_indices": True,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(2, 2, 7, 7, 7, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (1, 1, 1),
        "padding": (0, 0, 0),
        "dilation": (3, 3, 3),
        "return_indices": False,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(1, 1, 3, 3, 3, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (3, 3, 3),
        "padding": (1, 1, 1),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool3d_8"] = maxpool3d_inputs()



def maxpool3d_inputs():
    list_of_inputs = []

    # Input 1
    input = torch.randn(1, 1, 4, 4, 4, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input = torch.randn(2, 3, 10, 8, 6, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 2, 2),
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input = torch.randn(1, 2, 7, 7, 7, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": 2,
        "padding": 1,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4 (4D input)
    input = torch.randn(3, 9, 9, 9, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": 3,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 (dilation > 1)
    input = torch.randn(1, 4, 12, 10, 9, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 2, 2),
        "stride": 2,
        "padding": 1,
        "dilation": 2,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 (ceil_mode True)
    input = torch.randn(1, 1, 5, 5, 5, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (return_indices True)
    input = torch.randn(2, 2, 6, 6, 6, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": 3,
        "padding": 0,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (adjusted: padding compatible with kernel)
    input = torch.randn(1, 1, 4, 3, 2, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (5, 5, 5),
        "stride": 1,
        "padding": 2,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (dilation 3)
    input = torch.randn(1, 3, 9, 10, 11, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": 2,
        "padding": 1,
        "dilation": 3,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (4D input, larger kernel/stride)
    input = torch.randn(4, 10, 12, 14, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": (4, 4, 4),
        "stride": 4,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 (unit kernel)
    input = torch.randn(3, 3, 3, 3, 3, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (1, 1, 1),
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12 (float tensor)
    input = torch.arange(2 * 3 * 12 * 8 * 10, dtype=torch.float32).reshape(2, 3, 12, 8, 10).numpy()
    input_dict = {
        "kernel_size": (2, 3, 4),
        "stride": 5,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool3d_9"] = maxpool3d_inputs()



def logsigmoid_inputs():
    list_of_inputs = []
    
    input = torch.tensor([1.0, -1.0, 0.0, 10.0, -10.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor([[100.0, -100.0, 0.5],
                          [-0.5, 20.0, -20.0]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor(0.0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.randn(2, 3, 4, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.linspace(-5, 5, steps=2*3*4*5, dtype=torch.float32).reshape(2, 3, 4, 5).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.empty(0, 4, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor([[-1.5, 0.0, 1.5],
                          [6.0, -6.0, 2.0]], dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    t = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    input = t.numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    t = torch.linspace(-8, 8, steps=17, dtype=torch.float32)
    input = t[::2].numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.zeros(1, 2, 1, 3, 2, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = torch.tensor([float('nan'), float('inf'), float('-inf'), -0.0, 0.0, 1e-8], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    input = (-10.0 * torch.rand(4, 5, dtype=torch.float64)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.logsigmoid"] = logsigmoid_inputs()



def sparse__inputs():
    list_of_inputs = []

    t = torch.zeros((3, 4), dtype=torch.float32).numpy()
    input_dict = {"tensor": t, "sparsity": 0.5, "std": 0.01}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.ones((10, 10), dtype=torch.float64).numpy()
    input_dict = {"tensor": t, "sparsity": 0.1, "std": 0.05}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.randn((8, 3), dtype=torch.float16).numpy()
    input_dict = {"tensor": t, "sparsity": 0.8, "std": 0.001}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.tensor([[1.0, -2.0, 3.0, -4.0, 5.0]], dtype=torch.float32).numpy()
    input_dict = {"tensor": t, "sparsity": 0.0, "std": 0.02}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.tensor([[-1.0], [2.0], [-3.0], [4.0], [-5.0]], dtype=torch.float32).numpy()
    input_dict = {"tensor": t, "sparsity": 1.0, "std": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.zeros((100, 50), dtype=torch.float32).numpy()
    input_dict = {"tensor": t, "sparsity": 0.95, "std": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.tensor([[0.5, -0.5], [1.5, -1.5]], dtype=torch.float32).numpy()
    input_dict = {"tensor": t, "sparsity": 0.25, "std": 0.1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.full((7, 7), 0.3, dtype=torch.float32).numpy()
    input_dict = {"tensor": t, "sparsity": 0.9, "std": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = torch.arange(3 * 7, dtype=torch.float64).view(3, 7).numpy()
    t = base
    input_dict = {"tensor": t, "sparsity": 0.6, "std": 0.05}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.randn((64, 16), dtype=torch.float16).numpy()
    input_dict = {"tensor": t, "sparsity": 0.7, "std": 0.2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.linspace(-1, 1, steps=16, dtype=torch.float32).view(2, 8).numpy()
    input_dict = {"tensor": t, "sparsity": 0.99, "std": 0.0001}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.eye(32, dtype=torch.float64).numpy()
    input_dict = {"tensor": t, "sparsity": 0.33, "std": 3.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.init.sparse_"] = sparse__inputs()



def reshape_inputs():
    list_of_inputs = []

    input = torch.arange(6, dtype=torch.float32).numpy()
    shape = (2, 3)
    list_of_inputs.append(copy.deepcopy({"input": input, "shape": shape}))

    input = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int64).numpy()
    shape = (-1,)
    list_of_inputs.append(copy.deepcopy({"input": input, "shape": shape}))

    input = (torch.arange(24) % 2 == 0).view(2, 3, 4).numpy()
    shape = (3, 8)
    list_of_inputs.append(copy.deepcopy({"input": input, "shape": shape}))

    input = torch.tensor([1+2j, 3+4j, 5+6j, 7+8j], dtype=torch.complex64).view(1, 1, 4).numpy()
    shape = (2, 2)
    list_of_inputs.append(copy.deepcopy({"input": input, "shape": shape}))

    input = torch.tensor([], dtype=torch.float32).numpy()
    shape = (0, 2)
    list_of_inputs.append(copy.deepcopy({"input": input, "shape": shape}))

    input = torch.randn(2, 1, 3, 4, dtype=torch.float64).numpy()
    shape = (-1, 2, 4)
    list_of_inputs.append(copy.deepcopy({"input": input, "shape": shape}))

    t = torch.arange(12, dtype=torch.int32).view(6, 2)
    input = t[:, 0].numpy()
    shape = (2, 3)
    list_of_inputs.append(copy.deepcopy({"input": input, "shape": shape}))

    input = torch.tensor([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=torch.int16).numpy()
    shape = (5, 2)
    list_of_inputs.append(copy.deepcopy({"input": input, "shape": shape}))

    input = torch.arange(32, dtype=torch.float16).view(2, 4, 4).numpy()
    shape = (4, 2, 4)
    list_of_inputs.append(copy.deepcopy({"input": input, "shape": shape}))

    input = torch.tensor(5.0, dtype=torch.float32).numpy()
    shape = (1,)
    list_of_inputs.append(copy.deepcopy({"input": input, "shape": shape}))

    input = torch.arange(24, dtype=torch.int32).numpy()
    shape = (2, -1, 3)
    list_of_inputs.append(copy.deepcopy({"input": input, "shape": shape}))

    input = torch.tensor([[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j]], dtype=torch.complex128).numpy()
    shape = (-1,)
    list_of_inputs.append(copy.deepcopy({"input": input, "shape": shape}))

    return list_of_inputs

generated_inputs["torch.reshape"] = reshape_inputs()



def set_autocast_enabled_inputs():
    list_of_inputs = []

    input_dict = {"enabled": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"enabled": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"enabled": bool(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"enabled": bool(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"enabled": bool(-1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"enabled": bool(2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"enabled": bool("")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"enabled": bool("enabled")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"enabled": bool([])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"enabled": bool([1])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"enabled": bool({})}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"enabled": bool({"a": 1})}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.set_autocast_enabled"] = set_autocast_enabled_inputs()



def ndtri_inputs():
    list_of_inputs = []

    inp_t = torch.tensor(0.5, dtype=torch.float32)
    input = inp_t.numpy()
    out = torch.empty_like(inp_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    inp_t = torch.tensor([1e-10, 0.2, 0.5, 0.8, 1 - 1e-10], dtype=torch.float64)
    input = inp_t.numpy()
    out = torch.empty_like(inp_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    inp_t = torch.tensor([[1e-3, 1e-2, 0.1],
                          [0.9, 0.99, 0.999]], dtype=torch.float32)
    input = inp_t.numpy()
    out = torch.empty_like(inp_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    inp_t = torch.rand((2, 1, 3), dtype=torch.float32)
    input = inp_t.numpy()
    out = torch.empty_like(inp_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    inp_t = torch.tensor([0.0, 1.0], dtype=torch.float64)
    input = inp_t.numpy()
    out = torch.empty_like(inp_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    inp_t = torch.tensor([-0.1, 1.2, 2.0, -5.0], dtype=torch.float32)
    input = inp_t.numpy()
    out = torch.empty_like(inp_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    inp_t = torch.empty((0,), dtype=torch.float32)
    input = inp_t.numpy()
    out = torch.empty_like(inp_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    base = torch.linspace(0.0, 1.0, steps=10, dtype=torch.float64)
    inp_t = base[::2]
    input = inp_t.numpy()
    out = torch.empty_like(inp_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    inp_t = torch.tensor([[float('nan'), 0.3],
                          [0.7, float('nan')]], dtype=torch.float32)
    input = inp_t.numpy()
    out = torch.empty_like(inp_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    inp_t = torch.rand((2, 2, 1, 3), dtype=torch.float64)
    input = inp_t.numpy()
    out = torch.empty_like(inp_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    inp_t = torch.rand((4, 3), dtype=torch.float32).t()
    input = inp_t.numpy()
    out = torch.empty_like(inp_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.special.ndtri"] = ndtri_inputs()



def xlog1py_inputs():
    list_of_inputs = []
    
    input = torch.tensor([0.0, 1.0, -2.0], dtype=torch.float32).numpy()
    other = torch.tensor([0.1, 0.5, 2.0], dtype=torch.float32).numpy()
    out = torch.empty(3, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([[1.0, -1.5, 0.0],
                          [3.2, -0.7, 5.0]], dtype=torch.float64).numpy()
    other = torch.tensor([[-0.9999, 0.0, 10.0],
                          [-0.5, 0.2, 3.5]], dtype=torch.float64).numpy()
    out = torch.empty((2, 3), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64).numpy()
    other = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float64).numpy()
    out = torch.empty((3, 4), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor(2.5, dtype=torch.float64).numpy()
    other = torch.tensor(-0.3, dtype=torch.float64).numpy()
    out = torch.empty((), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([[[0.5, 1.0, 2.0],
                           [3.0, 4.0, 5.0]],
                          [[6.0, 7.0, 8.0],
                           [9.0, 10.0, 11.0]]], dtype=torch.float16).numpy()
    other = torch.tensor([[[0.1, 0.2, 0.3],
                           [0.4, 0.5, 0.6]],
                          [[0.7, 0.8, 0.9],
                           [1.0, 1.1, 1.2]]], dtype=torch.float16).numpy()
    out = torch.empty((2, 2, 3), dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    t = torch.arange(6.0, dtype=torch.float32).reshape(3, 2)
    input = t.t().numpy()
    other = (t.t() + 1.0).numpy()
    out = torch.empty((2, 3), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([1e20, -1e20], dtype=torch.float64).numpy()
    other = torch.tensor([1e-12, 1e-12], dtype=torch.float64).numpy()
    out = torch.empty(2, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([1e-12, -1e-12, 3.0], dtype=torch.float64).numpy()
    other = torch.tensor([-1.0 + 1e-8, -0.9999999, -0.999], dtype=torch.float64).numpy()
    out = torch.empty(3, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    base = torch.linspace(-2.0, 2.0, steps=10, dtype=torch.float32)
    input = base[::2].numpy()
    other = torch.linspace(0.1, 0.5, steps=5, dtype=torch.float32).numpy()
    out = torch.empty(5, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([0.5, -1.0, 2.0, -3.5], dtype=torch.float32).numpy()
    other = torch.tensor([[[0.1, 0.2, 0.3, 0.4]],
                          [[-0.5, -0.4, -0.3, -0.2]]], dtype=torch.float32).numpy()
    out = torch.empty((2, 1, 4), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.empty((0,), dtype=torch.float32).numpy()
    other = torch.empty((0,), dtype=torch.float32).numpy()
    out = torch.empty((0,), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([float('nan'), float('inf'), -float('inf')], dtype=torch.float64).numpy()
    other = torch.tensor([0.5, -0.2, 10.0], dtype=torch.float64).numpy()
    out = torch.empty(3, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    return list_of_inputs

generated_inputs["torch.special.xlog1py_1"] = xlog1py_inputs()



def xlog1py_inputs():
    list_of_inputs = []

    input = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32).numpy()
    other = 0.0
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([[1.0, 2.0], [-3.5, 4.5]], dtype=torch.float64).numpy()
    other = 1.0
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor(3.14159265, dtype=torch.float32).numpy()
    other = 2.0
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    base = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2) - 12.0
    input = base.numpy()
    other = -1.0
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor(
        [[[-1e-6, 1e-6], [1.5, -2.5]],
         [[0.3, -0.3], [10.0, -10.0]]],
        dtype=torch.float64
    ).numpy()
    other = 1e-6
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([float('nan'), float('inf'), float('-inf'), 5.0], dtype=torch.float32).numpy()
    other = 3.0
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.empty(0, dtype=torch.float64).numpy()
    other = 0.5
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float32).reshape(1, 2, 1, 2, 1).numpy()
    other = 1e20
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    base = (torch.arange(12, dtype=torch.float32).reshape(3, 4) - 6.0)
    input = base[:, ::2].numpy()
    other = -2.0
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([0.0, 1000.0, -1000.0, 1e30, -1e30], dtype=torch.float64).numpy()
    other = 0.75
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.linspace(-5.0, 5.0, steps=11, dtype=torch.float32).reshape(1, 11).numpy()
    other = -0.999999999999
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.randn(2, 3, 4, dtype=torch.float64).numpy()
    other = 0.0
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    return list_of_inputs

generated_inputs["torch.special.xlog1py_2"] = xlog1py_inputs()



def xlog1py_3_inputs():
    list_of_inputs = []
    
    input = torch.tensor([0.1, 1.5, 10.0], dtype=torch.float32).numpy()
    other = np.int32(2)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([[-1.0, 0.0, 1.0],
                          [2.0, -3.0, 4.0]], dtype=torch.float64).numpy()
    other = np.int64(-1)
    out = np.zeros_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.arange(6, dtype=torch.float32).view(1, 2, 3).numpy()
    other = np.int16(-3)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor(3.1415926535, dtype=torch.float64).numpy()
    other = np.int8(5)
    out = np.zeros_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.arange(6, dtype=torch.float32).view(2, 1, 3, 1).numpy()
    other = np.int16(0)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.arange(10, dtype=torch.float64)[::2].numpy()
    other = np.int64(1)
    out = np.zeros_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([0.0, float('inf'), -float('inf'), float('nan')], dtype=torch.float32).numpy()
    other = np.int32(-1)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.empty((0, 3), dtype=torch.float32).numpy()
    other = np.int16(5)
    out = np.zeros_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.arange(12, dtype=torch.float64).view(3, 4).t().numpy()
    other = np.int32(10)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([[[-1e-5, 2.0], [3.5, -4.2]]], dtype=torch.float32).numpy()
    other = np.int8(-2)
    out = np.zeros_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([1e-300, 1e300, -1e-100, -1e100], dtype=torch.float64).numpy()
    other = np.int64(1000000)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.empty((0,), dtype=torch.float64).numpy()
    other = np.int32(0)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    return list_of_inputs

generated_inputs["torch.special.xlog1py_3"] = xlog1py_3_inputs()



def xlog1py_inputs():
    list_of_inputs = []

    # Input 1
    input_val = 0.0
    other = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    out = np.empty_like(other)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    # Input 2
    input_val = 2.5
    other = np.array([[-0.5, 0.0], [1.5, 10.0]], dtype=np.float64)
    out = np.empty_like(other)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    # Input 3
    input_val = -3.0
    other = np.linspace(-0.9, 0.9, 7, dtype=np.float32)
    out = np.empty_like(other)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    # Input 4
    input_val = np.float64(1.0)
    other = torch.linspace(-0.99, 5.0, steps=12, dtype=torch.float64).reshape(3, 4).numpy()
    out = np.empty_like(other)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    # Input 5
    input_val = 1e-6
    other = np.array([0.0, -1e-4, 1e-4, 10.0, 1000.0], dtype=np.float16)
    out = np.empty_like(other)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    # Input 6
    input_val = 2.0
    other = np.array([-1.0, -0.999, -0.5, 0.5], dtype=np.float64)
    out = np.empty_like(other)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    # Input 7
    input_val = 12345.6789
    other = np.full((3, 3), 1e-8, dtype=np.float32)
    out = np.empty_like(other)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    # Input 8
    input_val = 1.5
    other = np.array([1e10, 1e20, 1e-10], dtype=np.float64)
    out = np.empty_like(other)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    # Input 9
    input_val = np.float32(-2.5)
    other = torch.tensor([[0.0, 0.1, 0.2]], dtype=torch.float32).numpy()
    out = np.empty_like(other)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    # Input 10
    input_val = 3.14159
    other = np.array(0.5, dtype=np.float64)
    out = np.empty((), dtype=other.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    # Input 11
    input_val = -7.0
    base = np.arange(12, dtype=np.float32).reshape(4, 3)
    other = base[:, ::-1]
    out = np.empty_like(other)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    # Input 12
    input_val = 0.75
    rng = np.random.RandomState(0)
    other = rng.uniform(-0.9, 2.0, size=(1, 2, 1, 3)).astype(np.float32)
    out = np.empty_like(other)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    return list_of_inputs

generated_inputs["torch.special.xlog1py_4"] = xlog1py_inputs()



def xlog1py_inputs():
    list_of_inputs = []

    input_val = np.int32(3)
    other = np.array([0.0, 1.0, -0.5], dtype=np.float32)
    out = np.empty_like(other, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    input_val = np.int64(-2)
    other = np.array([[1.0, 2.0], [3.0, -0.9]], dtype=np.float64)
    out = np.empty_like(other, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    input_val = np.int16(0)
    other = np.array(2.5, dtype=np.float64)
    out = np.empty((), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    input_val = np.int8(1)
    other = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = np.zeros_like(other, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    rng = np.random.RandomState(0)
    input_val = np.int64(7)
    other = rng.randn(3, 1, 4).astype(np.float64)
    out = np.empty_like(other, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    input_val = np.int32(-5)
    other = np.array([-1.5, -1.0, -0.999, 0.0, 1.0], dtype=np.float64)
    out = np.empty_like(other, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    input_val = np.int64(2)
    other = np.array([[True, False], [False, True]], dtype=bool)
    out = np.empty_like(other, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    input_val = np.int64(4)
    other = np.array([10, 0, -1, 3], dtype=np.int32)
    out = np.empty_like(other, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    input_val = np.int64(1)
    other = np.linspace(-0.9999999, 100.0, 8, dtype=np.float64)
    out = np.empty_like(other, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    input_val = np.int64(-1)
    other = np.zeros((0,), dtype=np.float32)
    out = np.empty_like(other, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    input_val = np.int64(123456789)
    other = np.full((2, 2, 2), 1e-12, dtype=np.float64)
    out = np.empty_like(other, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    input_val = np.int64(3)
    other = np.array([np.inf, -np.inf, np.nan], dtype=np.float64)
    out = np.empty_like(other, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))

    return list_of_inputs

generated_inputs["torch.special.xlog1py_5"] = xlog1py_inputs()



def sqrt_inplace_inputs():
    list_of_inputs = []

    input = np.array([0.0, 1.0, 4.0, 9.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.array([[-1.0, 0.0, 1.0],
                      [2.0, -3.0, 16.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.array([[[0.25, 1.0, 4.0],
                       [9.0, 16.0, 25.0]],
                      [[36.0, 49.0, 64.0],
                       [81.0, 100.0, 121.0]]], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.array(-4.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.array([], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.arange(12, dtype=np.float32).reshape(3, 4).T
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.arange(24, dtype=np.float64).reshape(4, 6)[:, ::2]
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.array([np.inf, -np.inf, np.nan, 0.0, 100.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.array([1+0j, -1+0j, 3-4j, 0+0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.array([[-1+1j, 2-3j],
                      [4+0j, 0+5j]], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = (np.arange(12, dtype=np.float32).reshape(2, 1, 3, 1, 2) / 3.0) + 0.1
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.array([1e-45, 1e-38, 1e-20, 1e-10, 1e-5, 1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    return list_of_inputs

generated_inputs["torch.sqrt_"] = sqrt_inplace_inputs()



def take_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = torch.tensor([[4, 3, 5], [6, 7, 8]]).numpy()
    index_arr = torch.tensor([0, 2, 5], dtype=torch.long).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "index": index_arr}))

    # Input 2
    input_arr = torch.arange(12, dtype=torch.int32).reshape(2, 2, 3).numpy()
    index_arr = torch.tensor([[0, 7], [5, 10]], dtype=torch.long).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "index": index_arr}))

    # Input 3
    input_arr = torch.tensor([-1.0, -2.5, 3.3, 0.0, 4.4], dtype=torch.float32).numpy()
    index_arr = torch.tensor([0, 3, 4], dtype=torch.long).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "index": index_arr}))

    # Input 4
    input_arr = torch.tensor([[True, False], [False, True]], dtype=torch.bool).numpy()
    index_arr = torch.tensor([0, 3, 1, 2], dtype=torch.long).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "index": index_arr}))

    # Input 5
    input_arr = torch.tensor(
        [[1+2j, 3-4j, 5+0j],
         [0+1j, -2-2j, 7+7j],
         [9+0j, -1+1j, 2-3j]], dtype=torch.complex64
    ).numpy()
    index_arr = torch.tensor([8, 0, 4, 2, 6], dtype=torch.long).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "index": index_arr}))

    # Input 6
    input_arr = torch.tensor([1.5, -2.5, 3.0, 4.5], dtype=torch.float16).numpy()
    index_arr = torch.tensor(1, dtype=torch.long).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "index": index_arr}))

    # Input 7
    input_arr = torch.arange(120, dtype=torch.float32).reshape(2, 3, 4, 5).numpy()
    index_arr = torch.tensor([0, 59, 119, 24, 75], dtype=torch.long).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "index": index_arr}))

    # Input 8
    input_arr = torch.empty((0,), dtype=torch.float32).numpy()
    index_arr = torch.empty((0,), dtype=torch.long).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "index": index_arr}))

    # Input 9
    input_arr = torch.empty((1, 0, 2), dtype=torch.float32).numpy()
    index_arr = torch.empty((2, 0), dtype=torch.long).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "index": index_arr}))

    # Input 10
    input_arr = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.uint8).numpy()
    index_arr = torch.tensor([0, 1, 2, 3, 4, 7], dtype=torch.long).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "index": index_arr}))

    # Input 11
    input_arr = torch.linspace(0, 1, steps=5, dtype=torch.float64).numpy()
    index_arr = torch.tensor([[0, 2, 4], [1, 3, 0]], dtype=torch.long).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "index": index_arr}))

    # Input 12
    input_arr = torch.tensor(3+4j, dtype=torch.complex128).numpy()
    index_arr = torch.tensor([0], dtype=torch.long).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "index": index_arr}))

    return list_of_inputs

generated_inputs["torch.take"] = take_inputs()

