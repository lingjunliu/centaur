
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def fmax_inputs():
    def broadcast_shape(s1, s2):
        r1 = list(s1)[::-1]
        r2 = list(s2)[::-1]
        out = []
        for i in range(max(len(r1), len(r2))):
            d1 = r1[i] if i < len(r1) else 1
            d2 = r2[i] if i < len(r2) else 1
            if d1 == 1:
                out.append(d2)
            elif d2 == 1:
                out.append(d1)
            elif d1 == d2:
                out.append(d1)
            else:
                raise ValueError("Shapes not broadcastable")
        return tuple(out[::-1])

    list_of_inputs = []

    # 1) 1D float64 with NaNs
    t1 = torch.tensor([9.7, float('nan'), 3.1, float('nan')], dtype=torch.float64)
    t2 = torch.tensor([-2.2, 0.5, float('nan'), float('nan')], dtype=torch.float64)
    out_shape = broadcast_shape(t1.shape, t2.shape)
    out = torch.empty(out_shape, dtype=torch.result_type(t1, t2)).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": t1.numpy(),
        "other": t2.numpy(),
        "out": out
    }))

    # 2) 2D float32 vs scalar float32
    t1 = (torch.arange(6, dtype=torch.float32).reshape(2, 3) - 3.0)
    t2 = torch.tensor(-1.5, dtype=torch.float32)
    out_shape = broadcast_shape(t1.shape, t2.shape)
    out = torch.empty(out_shape, dtype=torch.result_type(t1, t2)).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": t1.numpy(),
        "other": t2.numpy(),
        "out": out
    }))

    # 3) 3D broadcast with NaNs and Infs (float32)
    t1 = torch.tensor([1.0, float('nan'), 3.0, -4.0, 5.0, float('nan')], dtype=torch.float32).reshape(2, 1, 3)
    t2 = torch.tensor([0.0, -2.0, float('inf'), float('nan')], dtype=torch.float32).reshape(1, 4, 1)
    out_shape = broadcast_shape(t1.shape, t2.shape)
    out = torch.empty(out_shape, dtype=torch.result_type(t1, t2)).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": t1.numpy(),
        "other": t2.numpy(),
        "out": out
    }))

    # 4) Integer tensors with different dtypes (int64 vs int32)
    t1 = torch.tensor([-10, 0, 10], dtype=torch.int64)
    t2 = torch.tensor([5, -20, 15], dtype=torch.int32)
    out_shape = broadcast_shape(t1.shape, t2.shape)
    out = torch.empty(out_shape, dtype=torch.result_type(t1, t2)).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": t1.numpy(),
        "other": t2.numpy(),
        "out": out
    }))

    # 5) Mixed int32 and float64, 2D
    t1 = torch.tensor([[1, -2], [3, -4]], dtype=torch.int32)
    t2 = torch.tensor([[0.5, 2.5], [-3.5, 4.5]], dtype=torch.float64)
    out_shape = broadcast_shape(t1.shape, t2.shape)
    out = torch.empty(out_shape, dtype=torch.result_type(t1, t2)).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": t1.numpy(),
        "other": t2.numpy(),
        "out": out
    }))

    # 6) Handling infinities and NaNs (float32)
    t1 = torch.tensor([float('inf'), float('-inf'), 0.0, float('nan')], dtype=torch.float32)
    t2 = torch.tensor([1.0, 2.0, float('inf'), float('-inf')], dtype=torch.float32)
    out_shape = broadcast_shape(t1.shape, t2.shape)
    out = torch.empty(out_shape, dtype=torch.result_type(t1, t2)).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": t1.numpy(),
        "other": t2.numpy(),
        "out": out
    }))

    # 7) 3D float64 broadcasting to (3,5,4)
    t1 = torch.linspace(-1, 1, steps=12, dtype=torch.float64).reshape(3, 1, 4)
    t2 = torch.tensor([[-2.0], [0.0], [2.0], [float('nan')], [1.5]], dtype=torch.float64).reshape(1, 5, 1)
    out_shape = broadcast_shape(t1.shape, t2.shape)
    out = torch.empty(out_shape, dtype=torch.result_type(t1, t2)).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": t1.numpy(),
        "other": t2.numpy(),
        "out": out
    }))

    # 8) 4D integer broadcasting (int32 vs int64)
    t1 = torch.arange(2 * 3 * 1 * 4, dtype=torch.int32).reshape(2, 3, 1, 4) - 5
    t2 = (torch.arange(3 * 5, dtype=torch.int64).reshape(1, 3, 5, 1) - 2)
    out_shape = broadcast_shape(t1.shape, t2.shape)
    out = torch.empty(out_shape, dtype=torch.result_type(t1, t2)).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": t1.numpy(),
        "other": t2.numpy(),
        "out": out
    }))

    # 9) Zero-sized dimension (float32)
    t1 = torch.empty((0, 3), dtype=torch.float32)
    t2 = torch.tensor([[1.0, -1.0, float('nan')]], dtype=torch.float32)  # shape (1,3)
    out_shape = broadcast_shape(t1.shape, t2.shape)
    out = torch.empty(out_shape, dtype=torch.result_type(t1, t2)).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": t1.numpy(),
        "other": t2.numpy(),
        "out": out
    }))

    # 10) Non-contiguous input via slicing (float32)
    base = torch.arange(20, dtype=torch.float32).reshape(4, 5) - 10.0
    t1 = base[:, ::2]  # shape (4,3), non-contiguous
    t2 = torch.tensor([[-5.0, 0.0, 5.0]], dtype=torch.float32)  # shape (1,3)
    out_shape = broadcast_shape(t1.shape, t2.shape)
    out = torch.empty(out_shape, dtype=torch.result_type(t1, t2)).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": t1.numpy(),
        "other": t2.numpy(),
        "out": out
    }))

    # 11) int16 with scalar broadcasting
    t1 = torch.tensor([[-32768, -1, 0, 1, 32767]], dtype=torch.int16)
    t2 = torch.tensor(7, dtype=torch.int16)  # scalar
    out_shape = broadcast_shape(t1.shape, t2.shape)
    out = torch.empty(out_shape, dtype=torch.result_type(t1, t2)).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": t1.numpy(),
        "other": t2.numpy(),
        "out": out
    }))

    # 12) Both scalars (float64) with NaN
    t1 = torch.tensor(float('nan'), dtype=torch.float64)
    t2 = torch.tensor(-3.0, dtype=torch.float64)
    out_shape = broadcast_shape(t1.shape, t2.shape)
    out = torch.empty(out_shape, dtype=torch.result_type(t1, t2)).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": t1.numpy(),
        "other": t2.numpy(),
        "out": out
    }))

    return list_of_inputs

generated_inputs["torch.fmax"] = fmax_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.fmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fmax'.")


check_valid('torch.fmax', generated_inputs['torch.fmax'], lib="torch", suffix=0)
