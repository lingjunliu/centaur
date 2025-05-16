import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    indices_or_sections = input_dict["indices_or_sections"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.hsplit(input_tensor, indices_or_sections)

    if not cpu:
        result = [r.cpu() for r in result]

    return {"result": tuple(r.numpy() for r in result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    indices_or_sections = input_dict["indices_or_sections"]

    if isinstance(indices_or_sections, int):
        num_splits = indices_or_sections
        axis = 1 if len(input_tensor.shape) > 1 else 0
        length = input_tensor.shape[axis]
        if length % num_splits != 0:
            raise RuntimeError("torch.hsplit: indices_or_sections must evenly divide the split dimension")
        split_size = length // num_splits
        result = tf.split(input_tensor, num_or_size_splits=[split_size] * num_splits, axis=axis)
    else:
        indices = indices_or_sections
        axis = 1 if len(input_tensor.shape) > 1 else 0
        
        if axis == 0:
            indices = [0] + indices
            result = []
            for i in range(len(indices) - 1):
                result.append(input_tensor[indices[i]:indices[i+1]])
            result.append(input_tensor[indices[-1]:])

        else:
            indices = [0] + indices
            result = []
            for i in range(len(indices) - 1):
                result.append(input_tensor[:,indices[i]:indices[i+1]])
            result.append(input_tensor[:,indices[-1]:])

    return {"result": tuple(r.numpy() for r in result)}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.arange(16.0).reshape(4, 4).astype(np.float32),
        "indices_or_sections": 2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    for torch_res, tf_res in zip(torch_result["result"], tf_result["result"]):
        assert np.allclose(torch_res, tf_res, atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.arange(16.0).reshape(4, 4).astype(np.float32),
        "indices_or_sections": [3, 6]
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    for torch_res, tf_res in zip(torch_result["result"], tf_result["result"]):
        assert np.allclose(torch_res, tf_res, atol=A_TOL), "Results do not match"
        
    input_data = {
        "input": np.arange(4.0).astype(np.float32),
        "indices_or_sections": 2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    for torch_res, tf_res in zip(torch_result["result"], tf_result["result"]):
        assert np.allclose(torch_res, tf_res, atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()