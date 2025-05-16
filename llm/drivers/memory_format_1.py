import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    memory_format_str = input_dict.get("memory_format", "contiguous")
    if memory_format_str == "contiguous":
        memory_format = torch.contiguous_format
    elif memory_format_str == "channels_last":
        input_tensor = input_tensor.reshape(1, 1, input_tensor.shape[0], input_tensor.shape[1])
        memory_format = torch.channels_last
    else:
        raise ValueError(f"Unsupported memory format: {memory_format_str}")

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = input_tensor.contiguous(memory_format=memory_format)

    if not cpu:
        result = result.cpu()

    if memory_format_str == "channels_last":
        result = result.reshape(input_dict["input"].shape[0], input_dict["input"].shape[1])
        
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_np = input_dict["input"]
    input_tensor = tf.constant(input_np)
    memory_format = input_dict.get("memory_format", "contiguous")

    if memory_format == "channels_last":
        input_tensor = tf.transpose(input_tensor, perm=[1, 0])

    result = input_tensor.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "memory_format": "channels_last"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"].transpose(), atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()