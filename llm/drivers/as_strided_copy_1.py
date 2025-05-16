import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    size = input_dict["size"]
    stride = input_dict["stride"]
    storage_offset = input_dict.get("storage_offset", 0)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.as_strided(input_tensor, size, stride, storage_offset=storage_offset)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_np = input_dict["input"]
    size = input_dict["size"]
    stride = input_dict["stride"]
    storage_offset = input_dict.get("storage_offset", 0)

    input_tensor = tf.constant(input_np)

    
    if len(input_np.shape) != 1:
         raise ValueError("TensorFlow version only supports 1D input tensors.")

    total_elements = np.prod(size)
    
    output_np = np.zeros(size, dtype=input_np.dtype)

    for i in np.ndindex(*size):
        linear_index = 0
        for j in range(len(i)):
            linear_index += i[j] * stride[j]
        
        source_index = linear_index + storage_offset

        if source_index < 0 or source_index >= input_np.size:
            output_np[i] = 0  
        else:
            output_np[i] = input_np.flatten()[source_index]
    
    result = output_np
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.arange(1, 7, dtype=np.float32),
        "size": (2, 2),
        "stride": (1, 2),
        "storage_offset": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()