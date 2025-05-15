import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    size = input_dict["size"]
    stride = input_dict["stride"]
    storage_offset = input_dict.get("storage_offset", 0)

    if not cpu:
        input_tensor = input_tensor.cuda()

    input_tensor.as_strided_(size, stride, storage_offset)
    
    if not cpu:
        input_tensor = input_tensor.cpu()

    return {"result": input_tensor.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_np = input_dict["input"]
    size = input_dict["size"]
    stride = input_dict["stride"]
    storage_offset = input_dict.get("storage_offset", 0)
    
    input_tf = tf.constant(input_np)
    
    input_np_reshaped = input_np[storage_offset:]
    
    def calculate_indices(size, stride):
        indices = []
        for i in range(size[0]):
            if len(size) == 1:
                indices.append(i * stride[0])
            elif len(size) == 2:
                for j in range(size[1]):
                    indices.append(i * stride[0] + j * stride[1])
            elif len(size) == 3:
                for j in range(size[1]):
                    for k in range(size[2]):
                        indices.append(i * stride[0] + j * stride[1] + k * stride[2])
            else:
                raise NotImplementedError("Only supports up to 3 dimensions")
        return indices

    indices = calculate_indices(size, stride)

    result_np = input_np_reshaped.take(indices)
    result_np = result_np.reshape(size)
    
    return {"result": result_np}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.arange(12, dtype=np.float32).reshape(12),
        "size": (2, 2),
        "stride": (1, 2),
        "storage_offset": 0,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.arange(12, dtype=np.float32).reshape(12),
        "size": (3, 2),
        "stride": (2, 1),
        "storage_offset": 1,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()