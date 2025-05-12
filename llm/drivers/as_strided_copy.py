import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    size = input_dict["size"]
    stride = input_dict["stride"]
    storage_offset = input_dict.get("storage_offset", 0)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.as_strided_copy(input_tensor, size, stride, storage_offset=storage_offset)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_np = input_dict["input"]
    size = input_dict["size"]
    stride = input_dict["stride"]
    storage_offset = input_dict.get("storage_offset", 0)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_np)
        
        flat_input = tf.reshape(input_tensor, [-1])

        indices = []
        for i in range(np.prod(size)):
            temp = i
            lin_index = 0
            
            multiplier = 1
            for j in range(len(size)):
                lin_index += (temp // multiplier % size[j]) * stride[j]
                multiplier *= size[j]
        
            indices.append(lin_index + storage_offset)
        

        indices = tf.constant(indices, dtype=tf.int32)

        result_flat = tf.gather(flat_input, indices)

        result = tf.reshape(result_flat, size)
        result = result.numpy()
    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.arange(12, dtype=np.float32).reshape(3, 4),
        "size": (2, 2),
        "stride": (4, 1),
        "storage_offset": 1
    }
    

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()