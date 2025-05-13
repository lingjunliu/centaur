import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    chunks = input_dict["chunks"]
    dim = input_dict.get("dim", 0)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.unsafe_chunk(input_tensor, chunks, dim=dim)
    
    if not cpu:
        result = [r.cpu() for r in result]
    
    return {"result": [r.numpy() for r in result]}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        chunks = input_dict["chunks"]
        dim = input_dict.get("dim", 0)

        length = tf.shape(input_tensor)[dim]
        chunk_size = length // chunks
        remainder = length % chunks

        result = []
        start = 0
        for i in range(chunks):
            chunk_len = chunk_size + (1 if i < remainder else 0)
            
            slices = [slice(None)] * len(input_tensor.shape)
            slices[dim] = slice(start, start + chunk_len)
            
            chunk = input_tensor[tuple(slices)]
            result.append(chunk.numpy())
            start += chunk_len
        
        # Handle the case where the dimension is 0 by transposing
        if dim == 0:
            new_result = []
            for chunk in result:
                new_result.append(chunk.T)
            result = new_result

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.float32).reshape(3,4),
        "chunks": 3,
        "dim": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    for i in range(len(torch_result["result"])):
        assert np.allclose(torch_result["result"][i], tf_result["result"][i], atol=A_TOL), f"Results do not match at index {i}, torch: {torch_result['result'][i]}, tf: {tf_result['result'][i]}"

    input_data = {
        "input": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.float32).reshape(4,3),
        "chunks": 2,
        "dim": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    tf_result["result"] = [r.T for r in tf_result["result"]]

    for i in range(len(torch_result["result"])):
        assert np.allclose(torch_result["result"][i], tf_result["result"][i], atol=A_TOL), f"Results do not match at index {i}, torch: {torch_result['result'][i]}, tf: {tf_result['result'][i]}"

    print("Success")

if __name__ == "__main__":
    main()