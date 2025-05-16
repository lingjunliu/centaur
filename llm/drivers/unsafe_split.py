import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    split_size = input_dict["split_size"]
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    if isinstance(split_size, int):
      result = torch.unsafe_split(input_tensor, split_size)
    else:
      indices = [0]
      current_index = 0
      for size in split_size[:-1]:
        current_index += size
        indices.append(current_index)
      result = torch.split(input_tensor, split_size, dim=0)
    
    if not cpu:
        result = [r.cpu() for r in result]
    
    return {"result": [r.numpy() for r in result]}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        split_size = input_dict["split_size"]

        if isinstance(split_size, int):
            num_splits = int(np.ceil(input_tensor.shape[0] / split_size))
            
            last_split_size = input_tensor.shape[0] % split_size
            if last_split_size == 0:
                chunk_sizes = [split_size] * num_splits
            else:
                chunk_sizes = [split_size] * (num_splits - 1) + [last_split_size]
            result = tf.split(input_tensor, chunk_sizes, axis=0)

        elif isinstance(split_size, list):
            result = tf.split(input_tensor, split_size, axis=0)
    
        result = [r.numpy() for r in result]
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32),
        "split_size": 3
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert len(torch_result["result"]) == len(tf_result["result"]), "Different number of splits"
    for i in range(len(torch_result["result"])):
        assert np.allclose(torch_result["result"][i], tf_result["result"][i], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32),
        "split_size": [2, 2, 3, 3]
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert len(torch_result["result"]) == len(tf_result["result"]), "Different number of splits"
    for i in range(len(torch_result["result"])):
        assert np.allclose(torch_result["result"][i], tf_result["result"][i], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()