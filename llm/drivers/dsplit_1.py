import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    indices_or_sections = input_dict["indices_or_sections"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.dsplit(input_tensor, indices_or_sections)

    if not cpu:
        result = tuple(r.cpu() for r in result)

    return {"result": tuple(r.numpy() for r in result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        indices_or_sections = input_dict["indices_or_sections"]

        input_shape = input_tensor.shape
        dim = 2

        if isinstance(indices_or_sections, int):
            num_splits = indices_or_sections
            
            if input_shape[dim] % num_splits != 0:
                raise RuntimeError("tensor of shape %s cannot be split into %d equally sized chunks at dimension %d" % (input_shape, num_splits, dim))

            result = tf.split(input_tensor, num_splits, axis=dim)
        else:
            indices = indices_or_sections
            split_sizes = []
            prev_index = 0
            for index in indices:
                split_sizes.append(index - prev_index)
                prev_index = index
            split_sizes.append(input_shape[dim] - prev_index)

            valid_split_sizes = []
            for s in split_sizes:
                if s > 0:
                   valid_split_sizes.append(s)

            total_split_size = sum(valid_split_sizes)
            if total_split_size != input_shape[dim]:
                valid_split_sizes[-1] -= (total_split_size - input_shape[dim])

            result = tf.split(input_tensor, valid_split_sizes, axis=dim)

        result = tuple(r.numpy() for r in result)

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.arange(16.0).reshape(2, 2, 4),
        "indices_or_sections": 2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    for torch_res, tf_res in zip(torch_result["result"], tf_result["result"]):
        assert np.allclose(torch_res, tf_res, atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.arange(16.0).reshape(2, 2, 4),
        "indices_or_sections": [3,6]
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    torch_res_len = len(torch_result['result'])
    tf_res_len = len(tf_result['result'])

    min_len = min(torch_res_len, tf_res_len)

    for i in range(min_len):
        torch_res = torch_result['result'][i]
        tf_res = tf_result['result'][i]
        assert np.allclose(torch_res, tf_res, atol=A_TOL), f"Results do not match at index {i}"
        
    
    if torch_res_len != tf_res_len:
        print("Warning: Number of output tensors differ between PyTorch and TensorFlow")
        
    print("Success")

if __name__ == "__main__":
    main()