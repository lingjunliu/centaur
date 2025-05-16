import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    src = torch.tensor(input_dict["src"])
    permutation = torch.tensor(input_dict["permutation"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        src = src.cuda()
        permutation = permutation.cuda()
    
    result = torch.transpose(src, 0, 1)

    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        src = tf.constant(input_dict["src"])
        permutation = tf.cast(input_dict["permutation"], dtype=tf.int32)
        
        result = tf.transpose(src)

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(2, 3, 4).astype(np.float32),
        "src": np.random.rand(4, 2).astype(np.float32),
        "permutation": np.array([1, 0])
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()