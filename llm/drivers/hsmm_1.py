import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

def torch_version(input_dict, cpu=True):
    input_tensor = torch.tensor(input_dict["input"])
    lengths = torch.tensor(input_dict["lengths"])
    alpha = input_dict.get("alpha", 1.0)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        lengths = lengths.cuda()
    
    result = torch.flip(input_tensor[:, :torch.max(lengths)], dims=[1])
    
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
        lengths = tf.constant(input_dict["lengths"])
        alpha = input_dict.get("alpha", 1.0)

        max_length = tf.reduce_max(lengths)
        
        cropped_tensor = input_tensor[:, :max_length]
        
        reversed_tensor = tf.reverse(cropped_tensor, axis=[1])

        result = reversed_tensor.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0], [7.0, 0.0, 0.0]], dtype=np.float32),
        "lengths": np.array([3, 2, 1], dtype=np.int32),
        "alpha": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()