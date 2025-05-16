import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn.functional as F

    input_tensor = torch.tensor(input_dict["input"])
    normalized_shape = (input_tensor.shape[-1],)
    gamma = input_dict.get("gamma", 1.0)
    beta = input_dict.get("beta", 0.0)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        gamma = torch.tensor([gamma] * normalized_shape[0]).cuda()
        beta = torch.tensor([beta] * normalized_shape[0]).cuda()
    else:
        gamma = torch.tensor([gamma] * normalized_shape[0])
        beta = torch.tensor([beta] * normalized_shape[0])


    result = F.layer_norm(input_tensor, normalized_shape, weight=gamma, bias=beta)

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
        gamma = input_dict.get("gamma", 1.0)
        beta = input_dict.get("beta", 0.0)
        
        mean = tf.reduce_mean(input_tensor, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(input_tensor - mean), axis=-1, keepdims=True)
        epsilon = 1e-5
        normalized = (input_tensor - mean) / tf.sqrt(variance + epsilon)
        
        result = gamma * normalized + beta
    
    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "gamma": 2.0,
        "beta": 1.0,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()