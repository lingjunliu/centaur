import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    other_tensor = input_dict["other"]
    alpha = input_dict.get("alpha", 1.0)

    if isinstance(other_tensor, np.ndarray):
        other_tensor = torch.tensor(other_tensor)
    elif not isinstance(other_tensor, torch.Tensor):
        other_tensor = torch.tensor(np.array(other_tensor))

    if not cpu:
        input_tensor = input_tensor.cuda()
        if isinstance(other_tensor, torch.Tensor):
            other_tensor = other_tensor.cuda()

    result = torch.xlogy_(input_tensor, other_tensor, alpha=alpha)

    if not cpu:
        result = result.cpu()

    return {'result': result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        other_tensor = tf.constant(input_dict["other"])
        alpha = tf.constant(input_dict.get("alpha", 1.0))
        
        result = alpha * input_tensor * tf.math.log(other_tensor)
        
        result = result.numpy()

    return {'result': result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "other": 20.0,
        "alpha": 1.0
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()