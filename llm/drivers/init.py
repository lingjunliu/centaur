import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    
    tensor = torch.tensor(input_dict["tensor"])
    gain = input_dict.get("gain", 1.0)

    if not cpu:
        tensor = tensor.cuda()

    torch.nn.init.xavier_uniform_(tensor, gain=gain)

    if not cpu:
        tensor = tensor.cpu()

    return {"result": tensor.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    tensor_np = input_dict["tensor"]
    gain = input_dict.get("gain", 1.0)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        tensor = tf.Variable(tensor_np, dtype=tf.float32)
        shape = tensor.shape
        fan_in = np.prod(shape[:-1]) if len(shape) > 1 else 1
        fan_out = shape[-1] if len(shape) > 0 else 1

        if len(shape) <= 1:
            n = fan_in
        else:
            n = (fan_in + fan_out)

        std = gain * np.sqrt(2.0 / n) if n > 0 else 0.
        limit = np.sqrt(3.0) * std

        new_val = tf.random.uniform(shape, minval=-limit, maxval=limit)
        tensor.assign(new_val)
        
        result = tensor.numpy()
        
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "tensor": np.random.rand(3, 4).astype(np.float32),
        "gain": 2.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()