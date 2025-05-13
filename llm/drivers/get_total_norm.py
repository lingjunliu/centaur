import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    parameters = input_dict["parameters"]
    norm_type = input_dict.get("norm_type", 2.0)

    if not cpu:
        parameters = [torch.tensor(p).cuda() for p in parameters]
    else:
        parameters = [torch.tensor(p) for p in parameters]
    
    result = torch.nn.utils.get_total_norm(parameters, norm_type=norm_type)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        parameters = input_dict["parameters"]
        norm_type = input_dict.get("norm_type", 2.0)

        parameters = [tf.constant(p) for p in parameters]

        norms = [tf.norm(p, ord=norm_type) for p in parameters]
        norm_tensor = tf.stack(norms)
        result = tf.norm(norm_tensor, ord=norm_type)

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "parameters": [
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([5.0, 6.0], dtype=np.float32)
        ],
        "norm_type": 2.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()