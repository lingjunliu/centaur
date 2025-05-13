import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    parameters = input_dict["parameters"]

    torch_parameters = [torch.tensor(p) for p in parameters]

    if not cpu:
        torch_parameters = [p.cuda() for p in torch_parameters]

    result = torch.nn.utils.parameters_to_vector(torch_parameters)

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
        tf_parameters = [tf.constant(p) for p in parameters]

        concatenated = tf.concat([tf.reshape(p, [-1]) for p in tf_parameters], axis=0)
        result = concatenated.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "parameters": [
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([5.0, 6.0], dtype=np.float32)
        ]
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()