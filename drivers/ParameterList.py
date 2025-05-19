import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    parameter_list = input_dict["parameter_list"]
    
    torch_parameter_list = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(p)) for p in parameter_list])

    if not cpu:
        torch_parameter_list = torch.nn.ParameterList([torch.nn.Parameter(p.data.cuda()) for p in torch_parameter_list])

    result = [p.data.cpu().numpy() if not cpu else p.data.numpy() for p in torch_parameter_list]

    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    parameter_list = input_dict["parameter_list"]
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        tf_parameter_list = [tf.Variable(p) for p in parameter_list]
        
        result = [p.numpy() for p in tf_parameter_list]
    
    return {"result": result}

def main():
    A_TOL = 0.01
    # Example input
    input_data = {
        "parameter_list": [np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
                             np.array([1.0, 2.0, 3.0], dtype=np.float32)]
    }

    # Torch example
    torch_result = torch_version(input_data)
    
    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    
    # Assert to see if they are equal
    for i in range(len(torch_result["result"])):
        assert np.allclose(torch_result["result"][i], tf_result["result"][i], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()