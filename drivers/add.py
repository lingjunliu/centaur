import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack inputs from dictionary
    input_tensor = torch.tensor(input_dict["input"])
    other_tensor = torch.tensor(input_dict["other"])
    alpha = input_dict.get("alpha", 1.0)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        other_tensor = other_tensor.cuda()
    
    # Perform torch addition
    result = torch.add(input_tensor, other_tensor, alpha=alpha)
    
    # Move result to CPU for consistent return format
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
        # Unpack inputs from dictionary
        input_tensor = tf.constant(input_dict["input"])
        other_tensor = tf.constant(input_dict["other"])
        alpha = input_dict.get("alpha", 1.0)
        
        # Perform TensorFlow addition
        result = tf.add(input_tensor, tf.multiply(other_tensor, alpha))
        
        result = result.numpy()
    
    return {"result": result}

def main():
    # Example input for case 1
    input_data_case1 = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "other": 20.0,
        "alpha": 1
    }

    # Example input for case 2
    input_data_case2 = {
        "input": np.array([[-0.9732, -0.3497, 0.6245, 0.4022], [0.3743, -1.7724, -0.5811, -0.8017]], dtype=np.float32),
        "other": 10.0,
        "alpha": 10
    }

    # Torch example for case 1
    torch_result_case1 = torch_version(input_data_case1)
    print("Torch result case 1:", torch_result_case1)

    # TensorFlow example for case 1
    tf_result_case1 = tensorflow_version(input_data_case1)
    print("TensorFlow result case 1:", tf_result_case1)

    # Assert for case 1
    assert np.allclose(torch_result_case1["result"], tf_result_case1["result"]), "Results do not match for case 1"
    print("Results equal for case 1") if np.allclose(torch_result_case1["result"], tf_result_case1["result"]) else print("Results not equal for case 1")

    # Torch example for case 2
    torch_result_case2 = torch_version(input_data_case2)
    print("Torch result case 2:", torch_result_case2)

    # TensorFlow example for case 2
    tf_result_case2 = tensorflow_version(input_data_case2)
    print("TensorFlow result case 2:", tf_result_case2)

    # Assert for case 2
    assert np.allclose(torch_result_case2["result"], tf_result_case2["result"]), "Results do not match for case 2"
    print("Results equal for case 2") if np.allclose(torch_result_case2["result"], tf_result_case2["result"]) else print("Results not equal for case 2")

if __name__ == "__main__":
    main()