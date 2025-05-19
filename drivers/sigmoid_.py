import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    torch.sigmoid_(input_tensor)
    
    if not cpu:
        input_tensor = input_tensor.cpu()
    
    return {"result": input_tensor.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        
        result = tf.sigmoid(input_tensor)
        result_np = result.numpy()
        input_dict["input"][:] = result_np[:]
    
    return {"result": input_dict["input"]}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    
    input_data_tf = {
        "input": np.copy(input_data["input"])
    }
    tf_result = tensorflow_version(input_data_tf)
    
    assert np.allclose(torch_result["result"], input_data_tf["input"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()