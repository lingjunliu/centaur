import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    pad = input_dict["pad"]
    value = input_dict.get("value", 0.0)

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.nn.functional.pad(input_tensor, pad, mode='constant', value=value)
    
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
        pad = input_dict["pad"]
        value = input_dict.get("value", 0.0)

        rank = len(input_tensor.shape)
        paddings = []
        
        if rank == 1:
            paddings = [[pad[0], pad[1]]]
        elif rank == 2:
            paddings = [[pad[2], pad[3]], [pad[0], pad[1]]]
        elif rank == 3:
            paddings = [[pad[4], pad[5]], [pad[2], pad[3]], [pad[0], pad[1]]]
        elif rank == 4:
            paddings = [[pad[6], pad[7]], [pad[4], pad[5]], [pad[2], pad[3]], [pad[0], pad[1]]]
        elif rank == 5:
            paddings = [[pad[8], pad[9]], [pad[6], pad[7]], [pad[4], pad[5]], [pad[2], pad[3]], [pad[0], pad[1]]]

        result = tf.pad(input_tensor, paddings, "CONSTANT", constant_values=value)

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "pad": [1, 1, 2, 2],
        "value": 5.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()