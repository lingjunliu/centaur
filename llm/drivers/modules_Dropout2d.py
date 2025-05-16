import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    p = input_dict.get("p", 0.5)
    training = input_dict.get("training", True)
    inplace = input_dict.get("inplace", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        
    dropout = torch.nn.Dropout2d(p=p, inplace=inplace)
    dropout.train(training)
    result = dropout(input_tensor)

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
        p = input_dict.get("p", 0.5)
        training = input_dict.get("training", True)
        
        if training:
            rate = p
            output = tf.nn.dropout(input_tensor, rate=rate, noise_shape=(input_tensor.shape[0],1,1,input_tensor.shape[3]))
        else:
            output = input_tensor

        result = output.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "p": 0.2,
        "training": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL)

    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "p": 0.2,
        "training": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL)
    
    print("Success")

if __name__ == "__main__":
    main()