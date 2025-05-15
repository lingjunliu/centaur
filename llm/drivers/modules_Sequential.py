import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    modules = input_dict["modules"]

    if not cpu:
        modules = [module.cuda() for module in modules]

    input_tensor = torch.tensor(input_dict["input"], requires_grad=False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    seq = torch.nn.Sequential(*modules)
    result = seq(input_tensor)

    if not cpu:
        result = result.cpu()
    
    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    modules = input_dict["modules"]

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        
        x = input_tensor
        for module in modules:
            if isinstance(module, type(torch.nn.Linear(1, 1))):
                w = tf.constant(module.weight.detach().numpy().T, dtype=tf.float32)
                b = tf.constant(module.bias.detach().numpy(), dtype=tf.float32)
                x = tf.matmul(tf.reshape(x, (-1, w.shape[0])), w) + b
            elif isinstance(module, type(torch.nn.ReLU())):
                x = tf.nn.relu(x)
            elif isinstance(module, type(torch.nn.Sigmoid())):
                x = tf.nn.sigmoid(x)
            elif isinstance(module, type(torch.nn.Tanh())):
                x = tf.nn.tanh(x)
            else:
                raise Exception("Not implemented")
        result = x.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    # Example input
    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "modules": [torch.nn.Linear(4, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2)]
    }

    # Torch example
    torch_result = torch_version(input_data)
    
    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    
    # Assert to see if they are equal
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    import torch
    main()