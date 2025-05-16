import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    conv = input_dict["conv"]
    relu = input_dict["relu"]
    
    with torch.no_grad():
        result = relu(conv(input_tensor.permute(0, 3, 1, 2)))
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        
        conv_weights = tf.constant(input_dict["conv"].weight.detach().numpy().transpose(0, 2, 3, 1))
        conv_bias = tf.constant(input_dict["conv"].bias.detach().numpy())

        padding = 'VALID'
        conv = tf.nn.conv2d(input_tensor, conv_weights, strides=[1, 1, 1, 1], padding=padding)
        
        result = tf.nn.relu(conv + conv_bias)
        
        result = result.numpy()
        
    return {"result": result}

def main():
    A_TOL = 0.01
    
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    
    conv = torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=0)
    relu = torch.nn.ReLU()
    
    input_data = {
        "input": np.random.rand(1, 32, 32, 3).astype(np.float32),
        "conv": conv,
        "relu": relu
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()