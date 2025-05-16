import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    module = input_dict["module"]

    if not cpu:
        input_tensor = input_tensor.cuda()
        module = module.cuda()

    result = module(input_tensor)

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
        module = input_dict["module"]
        
        if isinstance(module, tf.keras.layers.Conv2D):
            input_tensor = tf.transpose(input_tensor, perm=[1, 2, 0])
            input_tensor = tf.reshape(input_tensor, [1, input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2]])
            result = module(input_tensor)
            result = tf.reshape(result, [result.shape[1], result.shape[2], result.shape[3]])
            result = tf.transpose(result, perm=[2, 0, 1])

        elif isinstance(module, tf.keras.layers.Conv3D):
            input_tensor = tf.reshape(input_tensor, [1, input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]])
            result = module(input_tensor)
            result = tf.reshape(result, [result.shape[1], result.shape[2], result.shape[3], result.shape[4]])
        else: 
             result = module(input_tensor)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    example_input = np.random.rand(3, 32, 32).astype(np.float32)
    
    # Create a dummy torch.nn.intrinsic.modules module (e.g., a Conv2d)
    torch_module = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)
    torch_module.weight.data = torch.randn(8, 3, 3, 3)
    torch_module.bias.data = torch.randn(8)

    # Create a corresponding TensorFlow module
    tf_module = tf.keras.layers.Conv2D(8, kernel_size=3, padding='same', use_bias=True)
    tf_module.build(input_shape=(1, 32, 32, 3))
    tf_module.kernel.assign(torch_module.weight.data.numpy().transpose(2, 3, 1, 0))
    tf_module.bias.assign(torch_module.bias.data.numpy())
    

    input_data = {
        "input": example_input,
        "module": torch_module
    }

    torch_result = torch_version(input_data)
    
    input_data_tf = {
        "input": example_input,
        "module": tf_module
    }

    tf_result = tensorflow_version(input_data_tf)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()