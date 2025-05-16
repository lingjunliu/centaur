import numpy as np

def torch_var_dim(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    dim = input["dim"]
    correction = 1
    if "unbiased" in input.keys():
        unbiased = input["unbiased"]
        if not unbiased:
            correction = 0
    elif "correction" in input.keys():
        correction = input["correction"]
        
    keepdim = input.get("keepdim", False)

    # Apply torch.var
    var_result = torch.var(input_tensor, dim=dim, correction=correction, keepdim=keepdim)

    if not cpu:
        var_result = var_result.cpu()

    return {"var_result": var_result.numpy()}

def tensorflow_var_dim(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input["input"])
        dim = input["dim"]
        correction = 1
        if "unbiased" in input.keys():
            unbiased = input["unbiased"]
            correction = 1 if unbiased else 0
        elif "correction" in input.keys():
            correction = input["correction"]
            
        keepdim = input.get("keepdim", False)

        mean_tensor = tf.reduce_mean(input_tensor, axis=dim, keepdims=True)
        squared_diff = tf.square(input_tensor - mean_tensor)
        var_result = tf.reduce_sum(squared_diff, axis=dim, keepdims=keepdim) / (tf.cast(tf.shape(input_tensor)[dim] - correction, tf.float32))

        return {"var_result": var_result.numpy()}

#### 2. Torch and TensorFlow Functions for `torch.var(input, unbiased)`

def torch_var_unbiased(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    unbiased = input["unbiased"]

    # Apply torch.var
    var_result = torch.var(input_tensor, unbiased=unbiased)

    if not cpu:
        var_result = var_result.cpu()

    return {"var_result": var_result.numpy()}

def tensorflow_var_unbiased(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input["input"])
        unbiased = input["unbiased"]
        correction = 1 if unbiased else 0

        mean_tensor = tf.reduce_mean(input_tensor)
        squared_diff = tf.square(input_tensor - mean_tensor)
        var_result = tf.reduce_sum(squared_diff) / (tf.cast(tf.size(input_tensor) - correction, tf.float32))

        return {"var_result": var_result.numpy()}

def main():
    # Example input for var with dim, unbiased, keepdim
    input_data_dim = {
        "input": np.array([[0.2035, 1.2959, 1.8101, -0.4644],
                           [1.5027, -0.3270, 0.5905, 0.6538],
                           [-1.5745, 1.3330, -0.5596, -0.6548],
                           [0.1264, -0.5080, 1.6420, 0.1992]], dtype=np.float32),
        "dim": 1,
        "unbiased": True,
        "keepdim": True
    }

    # Example input for var with unbiased
    input_data_unbiased = {
        "input": np.array([[0.2035, 1.2959, 1.8101, -0.4644],
                           [1.5027, -0.3270, 0.5905, 0.6538],
                           [-1.5745, 1.3330, -0.5596, -0.6548],
                           [0.1264, -0.5080, 1.6420, 0.1992]], dtype=np.float32),
        "unbiased": True
    }

    # Torch example for var with dim
    torch_result_dim = torch_var_dim(input_data_dim)
    print("Torch result with dim:", torch_result_dim)

    # TensorFlow example for var with dim
    tf_result_dim = tensorflow_var_dim(input_data_dim)
    print("TensorFlow result with dim:", tf_result_dim)

    if np.allclose(torch_result_dim["var_result"], tf_result_dim["var_result"]):
        print("equal")
    else:
        print("not equal")

    # Torch example for var with unbiased
    torch_result_unbiased = torch_var_unbiased(input_data_unbiased)
    print("Torch result with unbiased:", torch_result_unbiased)

    # TensorFlow example for var with unbiased
    tf_result_unbiased = tensorflow_var_unbiased(input_data_unbiased)
    print("TensorFlow result with unbiased:", tf_result_unbiased)

    if np.allclose(torch_result_unbiased["var_result"], tf_result_unbiased["var_result"]):
        print("equal")
    else:
        print("not equal")
    
if __name__ == "__main__":
    main()
