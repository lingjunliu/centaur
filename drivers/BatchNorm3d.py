import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    num_features = input["num_features"]
    eps = input.get("eps", 1e-5)
    momentum = input.get("momentum", 0.1)
    affine = input.get("affine", True)
    track_running_stats = input.get("track_running_stats", True)
    
    bn_layer = torch.nn.BatchNorm3d(num_features, eps, momentum, affine, track_running_stats)
    
    input_tensor = torch.tensor(input["input"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        bn_layer = bn_layer.cuda()
    
    # Apply BatchNorm3d
    output_tensor = bn_layer(input_tensor)
    
    if not cpu:
        output_tensor = output_tensor.cpu()
    
    return {"BatchNorm3d_output": output_tensor.detach().numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    num_features = input["num_features"]
    eps = input.get("eps", 1e-5)
    momentum = input.get("momentum", 0.1)
    affine = input.get("affine", True)
    train = input.get("train", True)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input["input"])
        batch_tensor = tf.reshape(input_tensor, [-1] + list(input_tensor.shape[1:]))

        bn_layer = tf.keras.layers.BatchNormalization(axis=1, momentum=momentum, epsilon=eps, center=affine, scale=affine)
        
        if not train:
            # set trainable=False to use the moving statistics
            bn_layer.trainable = False

        output_tensor = bn_layer(batch_tensor, training=train)

        return {"BatchNorm3d_output": output_tensor.numpy()}

def main():
    np.random.seed(2023)
    input_data = {
        "input": np.random.randn(2, 3, 5, 5, 5).astype(np.float32),
        "num_features": 3,
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "train": True,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    torch_output = torch_result["BatchNorm3d_output"]
    tf_output = tf_result["BatchNorm3d_output"]

    assert np.allclose(torch_output, tf_output, atol=1e-5), "The outputs are not equal!"
    
    if np.allclose(torch_output, tf_output, atol=1e-5):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()