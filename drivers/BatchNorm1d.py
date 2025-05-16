import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    num_features = input["num_features"]
    eps = input.get("eps", 1e-05)
    momentum = input.get("momentum", 0.1)
    affine = input.get("affine", True)
    track_running_stats = input.get("track_running_stats", True)

    # Create BatchNorm1d instance
    batch_norm = torch.nn.BatchNorm1d(num_features=num_features,
                                      eps=eps, momentum=momentum, affine=affine,
                                      track_running_stats=track_running_stats)

    # Get input tensor
    input_tensor = torch.tensor(input["data"])
    
    # Move to GPU if not using CPU
    if not cpu:
        input_tensor = input_tensor.cuda()
        batch_norm = batch_norm.cuda()

    # Apply the batch normalization layer
    output = batch_norm(input_tensor)

    if not cpu:
        output = output.cpu()

    return {"batch_norm_output": output.detach().numpy()}

### TensorFlow Implementation
def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        num_features = input["num_features"]
        eps = input.get("eps", 1e-05)
        momentum = input.get("momentum", 0.1)
        affine = input.get("affine", True)
        trainable = affine
        track_running_stats = input.get("track_running_stats", True)
        
        # Create BatchNormalization instance
        batch_norm = tf.keras.layers.BatchNormalization(
            axis=1,  # Along the features dimension
            momentum=momentum,
            epsilon=eps,
            center=affine,
            scale=affine,
            trainable=trainable
        )

        # Get input tensor
        input_tensor = tf.constant(input["data"])

        # Apply the batch normalization layer
        output = batch_norm(input_tensor, training=track_running_stats)

        output_np = output.numpy() if cpu else output.cpu().numpy()

        return {"batch_norm_output": output_np}

def main():
    # Example input
    input_data = {
        "num_features": 3,
        "data": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert to compare results
    if np.allclose(torch_result["batch_norm_output"], tf_result["batch_norm_output"], atol=1e-06):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()