import numpy as np

# PyTorch version of InstanceNorm3d
def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    # Create InstanceNorm3d layer
    instance_norm = torch.nn.InstanceNorm3d(
        input["num_features"],
        eps=input.get("eps", 1e-05),
        momentum=input.get("momentum", 0.1),
        affine=input.get("affine", False),
        track_running_stats=input.get("track_running_stats", False)
    )

    if not cpu:
        instance_norm = instance_norm.cuda()
    
    # Apply InstanceNorm3d
    output = instance_norm(input_tensor)
    
    if not cpu:
        output = output.cpu()
    
    return {"instance_norm_output": output.detach().numpy()}

# TensorFlow equivalent using LayerNormalization
def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])

        # Create LayerNormalization with behavior similar to InstanceNorm3d
        instance_norm = tf.keras.layers.LayerNormalization(
            axis=(2, 3, 4),  # Normalize over depth, height, and width dimensions
            epsilon=input.get("eps", 1e-05)
        )

        # Apply LayerNormalization
        output = instance_norm(input_tensor)

        return {"instance_norm_output": output.numpy()}

def main():
    # Set a seed function for reproducibility
    def set_seed(seed=0):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Example input
    input_data = {
        "input": np.random.randn(20, 100, 35, 45, 10).astype(np.float32),
        "num_features": 100,
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": False,
        "track_running_stats": False
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result shape:", torch_result["instance_norm_output"].shape)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result shape:", tf_result["instance_norm_output"].shape)

    # Compare results
    try:
        np.testing.assert_allclose(torch_result["instance_norm_output"], tf_result["instance_norm_output"], rtol=1e-5, atol=1e-5)
        print("equal")
    except AssertionError:
        print("not equal")

if __name__ == "__main__":
    main()