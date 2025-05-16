import numpy as np

# Helper function to create TensorFlow equivalents for variables and gradients
def create_tf_variables_and_gradients(input_tensors):
    variables = [tf.Variable(tensor) for tensor in input_tensors]
    with tf.GradientTape() as tape:
        # Mock a simple loss for gradient calculation
        loss = sum(tf.reduce_sum(var ** 2) for var in variables)
    gradients = tape.gradient(loss, variables)
    return variables, gradients

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Convert input to list of torch tensors
    if cpu:
        input_tensors = [torch.tensor(t) for t in input["parameters"]]
    else:
        input_tensors = [torch.tensor(t).cuda() for t in input["parameters"]]
    
    for t in input_tensors:
        t.requires_grad = True  # Ensure that gradients will be tracked

    max_norm = input["max_norm"]
    norm_type = input.get("norm_type", 2.0)
    error_if_nonfinite = input.get("error_if_nonfinite", False)
    foreach = input.get("foreach", None)

    # Mock backprop to populate gradients
    mock_loss = sum(torch.sum(t ** 2) for t in input_tensors)
    mock_loss.backward()

    # Apply the clip_grad_norm_ function to PyTorch tensors
    torch.nn.utils.clip_grad_norm_(
        input_tensors, max_norm, norm_type=norm_type, error_if_nonfinite=error_if_nonfinite, foreach=foreach
    )

    if not cpu:
        input_tensors = [t.cpu() for t in input_tensors]

    return [t.grad.numpy() for t in input_tensors]

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Convert input to list of TensorFlow tensors
        input_tensors = [tf.constant(t) for t in input["parameters"]]
        
        # Create TensorFlow variables and gradients
        variables, gradients = create_tf_variables_and_gradients(input_tensors)

        max_norm = input["max_norm"]
        norm_type = input.get("norm_type", 2.0)

        # Calculate the total norm for the provided gradients
        total_norm = tf.linalg.global_norm(gradients)

        # Clip gradients if the total norm exceeds the maximum norm
        clip_scale = tf.minimum(1.0, max_norm / (total_norm + 1e-6))
        clipped_gradients = [grad * clip_scale for grad in gradients]

        # If error_if_nonfinite is set, ensure no non-finite values
        if input.get("error_if_nonfinite", False):
            for grad in clipped_gradients:
                if not np.all(np.isfinite(grad.numpy())):
                    raise ValueError("Non-finite gradients detected")

    return [g.numpy() for g in clipped_gradients]

def main():
    # Example input
    input_data = {
        "parameters": [np.random.rand(2, 3), np.random.rand(4, 5)],  # Example tensors
        "max_norm": 1.0,
        "norm_type": 2.0,
        "error_if_nonfinite": False
    }

    # Torch example
    torch_result = torch_version(input_data, cpu=True)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data, cpu=True)
    print("TensorFlow result:", tf_result)

    # Compare results
    all_equal = all(np.allclose(t, tf) for t, tf in zip(torch_result, tf_result))
    if all_equal:
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()