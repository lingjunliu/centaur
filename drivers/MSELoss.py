import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    target_tensor = torch.tensor(input["target"]) 
    reduction = input.get("reduction", 'mean')
    size_average = input.get("size_average", True)
    reduce = input.get("reduce", True)

    # Apply to torch.nn.MSELoss
    criterion = torch.nn.MSELoss(reduction=reduction, size_average=size_average, reduce=reduce)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        target_tensor = target_tensor.cuda()
        criterion = criterion.cuda()

    loss = criterion(input_tensor, target_tensor)

    if not cpu:
        loss = loss.cpu()

    return {"mse_loss": float(loss.item())}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        target_tensor = tf.constant(input["target"])
        reduction = input.get("reduction", 'mean')

        # Apply to TensorFlow equivalent
        loss = tf.reduce_mean(tf.square(input_tensor - target_tensor))

        if reduction == 'sum':
            loss = tf.reduce_sum(loss)

        return {"mse_loss": float(loss.numpy())}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "target": np.array([[0.4, 0.3, 0.7], [0.1, 0.5, 0.8]], dtype=np.float32),  # Ensure target is float type
        "reduction": 'mean'
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Ensure results are equal
    torch_loss = np.array([torch_result["mse_loss"]])
    tf_loss = np.array([tf_result["mse_loss"]])

    if np.allclose(torch_loss, tf_loss, atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()