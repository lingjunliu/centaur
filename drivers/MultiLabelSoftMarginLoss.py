import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    target_tensor = torch.tensor(input["target"]) 
    weight_tensor = torch.tensor(input.get("weight", None)) if input.get("weight", None) is not None else None
    reduction = input.get("reduction", 'mean')

    # Apply to torch.nn.MultiLabelSoftMarginLoss
    loss_fn = torch.nn.MultiLabelSoftMarginLoss(weight=weight_tensor, reduction=reduction)
    loss = loss_fn(input_tensor, target_tensor)

    if not cpu:
        loss = loss.cpu()
    
    return {"MultiLabelSoftMarginLoss_loss": float(loss.item())}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        target_tensor = tf.constant(input["target"])
        weight_tensor = tf.constant(input.get("weight", None)) if input.get("weight", None) is not None else None

        # Apply to TensorFlow equivalent
        sig_log = tf.nn.sigmoid_cross_entropy_with_logits(labels=target_tensor, logits=input_tensor)
        
        if weight_tensor is not None:
            sig_log = sig_log * weight_tensor

        if input.get("reduction", 'mean') == 'sum':
            loss = tf.reduce_sum(sig_log)
        else: # 'mean' or other not explicitly allowed values will be handled as 'mean'
            loss = tf.reduce_mean(sig_log)

        return {"MultiLabelSoftMarginLoss_loss": float(loss.numpy())}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "target": np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32),  # Ensure target is float type
        "weight": None,
        "reduction": 'mean'
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    assert np.isclose(torch_result["MultiLabelSoftMarginLoss_loss"], tf_result["MultiLabelSoftMarginLoss_loss"], atol=1e-5), "Results are not equal"

    print("equal")

if __name__ == "__main__":
    main()