import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    target_tensor = torch.tensor(input["target"])
    weight_tensor = torch.tensor(input.get("weight", None)) if input.get("weight", None) is not None else None
    reduction = input.get("reduction", 'mean')
    ignore_index = input.get("ignore_index", -100)

    if not cpu:
        if weight_tensor is not None:
            weight_tensor = weight_tensor.cuda()

    # Apply to torch.nn.NLLLoss
    nll_loss = torch.nn.NLLLoss(weight=weight_tensor, reduction=reduction, ignore_index=ignore_index)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        target_tensor = target_tensor.cuda()
        weight_tensor = weight_tensor.cuda()
        nll_loss = nll_loss.cuda()

    # LogSoftmax for input
    log_softmax = torch.nn.LogSoftmax(dim=1)
    if not cpu:
        log_softmax = log_softmax.cuda()
    input_tensor = log_softmax(input_tensor)

    loss = nll_loss(input_tensor, target_tensor)

    if not cpu:
        loss = loss.cpu()

    return {"nll_loss": float(loss.item())}

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
        target_tensor = tf.constant(input["target"], dtype=tf.int64)
        weight_tensor = tf.constant(input.get("weight", None)) if input.get("weight", None) is not None else None
        reduction = input.get("reduction", 'mean')
        ignore_index = input.get("ignore_index", -100)

        # Apply LogSoftmax
        input_tensor = tf.nn.log_softmax(input_tensor, axis=1)

        # Apply to TensorFlow equivalent
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_tensor, logits=input_tensor)

        if weight_tensor is not None:
            weight_mask = tf.gather(weight_tensor, target_tensor)
            loss = loss * weight_mask

        if ignore_index >= 0:
            mask = tf.cast(tf.not_equal(target_tensor, ignore_index))
            loss = loss * mask

        # Apply reduction
        if reduction == 'mean':
            loss = tf.reduce_mean(loss)
        elif reduction == 'sum':
            loss = tf.reduce_sum(loss)
        # 'none' case does not need reduction

        return {"nll_loss": float(loss.numpy())}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "target": np.array([2, 1], dtype=np.int64),  # Ensure target is int64 type for classification
        "weight": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "reduction": 'mean',
        "ignore_index": -100
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert and compare
    np.testing.assert_almost_equal(torch_result["nll_loss"], tf_result["nll_loss"], decimal=5)
    print("Results are equal" if torch_result["nll_loss"] == tf_result["nll_loss"] else "Results are not equal")

if __name__ == "__main__":
    main()