import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["log_probs"])
    target_tensor = torch.tensor(input_dict["targets"])
    input_lengths = input_dict["input_lengths"]
    target_lengths = input_dict["target_lengths"]
    blank = input_dict.get("blank", 0)
    reduction = input_dict.get("reduction", 'mean')
    zero_infinity = input_dict.get("zero_infinity", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        target_tensor = target_tensor.cuda()
    
    result = torch.ctc_loss(input_tensor, target_tensor, torch.tensor([input_lengths], dtype=torch.int), torch.tensor([target_lengths], dtype=torch.int), blank=blank, reduction=reduction, zero_infinity=zero_infinity)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    log_probs = input_dict["log_probs"]
    targets = input_dict["targets"]
    input_lengths = input_dict["input_lengths"]
    target_lengths = input_dict["target_lengths"]
    blank = input_dict.get("blank", 0)
    reduction = input_dict.get("reduction", 'mean')
    zero_infinity = input_dict.get("zero_infinity", False)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        log_probs_tensor = tf.convert_to_tensor(log_probs, dtype=tf.float32)
        targets_tensor = tf.convert_to_tensor(targets, dtype=tf.int32)
        input_lengths_tensor = tf.convert_to_tensor([input_lengths], dtype=tf.int32)
        target_lengths_tensor = tf.convert_to_tensor([target_lengths], dtype=tf.int32)

        loss = tf.keras.backend.ctc_batch_cost(targets_tensor, log_probs_tensor, input_lengths_tensor, target_lengths_tensor)

        if zero_infinity:
            mask = tf.math.is_inf(loss)
            loss = tf.where(mask, tf.zeros_like(loss), loss)

        if reduction == 'none':
            result = loss.numpy()
        elif reduction == 'mean':
            result = tf.reduce_mean(loss).numpy()
        elif reduction == 'sum':
            result = tf.reduce_sum(loss).numpy()
        else:
            raise ValueError(f"Invalid reduction option: {reduction}")

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "log_probs": np.array([[[0.1, 0.6, 0.1, 0.1, 0.1],
                                 [0.1, 0.1, 0.6, 0.1, 0.1],
                                 [0.1, 0.1, 0.1, 0.6, 0.1],
                                 [0.6, 0.1, 0.1, 0.1, 0.1]]], dtype=np.float32),
        "targets": np.array([1, 2, 3], dtype=np.int32),
        "input_lengths": 4,
        "target_lengths": 3,
        "blank": 4,
        "reduction": "mean",
        "zero_infinity": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "log_probs": np.array([[[0.1, 0.6, 0.1, 0.1, 0.1],
                                 [0.1, 0.1, 0.6, 0.1, 0.1],
                                 [0.1, 0.1, 0.1, 0.6, 0.1],
                                 [0.6, 0.1, 0.1, 0.1, 0.1]]], dtype=np.float32),
        "targets": np.array([1, 2, 3], dtype=np.int32),
        "input_lengths": 4,
        "target_lengths": 3,
        "blank": 4,
        "reduction": "sum",
        "zero_infinity": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "log_probs": np.array([[[0.1, 0.6, 0.1, 0.1, 0.1],
                                 [0.1, 0.1, 0.6, 0.1, 0.1],
                                 [0.1, 0.1, 0.1, 0.6, 0.1],
                                 [0.6, 0.1, 0.1, 0.1, 0.1]]], dtype=np.float32),
        "targets": np.array([1, 2, 3], dtype=np.int32),
        "input_lengths": 4,
        "target_lengths": 3,
        "blank": 4,
        "reduction": "none",
        "zero_infinity": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()