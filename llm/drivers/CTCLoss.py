import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input_tensor"])
    targets = torch.tensor(input_dict["targets"])
    input_lengths = torch.tensor(input_dict["input_lengths"])
    target_lengths = torch.tensor(input_dict["target_lengths"])
    blank = input_dict.get("blank", 0)
    reduction = input_dict.get("reduction", 'mean')
    zero_infinity = input_dict.get("zero_infinity", False)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        targets = targets.cuda()
        input_lengths = input_lengths.cuda()
        target_lengths = target_lengths.cuda()

    ctc_loss = torch.nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=zero_infinity)
    result = ctc_loss(input_tensor.log_softmax(2), targets, input_lengths, target_lengths)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input_tensor"])
        targets = tf.constant(input_dict["targets"], dtype=tf.int32)
        input_lengths = tf.constant(input_dict["input_lengths"], dtype=tf.int32)
        target_lengths = tf.constant(input_dict["target_lengths"], dtype=tf.int32)
        blank = input_dict.get("blank", 0)
        reduction = input_dict.get("reduction", 'mean')
        zero_infinity = input_dict.get("zero_infinity", False)

        input_tensor_shape = tf.shape(input_tensor)
        input_tensor = tf.transpose(input_tensor, perm=[1, 0, 2])

        loss = tf.nn.ctc_loss(
            labels=tf.cast(targets, tf.int64),
            logits=input_tensor,
            label_length=tf.cast(target_lengths, tf.int64),
            log_probs=True,
            input_length=tf.cast(input_lengths, tf.int64),
            blank_index=blank
        )

        if reduction == 'none':
            pass
        elif reduction == 'mean':
            loss = tf.reduce_mean(loss)
        elif reduction == 'sum':
            loss = tf.reduce_sum(loss)
        else:
            raise ValueError(f"Invalid reduction option: {reduction}")
        
        if zero_infinity:
            loss = tf.where(tf.math.is_inf(loss), tf.zeros_like(loss), loss)

        result = loss.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input_tensor": np.array([[[0.1, 0.6, 0.1, 0.1, 0.1],
                                  [0.1, 0.1, 0.6, 0.1, 0.1],
                                  [0.1, 0.1, 0.1, 0.6, 0.1],
                                  [0.6, 0.1, 0.1, 0.1, 0.1]]], dtype=np.float32),
        "targets": np.array([1, 2, 3], dtype=np.int32),
        "input_lengths": np.array([4], dtype=np.int32),
        "target_lengths": np.array([3], dtype=np.int32),
        "blank": 0,
        "reduction": 'mean',
        "zero_infinity": False
    }

    input_data["input_lengths"] = np.array([input_data["input_lengths"].item()], dtype=np.int32)
    input_data["target_lengths"] = np.array([input_data["target_lengths"].item()], dtype=np.int32)

    input_data["input_tensor"] = np.repeat(input_data["input_tensor"], 2, axis=0)
    input_data["input_lengths"] = np.repeat(input_data["input_lengths"], 2, axis=0)
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()