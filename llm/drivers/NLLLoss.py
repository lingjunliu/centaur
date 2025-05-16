import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    target_tensor = torch.tensor(input_dict["target"])
    weight = input_dict.get("weight", None)
    if weight is not None:
        weight = torch.tensor(weight)
    ignore_index = input_dict.get("ignore_index", -100)
    reduction = input_dict.get("reduction", 'mean')

    if not cpu:
        input_tensor = input_tensor.cuda()
        target_tensor = target_tensor.cuda()
        if weight is not None:
            weight = weight.cuda()

    result = torch.nn.NLLLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)(input_tensor, target_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        target_tensor = tf.cast(tf.constant(input_dict["target"]), dtype=tf.int32)
        weight = input_dict.get("weight", None)
        ignore_index = input_dict.get("ignore_index", -100)
        reduction = input_dict.get("reduction", 'mean')

        if weight is not None:
            weight = tf.constant(weight)

        input_shape = tf.shape(input_tensor)
        target_shape = tf.shape(target_tensor)

        if len(input_shape.numpy()) == 2 and len(target_shape.numpy()) == 1:
            num_classes = tf.shape(input_tensor)[1]

            log_probs = input_tensor
            targets = target_tensor

            if ignore_index != -100:
                mask = tf.not_equal(targets, ignore_index)
                valid_log_probs = tf.boolean_mask(log_probs, mask)
                valid_targets = tf.boolean_mask(targets, mask)
                if tf.reduce_sum(tf.cast(mask, tf.int32)) == 0:
                    result = tf.constant(0.0, dtype=tf.float32)
                    return {"result": result.numpy()}
            else:
                valid_log_probs = log_probs
                valid_targets = targets

            one_hot_labels = tf.one_hot(tf.cast(valid_targets, tf.int32), depth=num_classes)

            nll_loss = -tf.reduce_sum(one_hot_labels * valid_log_probs, axis=1)

            if weight is not None:
                nll_loss = nll_loss * tf.gather(weight, tf.cast(valid_targets, tf.int32))

            if reduction == 'mean':
                result = tf.reduce_mean(nll_loss)
            elif reduction == 'sum':
                result = tf.reduce_sum(nll_loss)
            else:
                result = nll_loss

        else:
            raise ValueError("Unsupported input shapes.  TensorFlow version requires input to be 2D and target to be 1D.")

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[-0.7715, -0.0695, -0.3954],
                           [-0.8246, -0.6807, -0.0766],
                           [-0.9798, -0.7466, -0.6936],
                           [-0.7233, -0.9665, -0.8727]], dtype=np.float32),
        "target": np.array([0, 1, 2, 0], dtype=np.int64),
        "weight": np.array([0.1, 0.5, 0.2], dtype=np.float32),
        "reduction": 'mean',
        "ignore_index": -100
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()