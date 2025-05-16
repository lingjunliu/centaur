import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    target = torch.tensor(input_dict["target"])
    size_average = input_dict.get("size_average", None)
    reduce = input_dict.get("reduce", None)
    reduction = input_dict.get("reduction", 'mean')

    if not cpu:
        input_tensor = input_tensor.cuda()
        target = target.cuda()

    loss = torch.nn.MultiLabelMarginLoss(reduction=reduction)
    result = loss(input_tensor, target)

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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        target = tf.cast(tf.constant(input_dict["target"], dtype=tf.int64), dtype=tf.float32)
        reduction = input_dict.get("reduction", 'mean')

        batch_size = tf.shape(input_tensor)[0]
        num_classes = tf.shape(input_tensor)[1]

        losses = []
        for i in range(batch_size):
            x = input_tensor[i]
            y = target[i]

            positive_indices = tf.where(tf.equal(y, 1.0))
            positive_indices = tf.reshape(positive_indices, [-1])
            num_positives = tf.cast(tf.shape(positive_indices)[0], dtype=tf.float32)

            if num_positives > 0:
                loss_sum = 0.0
                for j in range(num_classes):
                    is_positive = tf.reduce_sum(tf.cast(tf.equal(positive_indices, tf.cast(j, dtype=tf.int64)), dtype=tf.float32))
                    if is_positive == 0:
                        margins = 1 - tf.gather(x, tf.cast(positive_indices, dtype=tf.int32)) + x[j]
                        margins = tf.maximum(margins, 0.0)
                        loss_sum += tf.reduce_sum(margins)

                loss_per_sample = loss_sum / (tf.cast(num_classes, dtype=tf.float32) * num_positives)
                losses.append(loss_per_sample)
            else:
                losses.append(tf.constant(0.0, dtype=tf.float32))

        if reduction == 'mean':
            result = tf.reduce_mean(losses)
        elif reduction == 'sum':
            result = tf.reduce_sum(losses)
        else:
            result = tf.convert_to_tensor(losses, dtype=tf.float32).numpy()

        if reduction != 'none':
            result = tf.convert_to_tensor(result, dtype=tf.float32).numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[0.1, 0.2, 0.4, 0.8]], dtype=np.float32),
        "target": np.array([[0, 0, 1, 1]], dtype=np.int64),
        "reduction": 'mean'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[0.1, 0.2, 0.4, 0.8], [0.1, 0.2, 0.4, 0.8]], dtype=np.float32),
        "target": np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.int64),
        "reduction": 'mean'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([[0.1, 0.2, 0.4, 0.8]], dtype=np.float32),
        "target": np.array([[0, 0, 1, 1]], dtype=np.int64),
        "reduction": 'sum'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([[0.1, 0.2, 0.4, 0.8], [0.1, 0.2, 0.4, 0.8]], dtype=np.float32),
        "target": np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.int64),
        "reduction": 'sum'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[0.1, 0.2, 0.4, 0.8], [0.1, 0.2, 0.4, 0.8]], dtype=np.float32),
        "target": np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.int64),
        "reduction": 'none'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    

    print("Success")

if __name__ == "__main__":
    main()