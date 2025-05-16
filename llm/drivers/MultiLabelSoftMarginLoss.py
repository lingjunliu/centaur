import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    target_tensor = torch.tensor(input_dict["target"])
    weight = input_dict.get("weight", None)
    if weight is not None:
        weight = torch.tensor(input_dict["weight"])
    size_average = input_dict.get("size_average", None)
    reduce = input_dict.get("reduce", None)
    reduction = input_dict.get("reduction", 'mean')

    if not cpu:
        input_tensor = input_tensor.cuda()
        target_tensor = target_tensor.cuda()
        if weight is not None:
            weight = weight.cuda()

    loss_function = torch.nn.MultiLabelSoftMarginLoss(reduction=reduction)
    result = loss_function(input_tensor, target_tensor)

    if weight is not None and reduction is None or reduction == 'none':
        result = result * weight

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
        target_tensor = tf.constant(input_dict["target"])
        weight = input_dict.get("weight", None)
        if weight is not None:
            weight = tf.constant(weight)
        reduction = input_dict.get("reduction", 'mean')

        def _tf_multilabel_soft_margin_loss(logits, labels, weights=None, reduction='mean'):
            log_prob = tf.nn.log_sigmoid(logits)
            loss = -tf.reduce_sum(labels * log_prob + (1 - labels) * (log_prob - logits), axis=1)

            if weights is not None:
                loss = loss * weights
                if reduction == 'mean':
                    return tf.reduce_mean(loss)
                elif reduction == 'sum':
                    return tf.reduce_sum(loss)
                else:
                    return loss

            if reduction == 'mean':
                return tf.reduce_mean(loss)
            elif reduction == 'sum':
                return tf.reduce_sum(loss)
            else:
                return loss

        result = _tf_multilabel_soft_margin_loss(input_tensor, target_tensor, weights=weight, reduction=reduction)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float32),
        "target": np.array([[0, 0, 1, 0], [0, 1, 0, 1]], dtype=np.float32),
        "weight": np.array([0.5, 0.5], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()