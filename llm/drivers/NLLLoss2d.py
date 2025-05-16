import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    target = torch.tensor(input_dict["target"])
    weight = input_dict.get("weight", None)
    if weight is not None:
        weight = torch.tensor(weight)
    size_average = input_dict.get("size_average", None)
    reduce = input_dict.get("reduce", None)
    ignore_index = input_dict.get("ignore_index", -100)

    if not cpu:
        input_tensor = input_tensor.cuda()
        target = target.cuda()
        if weight is not None:
            weight = weight.cuda()

    loss = torch.nn.NLLLoss(weight=weight, ignore_index=ignore_index, reduction='mean' if (size_average is not None and size_average) or (reduce is not None and reduce) else ('sum' if (size_average is not None and not size_average) or (reduce is not None and not reduce) else 'mean'))
    
    input_tensor = input_tensor.reshape(1, 2, 2, 2)
    target = target.reshape(1, 2, 2)
    target = target.long()
    result = loss(input_tensor, target)
    
    if not cpu:
        result = result.cpu()

    return {'result': result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    target = tf.constant(input_dict["target"])
    weight = input_dict.get("weight", None)
    if weight is not None:
        weight = tf.constant(weight)
    size_average = input_dict.get("size_average", None)
    reduce = input_dict.get("reduce", None)
    ignore_index = input_dict.get("ignore_index", -100)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        n, c, h, w = input_tensor.shape
        input_tensor = tf.reshape(input_tensor, [n, c, h * w])
        input_tensor = tf.transpose(input_tensor, perm=[0, 2, 1])
        input_tensor = tf.reshape(input_tensor, [-1, c])
        target = tf.reshape(target, [-1])

        if ignore_index != -100:
            mask = tf.not_equal(target, ignore_index)
            target = tf.boolean_mask(target, mask)
            input_tensor = tf.boolean_mask(input_tensor, mask)

        log_probs = tf.nn.log_softmax(input_tensor, axis=1)
        
        if weight is not None:
            weights = tf.gather(weight, target)
            indices = tf.stack([tf.range(tf.shape(target)[0], dtype=tf.int32), tf.cast(target, tf.int32)], axis=1)
            neg_log_likelihood = -tf.gather_nd(log_probs, indices) * weights
        else:
            indices = tf.stack([tf.range(tf.shape(target)[0], dtype=tf.int32), tf.cast(target, tf.int32)], axis=1)
            neg_log_likelihood = -tf.gather_nd(log_probs, indices)
            

        if (size_average is not None and size_average) or (reduce is not None and reduce):
            loss = tf.reduce_mean(neg_log_likelihood)
        elif (size_average is not None and not size_average) or (reduce is not None and not reduce):
            loss = tf.reduce_sum(neg_log_likelihood)
        else:
            loss = tf.reduce_mean(neg_log_likelihood)
        
        result = loss.numpy()

    return {'result': result}

def main():
    A_TOL = 0.01

    input_data = {
        'input': np.array([[[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]]).astype(np.float32),
        'target': np.array([[[0, 1], [1, 0]]]).astype(np.int64),
        'weight': np.array([0.5, 0.5]).astype(np.float32)
    }
    
    input_data['input'] = np.log(input_data['input'])

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result['result'], tf_result['result'], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()