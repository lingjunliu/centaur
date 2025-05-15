import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    from torch.nn import AdaptiveLogSoftmaxWithLoss

    n_classes = input_dict["n_classes"]
    input_tensor = torch.tensor(input_dict["input"])
    target = torch.tensor(input_dict["target"])
    cutoff = input_dict["cutoff"]
    div_value = input_dict.get("div_value", 4.0)
    head_bias = input_dict.get("head_bias", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        target = target.cuda()

    adaptive_log_softmax = AdaptiveLogSoftmaxWithLoss(
        in_features=input_tensor.shape[-1],
        n_classes=n_classes,
        cutoffs=cutoff,
        div_value=div_value,
        head_bias=head_bias,
    )

    if not cpu:
        adaptive_log_softmax = adaptive_log_softmax.cuda()
    
    output = adaptive_log_softmax(input_tensor, target)
    loss = output.loss
    if not cpu:
        loss = loss.cpu()
    
    return {"loss": loss.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    n_classes = input_dict["n_classes"]
    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    target = tf.constant(input_dict["target"], dtype=tf.int32)
    cutoff = input_dict["cutoff"]
    div_value = input_dict.get("div_value", 4.0)
    head_bias = input_dict.get("head_bias", False)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        
        def adaptive_softmax_loss(logits, labels, n_classes, cutoff, div_value, head_bias):

            def log_prob_from_logits(logits):
                return logits - tf.reduce_logsumexp(logits, axis=-1, keepdims=True)

            def sample_softmax_loss(logits, labels, n_classes, input_dim):
                batch_size = tf.shape(logits)[0]
                labels = tf.minimum(labels, n_classes - 1)
                range_ = tf.expand_dims(tf.range(batch_size), 1)
                indices = tf.concat([range_, tf.expand_dims(labels, 1)], axis=-1)
                
                valid_indices = tf.where(labels < n_classes)
                
                local_logits = tf.gather_nd(logits, tf.gather_nd(indices, valid_indices))
                local_log_prob = log_prob_from_logits(logits)
                local_log_prob = tf.gather_nd(local_log_prob, tf.gather_nd(indices, valid_indices))

                return tf.reduce_mean(-local_log_prob)
            
            full_loss = sample_softmax_loss(logits, labels, n_classes, input_tensor.shape[-1])
            return full_loss
        
        loss = adaptive_softmax_loss(input_tensor, target, n_classes, cutoff, div_value, head_bias)
        
    return {"loss": loss.numpy()}

def main():
    A_TOL = 0.01
    input_dim = 10
    n_classes = 100
    cutoff = [20, 50]
    batch_size = 32
    
    input_data = {
        "n_classes": n_classes,
        "input": np.random.randn(batch_size, input_dim).astype(np.float32),
        "target": np.random.randint(0, n_classes, size=(batch_size,)).astype(np.int64),
        "cutoff": cutoff,
        "div_value": 4.0,
        "head_bias": False,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    print(f"Torch Loss: {torch_result['loss']}")
    print(f"TensorFlow Loss: {tf_result['loss']}")

    assert np.allclose(torch_result["loss"], tf_result["loss"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()