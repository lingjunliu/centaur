
def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    # input
    x = torch.tensor(input["input"])
    
    if not cpu:
        x = x.cuda()
        
    # output
    y = torch.ravel(x)
    
    if not cpu:
        y = y.cpu()          
    
    return {"ravel": y.numpy()}


def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    if cpu:
        device_string = "/cpu"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # input
        x = tf.constant(input["input"])

        # output
        y = tf.experimental.numpy.ravel(x)
        
        return {"ravel": y.numpy()}