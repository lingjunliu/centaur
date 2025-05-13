import numpy as np
import tensorflow as tf

def torch_version(input_dict, cpu=True):
    import torch
    
    parameters = input_dict['parameters']
    max_norm = input_dict.get('max_norm')
    norm_type = input_dict.get('norm_type', 2.0)
    error_if_nonfinite = input_dict.get('error_if_nonfinite', False)

    torch_parameters = [torch.tensor(p, requires_grad=True) for p in parameters]

    if not cpu:
        torch_parameters = [p.cuda() for p in torch_parameters]

    torch.nn.utils.clip_grad_norm_(torch_parameters, max_norm, norm_type=norm_type, error_if_nonfinite=error_if_nonfinite)

    if not cpu:
        torch_parameters = [p.cpu() for p in torch_parameters]

    return {'parameters': [p.detach().numpy() for p in torch_parameters]}


def tensorflow_version(input_dict, cpu=True):
    parameters = input_dict['parameters']
    max_norm = input_dict.get('max_norm')
    norm_type = input_dict.get('norm_type', 2.0)
    error_if_nonfinite = input_dict.get('error_if_nonfinite', False)
    
    tf_parameters = [tf.Variable(p, dtype=tf.float32) for p in parameters]
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        grads, _ = tf.clip_by_global_norm(
            [p for p in tf_parameters],
            clip_norm=max_norm,
            use_norm=tf.constant(norm_type, dtype=tf.float32)
        )

        if error_if_nonfinite:
            global_norm = tf.norm([tf.norm(g) for g in grads])
            global_norm_check = tf.debugging.check_numerics(
                global_norm,
                message='global_norm is inf or nan'
            )
            with tf.control_dependencies([global_norm_check]):
                assign_ops = [tf_parameters[i].assign(grads[i]) for i in range(len(tf_parameters))]
                
                # Need to explicitly run the assign ops in eager mode
                for op in assign_ops:
                    op.numpy()
        else:
            assign_ops = [tf_parameters[i].assign(grads[i]) for i in range(len(tf_parameters))]
            
            # Need to explicitly run the assign ops in eager mode
            for op in assign_ops:
                op.numpy()
                

    return {'parameters': [p.numpy() for p in tf_parameters]}


def main():
    A_TOL = 0.01

    # Example input
    input_data = {
        'parameters': [
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([4.0, 5.0, 6.0], dtype=np.float32)
        ],
        'max_norm': 5.0,
        'norm_type': 2.0,
        'error_if_nonfinite': False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    for i in range(len(torch_result['parameters'])):
        assert np.allclose(torch_result['parameters'][i], tf_result['parameters'][i], atol=A_TOL), "Results do not match"
    
    print("Success")


if __name__ == "__main__":
    main()