import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    modules = input_dict['modules']
    
    torch_modules = []
    for module in modules:
        torch_modules.append(torch.nn.Parameter(torch.tensor(module)))
    
    module_list = torch.nn.ParameterList(torch_modules)
    
    if not cpu:
        module_list = module_list.cuda()
    
    result = [param.detach().cpu().numpy() for param in module_list]
    
    return {'result': result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    
    modules = input_dict['modules']
    
    tf_modules = []
    for module in modules:
        tf_modules.append(tf.Variable(module))

    class ParameterListMimic:
        def __init__(self, parameters):
            self.parameters = parameters
        
        def __len__(self):
            return len(self.parameters)
        
        def __getitem__(self, index):
            return self.parameters[index]
    
    module_list = ParameterListMimic(tf_modules)
    
    result = [param.numpy() for param in module_list.parameters]
    
    return {'result': result}

def main():
    A_TOL = 0.01
    
    input_data = {
        "modules": [np.array([1.0, 2.0], dtype=np.float32), np.array([3.0, 4.0], dtype=np.float32)]
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    for i in range(len(torch_result['result'])):
        assert np.allclose(torch_result['result'][i], tf_result['result'][i], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()