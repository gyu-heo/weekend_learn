import torch
import torch.nn as nn
import numpy as np

import pytest
import copy
import matplotlib.pyplot as plt




def test_sampled_gradients():
    ## Create a very simple 2-layer MLP network with scalar output
    input_size = 3
    hidden_size = 10
    output_size = 1
    class simple_net(nn.Module):
        def __init__(self,
                     input_size,
                     hidden_size,
                     output_size):
            super(simple_net, self).__init__()
            self.layer_1 = torch.nn.Linear(input_size, hidden_size)
            self.layer_2 = torch.nn.Linear(hidden_size, hidden_size)
            self.output_layer = torch.nn.Linear(hidden_size, output_size)
        def forward(self, input):
            act_1 = torch.nn.functional.relu(self.layer_1(input))
            act_2 = torch.nn.functional.relu(self.layer_2(act_1))
            y = self.output_layer(act_2)
            return y
        
    ## Create a dummy input-output pair
    dummy_input = torch.randn(1, input_size)
    dummy_output = torch.randn(1, output_size)

    ## Create the network
    net = simple_net()

    ## Get the gradients with autodiff
    net.zero_grad()
    output = net(dummy_input)
    loss = torch.nn.functional.mse_loss(output, dummy_output)
    loss.backward()

    gradients_autodiff = {}
    for name, param in net.named_parameters():
        gradients_autodiff[name] = param.grad
    
    ## Get the gradients with weight perturbation
    noise_std = 0.1
    num_samples = 100
    wp_gradients = WP_estimation(net, dummy_input, dummy_output, noise_std, num_samples)
    np_gradients = NP_estimation(net, dummy_input, dummy_output, noise_std, num_samples)


######## Housekeeping ########

## Weight perturbation rule for MLP
def WP_estimation(net, input, output, noise_std, num_samples):
    ## Sample k random noises
    num_params = sum([np.prod(p.size()) for p in net.parameters()])
    noises = [noise_std * torch.randn(num_params) for _ in range(num_samples)]

    ## compute noiseless pass
    clean_loss = torch.nn.functional.mse_loss(net(input), output)

    ## compute noisy pass
    noisy_losses, noisy_gradients = [], []
    for noise in noises:
        net_copy = copy.deepcopy(net)
        flatten_idx = 0
        for name, param in net_copy.named_parameters():
            param.data += noise[flatten_idx:flatten_idx+np.prod(param.size())].view(param.size())
            flatten_idx += np.prod(param.size())
        noisy_loss = torch.nn.functional.mse_loss(net_copy(input), output)
        noisy_losses.append(noisy_loss)

        noisy_gradient = ((noisy_loss - clean_loss) / noise_std) * noise
        noisy_gradients.append(noisy_gradient)

    return noisy_gradients


def NP_estimation(net, input, output, noise_std, num_samples):
    ## Sample k random noises
    node_per_layer = [p.size()[0] for name, p in net.named_parameters() if "weight" in name]
    noises = [noise_std * torch.randn(sum(node_per_layer)) for _ in range(num_samples)]

    ## compute noiseless pass
    clean_loss = torch.nn.functional.mse_loss(net(input), output)

    ## compute noisy pass
    noisy_losses, noisy_gradients = [], []
    for noise in noises:
        net_copy = copy.deepcopy(net)
        flatten_idx = 0
        for name, param in net_copy.named_parameters():
            if "bias" in name:
                ## For node perturbation, we should add the noise to bias to effectively perturb the weight
                param.data += noise[flatten_idx:flatten_idx+param.size()]
                flatten_idx += param.size()
        
        ## For the node perturbation, we need intermediate outputs!!!
        hook_handles = {}
        def get_intermediate_activation(name):
            def hook(model, input, output):
                hook_handles[name] = output
        
        net_copy.layer_1.register_forward_hook(get_intermediate_activation("act_1"))
        net_copy.layer_2.register_forward_hook(get_intermediate_activation("act_2"))

        ## Okay! noisy forward pass
        noisy_loss = torch.nn.functional.mse_loss(net_copy(input), output)
        noisy_losses.append(noisy_loss)

        ## Compute the noisy gradient with flattened element-wise multiplication
        flattened_previous_activations = torch.cat([input, hook_handles["act_1"], hook_handles["act_2"]], dim=1)
        noisy_gradient = ((noisy_loss - clean_loss) / noise_std) * noise * flattened_previous_activations
        noisy_gradients.append(noisy_gradient)

    return noisy_gradients