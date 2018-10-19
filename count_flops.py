from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable
from functools import reduce
import operator


count_ops = 0
count_params = 0


def get_num_gen(gen):
    return sum(1 for x in gen)


def is_pruned(layer):
    try:
        layer.mask
        return True
    except AttributeError:
        return False


def is_leaf(model):
    return get_num_gen(model.children()) == 0

from pytorch_acdc.layers import FastStackedConvACDC, ConvACDC

def is_acdc(layer):
    return isinstance(layer, FastStackedConvACDC) or isinstance(layer, ConvACDC)


def get_layer_info(layer):
    layer_str = str(layer)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name


def get_layer_param(model):
    return sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])


### The input batch size should be 1 to call this function
def measure_layer(layer, x):
    global count_ops, count_params
    delta_ops = 0
    delta_params = 0
    multi_add = 1
    type_name = get_layer_info(layer)

    ### ops_conv
    if type_name in ['Conv2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
        delta_params = get_layer_param(layer)

    ### ops_nonlinearity
    elif type_name in ['ReLU']:
        delta_ops = x.numel()
        delta_params = get_layer_param(layer)

    ### ops_pooling
    elif type_name in ['AvgPool2d','MaxPool2d']:
        in_w = x.size()[2]
        kernel_ops = layer.kernel_size * layer.kernel_size
        out_w = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        out_h = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        delta_ops = x.size()[0] * x.size()[1] * out_w * out_h * kernel_ops
        delta_params = get_layer_param(layer)

    ### ops_linear
    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        bias_ops = layer.bias.numel()
        delta_ops = x.size()[0] * (weight_ops + bias_ops)
        delta_params = get_layer_param(layer)

    ### ops_nothing
    elif type_name in ['BatchNorm2d', 'Dropout2d', 'DropChannel', 'Dropout']:
        delta_params = get_layer_param(layer)

    ### sequential takes no extra time
    elif type_name in ['Sequential']:
        pass
    
    ### riffle shuffle
    elif type_name in ['Riffle']:
        # technically no floating point operations
        pass

    ### channel expansion
    elif type_name in ['ChannelExpand']:
        # assume concatentation doesn't take extra FLOPs
        pass

    ### channel contraction
    elif type_name in ['ChannelCollapse']:
        # do as many additions as we have channels
        delta_ops += x.size(1)

    ### ACDC Convolution
    elif type_name in ['FastStackedConvACDC']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)       
        # pretend we're actually passing through the ACDC layers within
        N = max(layer.out_channels, layer.in_channels) # size of ACDC layers
        acdc_ops = 4*N + 5*N*math.log(N,2)
        conv_ops = N * N * layer.kernel_size[0] *  \
                   layer.kernel_size[1]  / layer.groups
        ops = min(acdc_ops, conv_ops)
        delta_ops += ops*out_h*out_w
        delta_params += 2*N

    ### Grouped ACDC Convolution
    elif type_name in ['GroupedConvACDC']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)       
        # pretend we're actually passing through the ACDC layers within
        N = layer.kernel_size[0]
        acdc_ops = layer.groups*(4*N + 5*N*math.log(N,2))
        conv_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                   layer.kernel_size[1]  / layer.groups
        ops = min(acdc_ops, conv_ops)
        delta_ops += ops*out_h*out_w
        delta_params += 2*N

    ### unknown layer type
    else:
        raise TypeError('unknown layer type: %s' % type_name)

    count_ops += delta_ops
    count_params += delta_params
    return


def measure_model(model, H, W):
    global count_ops, count_params
    count_ops = 0
    count_params = 0
    data = Variable(torch.zeros(1, 3, H, W))

    def should_measure(x):
        return is_leaf(x) or is_pruned(x) or is_acdc(x)

    # this is a dangerous recursive function
    def modify_forward(model):
        for child in model.children():
            if should_measure(child):
                def new_forward(m):
                    def lambda_forward(x):
                        measure_layer(m, x)
                        return m.old_forward(x)
                    return lambda_forward
                child.old_forward = child.forward
                child.forward = new_forward(child)
            else:
                modify_forward(child)

    def restore_forward(model):
        for child in model.children():
            # leaf node
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)

    modify_forward(model)
    model.forward(data)
    restore_forward(model)

    return count_ops, count_params

if __name__ == '__main__':
    from models import *

    print("Tiny ConvNet\tFLOPS\t\tparams")
    model = AllConv()
    flops, params = measure_model(model, 32, 32)
    print("  Original:\t%.5E\t%.5E"%(flops, params))

    model = AllConvACDC()
    flops, params = measure_model(model, 32, 32)
    print("  ACDC:\t\t%.5E\t%.5E"%(flops, params))

    print("ResNet18\tFLOPS\t\tparams")
    model = ResNet18()
    flops, params = measure_model(model, 32, 32)
    print("  Original:\t%.5E\t%.5E"%(flops, params))

    model = ACDCResNet18()
    flops, params = measure_model(model, 32, 32)
    print("  ACDC:\t\t%.5E\t%.5E"%(flops, params))

    print("WRN(40,2)\tFLOPS\t\tparams")
    model = WideResNetDefault(40,2)
    flops, params = measure_model(model, 32, 32)
    print("  Original:\t%.5E\t%.5E"%(flops, params))

    model = WideResNetACDC(40,2)
    flops, params = measure_model(model, 32, 32)
    print("  ACDC:\t\t%.5E\t%.5E"%(flops, params))
