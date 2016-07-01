import argparse
import logging
import os
import caffe
import numpy as np
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
from google import protobuf
import IPython

import sys
"""
Put path to caffe installation from this fork:
  https://github.com/dpaiton/caffe/tree/gramian
"""
caffe_path = "..."
sys.path.insert(0,caffe_path)

# logging
LOG_FORMAT = "%(filename)s:%(funcName)s:%(asctime)s.%(msecs)03d -- %(message)s"

DEEPSTYLE_WEIGHTS = {"content": {"conv4_2": 1},
                     "style": {"conv1_1": 0.2,
                               "conv2_1": 0.2,
                               "conv3_1": 0.2,
                               "conv4_1": 0.2,
                               "conv5_1": 0.2}}

LENET_WEIGHTS = {"content": {"conv1": 1},
                "style": {"conv1": 0.5,
                          "conv2": 0.5}}

parser = argparse.ArgumentParser(description="Generate network using Caffe Python interface.",
                                 usage="generateNetwork.py -m <model_name>")
parser.add_argument("-m", "--model-name", default="vgg19", type=str, required=False, help="Model type (vgg19 only)")
parser.add_argument("-r", "--ratio", default="1e4", type=str, required=False, help="style-to-content ratio")
parser.add_argument("-o", "--output", default="./", required=False, help="output path")
parser.add_argument("-b", "--batch-size", default=1, type=int, required=False, help="Number of items in a batch")

# Helper functions
def _one_conv_relu(bottom, kernel_size, num_output, pad=0):
    if bottom is None:
        conv = L.Convolution(kernel_size=kernel_size, num_output=num_output, 
                pad=pad)
        conv.fn.params['param'] = [{'lr_mult': 0}, {'lr_mult': 0}]
    else:
        conv = L.Convolution(bottom, kernel_size=kernel_size, num_output=num_output, 
                pad=pad)
        conv.fn.params['param'] = [{'lr_mult': 0}, {'lr_mult': 0}]
    return conv, L.ReLU(conv, in_place=True)

def _two_conv_relu(bottom, kernel_size, num_output, pad=0):
    conv1, relu1 = _one_conv_relu(bottom, kernel_size, num_output, pad)
    conv2, relu2 = _one_conv_relu(relu1, kernel_size, num_output, pad)
    return conv1, relu1, conv2, relu2

def _four_conv_relu(bottom, kernel_size, num_output, pad=0):
    conv1, relu1, conv2, relu2 = _two_conv_relu(bottom, kernel_size, num_output, pad)
    conv3, relu3, conv4, relu4 = _two_conv_relu(relu2, kernel_size, num_output, pad)
    return conv1, relu1, conv2, relu2, conv3, relu3, conv4, relu4

def _max_pool(bottom, kernel_size, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=kernel_size,
            stride=stride)

def _ave_pool(bottom, kernel_size, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.AVE, kernel_size=kernel_size,
            stride=stride)

def _add_conv_relu_layers(func, network, **args):
    layer_no = args["layer_number"]
    if layer_no > 1:
        bottom = getattr(network.net, network.prefix+"pool"+str(layer_no-1))
    else:
        bottom = None
    for idx, layer in enumerate(func(bottom, args["kernel_size"], args["num_output"], args["pad"])):
        unit_no = (idx - idx%2)/2+1
        if idx%2:
            setattr(network.net, network.prefix+"relu"+str(layer_no)+"_"+str(unit_no), layer)
        else:
            setattr(network.net, network.prefix+"conv"+str(layer_no)+"_"+str(unit_no), layer)
    bottom = getattr(network.net, network.prefix+"relu"+str(layer_no)+"_"+str(args["num_conv"]))
    setattr(network.net, network.prefix+"pool"+str(layer_no), _ave_pool(bottom, args["kernel_size"], args["stride"]))

class NetSpec(object):
    """
        Create a network
    """
    def __init__(self, network_type, prefix, args=-1):
        """
            Initialize the model used for style transfer.

            :param str network_type:
                Model to build.
        """
        self.network_type = network_type
        self.prefix = prefix
        self.args = args
        self.create_net()

    def create_net(self):
        generator_method = getattr(self,"_create_"+self.network_type+"_net")
        if self.args == -1:
            generator_method()
        else:
            generator_method(self.args)

    def write_net(self, output_path):
        file_name = output_path + self.prefix + self.network_type + "_gen.prototxt"
        open(file_name,"w").write(str(self.net.to_proto()))
        return file_name

    def _create_lenet(self, lmdb, batch_size):
        # Caffe's version of LeNet: http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/01-learning-lenet.ipynb
        self.net = caffe.NetSpec()
        self.net.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb, \
                                        transform_param=dict(scale=1./255), ntop=2)
        self.net.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
        self.net.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
        self.net.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
        self.net.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
        self.net.ip1 = L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
        self.net.relu1 = L.ReLU(n.ip1, in_place=True)
        self.net.ip2 = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
        self.net.loss = L.SoftmaxWithLoss(n.ip2, n.label)

    def _create_vgg19_net(self):
        self.net = caffe.NetSpec()
        # conv1
        _add_conv_relu_layers(_two_conv_relu,\
                    self,\
                    num_conv=2,\
                    layer_number=1,\
                    kernel_size=3,\
                    num_output=64,\
                    pad=1,\
                    stride=2)
        # conv2
        _add_conv_relu_layers(_two_conv_relu,\
                    self,\
                    num_conv=2,\
                    layer_number=2,\
                    kernel_size=3,\
                    num_output=128,\
                    pad=1,\
                    stride=2)
        # conv3
        _add_conv_relu_layers(_four_conv_relu,\
                    self,\
                    num_conv=4,\
                    layer_number=3,\
                    kernel_size=3,\
                    num_output=256,\
                    pad=1,\
                    stride=2)
        # conv4
        _add_conv_relu_layers(_four_conv_relu,\
                    self,\
                    num_conv=4,\
                    layer_number=4,\
                    kernel_size=3,\
                    num_output=512,\
                    pad=1,\
                    stride=2)
        # conv5
        _add_conv_relu_layers(_four_conv_relu,\
                    self,\
                    num_conv=4,\
                    layer_number=5,\
                    kernel_size=3,\
                    num_output=512,\
                    pad=1,\
                    stride=2)

    def _create_merge_net(self, networks):
        self.net = caffe.NetSpec()
        for network in networks:
            for key in network.net.tops.keys():
                layer = getattr(network.net, key)
                setattr(self.net, key, layer)

def main(args):
    """
        Entry point.
    """

    # Logging
    level = logging.INFO
    logging.basicConfig(format=LOG_FORMAT, datefmt="%H:%M:%S", level=level)
    logging.info("Starting style transfer.")

    if args.model_name == "vgg19":
        weights = DEEPSTYLE_WEIGHTS
    elif args.model_name == "lenet":
        weights = LENET_WEIGHTS
    else:
        assert False, "Model "+args.model_name+" is not available."

    # Style net spec
    style_spec = NetSpec(args.model_name.lower(),prefix="style_")
    logging.info("Successfully loaded style model {0}.".format(args.model_name))

    # Content net spec
    content_spec = NetSpec(args.model_name.lower(),prefix="content_")
    content_spec.net.content_bias = L.Bias(param=dict(name="bias",lr_mult=1,decay_mult=0),
        bias_param=dict(bias_filler=dict(type="constant",value=0)))
    content_spec.net.content_conv1_1.fn.inputs = (content_spec.net.content_bias,)
    logging.info("Successfully loaded content model {0}.".format(args.model_name))

    # Merged net spec
    deepstyle_spec = NetSpec("merge", "deepstyle_", (style_spec, content_spec))

    # Set up input layer
    deepstyle_spec.net.style_data = L.Python(python_param=dict(module="pyData",layer="PyDataLayer"))
    deepstyle_spec.net.content_data = L.Python(python_param=dict(module="pyData",layer="PyDataLayer"))
    deepstyle_spec.net.content_bias.fn.inputs = (deepstyle_spec.net.content_data,)
    deepstyle_spec.net.style_conv1_1.fn.inputs = (deepstyle_spec.net.style_data,)

    for layer_name in weights["style"].keys():
        layer = L.Gramian(bottom="style_"+layer_name, gramian_param=dict(normalize_output=True))
        setattr(deepstyle_spec.net, "gramian_style_"+layer_name, layer)
        layer = L.Gramian(bottom="content_"+layer_name, gramian_param=dict(normalize_output=True))
        setattr(deepstyle_spec.net, "gramian_content_"+layer_name, layer)

    style_gram_list = [key for (key, value) in deepstyle_spec.net.tops.iteritems() if "gramian_style" in key]
    content_gram_list = [key for (key, value) in deepstyle_spec.net.tops.iteritems() if "gramian_content" in key]
    deepstyle_spec.net.concat_style_gramians = L.Concat(bottom=style_gram_list)
    deepstyle_spec.net.concat_content_gramians = L.Concat(bottom=content_gram_list)
    deepstyle_spec.net.style_loss = L.EuclideanLoss(bottom=["concat_style_gramians", "concat_content_gramians"],\
                                   loss_weight=np.float(args.ratio))

    style_activity_list = ["style_"+key for key in weights["content"].keys()]
    content_activity_list = ["content_"+key for key in weights["content"].keys()]
    deepstyle_spec.net.concat_style_activity = L.Concat(bottom=style_activity_list)
    deepstyle_spec.net.concat_content_activity = L.Concat(bottom=content_activity_list)
    deepstyle_spec.net.content_loss = L.EuclideanLoss(bottom=["concat_style_activity", "concat_content_activity"],\
                                   loss_weight=1)
    deepstyle_proto = deepstyle_spec.write_net(args.output)
    logging.info("Successfully wrote prototxt file {0}.".format(deepstyle_proto))

    vgg19_proto = "models/vgg19/VGG_ILSVRC_19_layers_deploy.prototxt"
    vgg19_model = "models/vgg19/VGG_ILSVRC_19_layers.caffemodel"

    # Load nets (supressing stderr output)
    null_fds = os.open(os.devnull, os.O_RDWR)
    out_orig = os.dup(2)
    os.dup2(null_fds, 2)
    vgg19_net = caffe.Net(vgg19_proto, vgg19_model, caffe.TEST)
    deepstyle_net = caffe.Net(deepstyle_proto, caffe.TEST)
    os.dup2(out_orig, 2)
    os.close(null_fds)

    logging.info("Successfully loaded deepstyle model {0}.".format(args.model_name))

    # Transfer params
    for vkey in vgg19_net.params.keys():
        for dkey in deepstyle_net.params.keys():
            if vkey in dkey:
                deepstyle_net.params[dkey] = vgg19_net.params[vkey][:]

    # Write params to file
    deepstyle_model = args.output + deepstyle_spec.prefix + deepstyle_spec.network_type + "_gen.caffemodel"
    deepstyle_net.save(deepstyle_model)
    logging.info("Successfully created weight file for deepstyle model {0}.".format(deepstyle_model))
    
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
