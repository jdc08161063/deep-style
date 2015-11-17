from __future__ import print_function
import argparse
import logging
import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
from google import protobuf
import IPython

# logging
LOG_FORMAT = "%(filename)s:%(funcName)s:%(asctime)s.%(msecs)03d -- %(message)s"

parser = argparse.ArgumentParser(description="Generate network using Caffe Python interface.",
                                 usage="generateNetwork.py -m <model_name>")
parser.add_argument("-m", "--model-name", type=str, required=True, help="Model type (vgg19 only)")
parser.add_argument("-b", "--batch-size", default=1, type=int, required=False, help="Number of items in a batch")

        
# Helper functions
def _one_conv_relu(bottom, kernel_size, num_output, pad=0):
    conv = L.Convolution(bottom, kernel_size=kernel_size, num_output=num_output, 
            pad=pad)
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

class Network(object):
    """
        Create a network
    """
    def __init__(self, network_name, prefix):
        """
            Initialize the model used for style transfer.

            :param str network_name:
                Model to build.
        """
        self.network_name = network_name
        self.prefix = prefix
        self.create_net()

    def create_net(self):
        generator_method = getattr(self,"_create_"+self.network_name+"_net")
        generator_method()

    def print_net(self):
        with open(self.prefix+self.network_name+'_gen.prototxt', 'w') as f:
            print(self.net.to_proto(), file=f)

    def _create_vgg19_net(self):
        self.net = caffe.NetSpec()

        setattr(self.net, self.prefix+"conv1_1", L.Convolution(kernel_size=3, num_output=64, pad=1))
        setattr(self.net, self.prefix+"relu1_1", L.ReLU(getattr(self.net, self.prefix+"conv1_1"), in_place=True))

        bottom = getattr(self.net, self.prefix+"relu1_1")
        for idx, layer in enumerate(_one_conv_relu(bottom, kernel_size=3, num_output=64, pad=1)):
            if idx%2:
                setattr(self.net, self.prefix+"conv1_2", layer)
            else:
                setattr(self.net, self.prefix+"relu1_2", layer)
        bottom = getattr(self.net, self.prefix+"relu1_2")
        setattr(self.net, self.prefix+"pool1", _ave_pool(bottom, kernel_size=2, stride=2))

        bottom = getattr(self.net, self.prefix+"pool1")
        for idx, layer in enumerate(_two_conv_relu(bottom, kernel_size=3, num_output=128, pad=1)):
            layer_no = (idx - idx%2)/2+1
            if idx%2:
                setattr(self.net, self.prefix+"conv2_"+str(layer_no), layer)
            else:
                setattr(self.net, self.prefix+"relu2_"+str(layer_no), layer)
        bottom = getattr(self.net, self.prefix+"relu2_2")
        setattr(self.net, self.prefix+"pool2", _ave_pool(bottom, kernel_size=2, stride=2))

        bottom = getattr(self.net, self.prefix+"pool2")
        for idx, layer in enumerate(_four_conv_relu(bottom, kernel_size=3, num_output=256, pad=1)):
            layer_no = (idx - idx%2)/2+1
            if idx%2:
                setattr(self.net, self.prefix+"conv3_"+str(layer_no), layer)
            else:
                setattr(self.net, self.prefix+"relu3_"+str(layer_no), layer)
        bottom = getattr(self.net, self.prefix+"relu3_4")
        setattr(self.net, self.prefix+"pool3", _ave_pool(bottom, kernel_size=2, stride=2))

        bottom = getattr(self.net, self.prefix+"pool3")
        for idx, layer in enumerate(_four_conv_relu(bottom, kernel_size=3, num_output=512, pad=1)):
            layer_no = (idx - idx%2)/2+1
            if idx%2:
                setattr(self.net, self.prefix+"conv4_"+str(layer_no), layer)
            else:
                setattr(self.net, self.prefix+"relu4_"+str(layer_no), layer)
        bottom = getattr(self.net, self.prefix+"relu4_4")
        setattr(self.net, self.prefix+"pool4", _ave_pool(bottom, kernel_size=2, stride=2))

        bottom = getattr(self.net, self.prefix+"pool4")
        for idx, layer in enumerate(_four_conv_relu(bottom, kernel_size=3, num_output=512, pad=1)):
            layer_no = (idx - idx%2)/2+1
            if idx%2:
                setattr(self.net, self.prefix+"conv5_"+str(layer_no), layer)
            else:
                setattr(self.net, self.prefix+"relu5_"+str(layer_no), layer)
        bottom = getattr(self.net, self.prefix+"relu5_4")
        setattr(self.net, self.prefix+"pool5", _ave_pool(bottom, kernel_size=2, stride=2))

    def _create_blank_net(self):
        self.net = caffe.NetSpec()

def main(args):
    """
        Entry point.
    """

    # logging
    level = logging.INFO
    logging.basicConfig(format=LOG_FORMAT, datefmt="%H:%M:%S", level=level)
    logging.info("Starting style transfer.")

    style_net = Network(args.model_name.lower(),prefix="style_")
    logging.info("Successfully loaded style model {0}.".format(args.model_name))

    style_net.print_net()
    logging.info("Printed style model.")

    content_net = Network(args.model_name.lower(),prefix="content_")
    content_net.net.content_bias1 = L.Bias(param=dict(name="bias",lr_mult=1,decay_mult=0),
        bias_param=dict(bias_filler=dict(type="constant",value=0)))
    content_net.net.content_conv1_1.fn.inputs = (content_net.net.content_bias1,)
    logging.info("Successfully loaded content model {0}.".format(args.model_name))

    content_net.print_net()
    logging.info("Printed content model.")
    
    deepstyle_net = Network("blank",prefix="deepstyle_")
    logging.info("Successfully loaded deepstyle model")

    style_proto = style_net.net.to_proto()
    temp = protobuf.text_format.Merge(str(style_net.net.to_proto()),content_net.net.to_proto())
    temp2 = protobuf.text_format.Merge(str(temp),deepstyle_net.net.to_proto())
    open('tmp.prototxt','w').write(str(temp2))

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
