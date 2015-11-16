# system imports
import argparse
import logging
import os
import timeit

# library imports
import caffe
import numpy as np
from scipy.misc import imsave
from skimage import img_as_ubyte
from skimage.transform import rescale

import IPython

# logging
LOG_FORMAT = "%(filename)s:%(funcName)s:%(asctime)s.%(msecs)03d -- %(message)s"

VGG19_WEIGHTS = {"content": {"conv4_2": 1},
                 "style": {"conv1_1": 0.2,
                           "conv2_1": 0.2,
                           "conv3_1": 0.2,
                           "conv4_1": 0.2,
                           "conv5_1": 0.2}}

# argparse
parser = argparse.ArgumentParser(description="Transfer the style of one image to another.",
                                 usage="style.py -s <style_image> -c <content_image>")
parser.add_argument("-s", "--style-img", type=str, required=True, help="input style (art) image")
parser.add_argument("-c", "--content-img", type=str, required=True, help="input content image")
parser.add_argument("-g", "--gpu-id", default=-1, type=int, required=False, help="GPU device number")
parser.add_argument("-m", "--model", default="vgg19", type=str, required=False, help="model to use")
parser.add_argument("-i", "--init", default="content", type=str, required=False, help="initialization strategy")
parser.add_argument("-r", "--ratio", default="1e5", type=str, required=False, help="style-to-content ratio")
parser.add_argument("-n", "--num-iters", default=512, type=int, required=False, help="L-BFGS iterations")
parser.add_argument("-l", "--length", default=512, type=float, required=False, help="maximum image length")
parser.add_argument("-v", "--verbose", action="store_true", required=False, help="print minimization outputs")
parser.add_argument("-o", "--output", default=None, required=False, help="output path")

class Network(object):
    """
        Load in a network
    """

    def __init__(self, model_name, style_root=''):
        """
            Initialize the model used for style transfer.

            :param str model_name:
                Model to use.

            :param bool use_pbar:
                Use progressbar flag.
        """
        base_path = os.path.join(style_root,"models", model_name)

        # deepstyle network
        if model_name == "deepstyle":
            model_file = os.path.join(base_path, "DEEP_STYLE.prototxt")
            mean_file = os.path.join(base_path, "ilsvrc_2012_mean.npy")
            pretrained_file = "none"
            weights = "none";

        # vgg19
        elif model_name == "vgg19":
            model_file = os.path.join(base_path, "VGG_ILSVRC_19_layers_deploy.prototxt")
            pretrained_file = os.path.join(base_path, "VGG_ILSVRC_19_layers.caffemodel")
            mean_file = os.path.join(base_path, "ilsvrc_2012_mean.npy")
            weights = VGG19_WEIGHTS

        # vgg16
        elif model_name == "vgg16":
            model_file = os.path.join(base_path, "VGG_ILSVRC_16_layers_deploy.prototxt")
            pretrained_file = os.path.join(base_path, "VGG_ILSVRC_16_layers.caffemodel")
            mean_file = os.path.join(base_path, "ilsvrc_2012_mean.npy")
            weights = VGG16_WEIGHTS

        # googlenet
        elif model_name == "googlenet":
            model_file = os.path.join(base_path, "deploy.prototxt")
            pretrained_file = os.path.join(base_path, "googlenet_style.caffemodel")
            mean_file = os.path.join(base_path, "ilsvrc_2012_mean.npy")
            weights = GOOGLENET_WEIGHTS

        # caffenet
        elif model_name == "caffenet":
            model_file = os.path.join(base_path, "deploy.prototxt")
            pretrained_file = os.path.join(base_path, "bvlc_reference_caffenet.caffemodel")
            mean_file = os.path.join(base_path, "ilsvrc_2012_mean.npy")
            weights = CAFFENET_WEIGHTS

        else:
            assert False, "model not available"

        # add model and weights
        self.load_model(model_file, pretrained_file, mean_file)
        if type(weights) is dict:
            self.weights = weights.copy()
        else:
            self.weights = {"content":{"None"},"style":{"None"}}
        self.layers = []
        for layer in self.net.params.keys():
            if layer in self.weights["style"] or layer in self.weights["content"]:
                self.layers.append(layer)

    def load_model(self, model_file, pretrained_file, mean_file):
        """
            Loads specified model from caffe install (see caffe docs).

            :param str model_file:
                Path to model protobuf.

            :param str pretrained_file:
                Path to pretrained caffe model.

            :param str mean_file:
                Path to mean file.
        """

        assert os.path.isfile(model_file), "Model protobuf "+model_file+" not found."

        # load net (supressing stderr output)
        null_fds = os.open(os.devnull, os.O_RDWR)
        out_orig = os.dup(2)
        os.dup2(null_fds, 2)
        if pretrained_file == "none":
            net = caffe.Net(model_file, caffe.TEST)
        elif (os.path.isfile(pretrained_file)):
            net = caffe.Net(model_file, pretrained_file, caffe.TEST)
        else:
            #TODO: program does not exit correctly
            assert os.path.isfile(pretrained_file), "Pretrained file "+pretrained_file+" not found."
            print "ERROR: load_model: Unknown error." # Should not happen - assert should fail
            return
        os.dup2(out_orig, 2)
        os.close(null_fds)

        # all models are assumed to be trained on imagenet data
        transformer = caffe.io.Transformer({"data": net.blobs["data"].data.shape})
        transformer.set_mean("data", np.load(mean_file).mean(1).mean(1))
        transformer.set_channel_swap("data", (2,1,0))
        transformer.set_transpose("data", (2,0,1))
        transformer.set_raw_scale("data", 255)

        # add net parameters
        self.net = net
        self.transformer = transformer

def main(args):
    """
        Entry point.
    """

    # logging
    level = logging.INFO if args.verbose else logging.DEBUG
    logging.basicConfig(format=LOG_FORMAT, datefmt="%H:%M:%S", level=level)
    logging.info("Starting style transfer.")

    # set GPU/CPU mode
    if args.gpu_id == -1:
        caffe.set_mode_cpu()
        logging.info("Running net on CPU.")
    else:
        caffe.set_device(args.gpu_id)
        caffe.set_mode_gpu()
        logging.info("Running net on GPU {0}.".format(args.gpu_id))

    # load images
    img_style = caffe.io.load_image(args.style_img)
    img_content = caffe.io.load_image(args.content_img)
    logging.info("Successfully loaded images.")


    # load nets
    style_net = Network(args.model.lower())
    logging.info("Successfully loaded style model {0}.".format(args.model))

    content_net = Network(args.model.lower())
    logging.info("Successfully loaded content model {0}.".format(args.model))

    combined_net = Network('deepstyle')
    logging.info("Successfully loaded deepstyle model {0}.".format(args.model))

    ## perform style transfer
    #start = timeit.default_timer()
    ##n_iters = st.transfer_style(img_style, img_content, length=args.length, 
    ##                            init=args.init, ratio=np.float(args.ratio), 
    ##                            n_iter=args.num_iters, verbose=args.verbose)
    #end = timeit.default_timer()
    #logging.info("Ran {0} iterations in {1:.0f}s.".format(n_iters, end-start))
    #img_out = st.get_generated()

    ## output path
    #if args.output is not None:
    #    out_path = args.output
    #else:
    #    out_path_fmt = (os.path.splitext(os.path.split(args.content_img)[1])[0], 
    #                    os.path.splitext(os.path.split(args.style_img)[1])[0], 
    #                    args.model, args.init, args.ratio, args.num_iters)
    #    out_path = "outputs/{0}-{1}-{2}-{3}-{4}-{5}.jpg".format(*out_path_fmt)

    ## DONE!
    #if os.path.dirname(out_path):
    #    if not os.path.exists(os.path.dirname(out_path)):
    #        os.makedirs(os.path.dirname(out_path))
    #imsave(out_path, img_as_ubyte(img_out))
    #logging.info("Output saved to {0}.".format(out_path))
    

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
