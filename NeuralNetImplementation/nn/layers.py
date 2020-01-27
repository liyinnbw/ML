import numpy as np
from utils.initializers import *
from .operations import *


class Layer(object):
    """
    Layer abstraction
    """

    def __init__(self, name):
        """Initialization"""
        self.name = name
        self.training = True  # The phrase, if for training then true
        self.trainable = False  # Whether there are parameters in this layer that can be trained
        self.output_shape = None

    def update_output_shape(self, input_shape):
        raise NotImplementedError

    def forward(self, input):
        """Forward pass, reture output"""
        raise NotImplementedError

    def backward(self, out_grad, input):
        """Backward pass, return gradient to input"""
        raise NotImplementedError

    def update(self, optimizer):
        """Update parameters in this layer"""
        pass

    def set_mode(self, training):
        """Set the phrase/mode into training (True) or tesing (False)"""
        self.training = training

    def set_trainable(self, trainable):
        """Set the layer can be trainable (True) or not (False)"""
        self.trainable = trainable

    def get_params(self, prefix):
        """Reture parameters and gradient of this layer"""
        return None

class FCLayer(Layer):
    def __init__(self, out_features, name='fclayer', initializer=Gaussian(std=0.01)):
        """Initialization

        # Arguments
            in_features: int, the number of input features
            out_features: int, the numbet of required output features
            initializer: Initializer class, to initialize weights
        """
        super(FCLayer, self).__init__(name=name)
        self.fc = fc()
        self.trainable = True
        self.out_features = out_features
        self.initializer = initializer

    def update_output_shape(self, input_shape):
        in_features = input_shape[1]
        out_features = self.out_features
        self.output_shape = (input_shape[0], out_features)
        self.weights = self.initializer.initialize((in_features, out_features))
        self.bias = np.zeros(out_features)
        self.w_grad = np.zeros(self.weights.shape)
        self.b_grad = np.zeros(self.bias.shape)

    def forward(self, input):
        output = self.fc.forward(input, self.weights, self.bias)
        return output

    def backward(self, out_grad, input):
        in_grad, self.w_grad, self.b_grad = self.fc.backward(
            out_grad, input, self.weights, self.bias)
        return in_grad

    def update(self, params):
        """Update parameters (self.weights and self.bias) with new params

        # Arguments
            params: dictionary, one key contains 'weights' and the other contains 'bias'

        # Returns
            none
        """
        for k, v in params.items():
            if 'weights' in k:
                self.weights = v
            else:
                self.bias = v

    def get_params(self, prefix):
        """Return parameters (self.weights and self.bias) as well as gradient (self.w_grad and self.b_grad)

        # Arguments
            prefix: string, to contruct prefix of keys in the dictionary (usually is the layer-ith)

        # Returns
            params: dictionary, store parameters of this layer, one key contains 'weights' and the other contains 'bias'
            grads: dictionary, store gradient of this layer, one key contains 'weights' and the other contains 'bias'

            None: if not trainable
        """
        if self.trainable:
            params = {
                prefix+':'+self.name+'/weights': self.weights,
                prefix+':'+self.name+'/bias': self.bias
            }
            grads = {
                prefix+':'+self.name+'/weights': self.w_grad,
                prefix+':'+self.name+'/bias': self.b_grad
            }
            return params, grads
        else:
            return None


class Convolution(Layer):
    def __init__(self, conv_params, initializer=Gaussian(std=0.01), name='conv'):
        """Initialization

        # Arguments
            conv_params: dictionary, containing these parameters:
                'kernel_h': The height of kernel.
                'kernel_w': The width of kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The number of pixels padded to the bottom, top, left and right of each feature map. Here, pad=2 means a 2-pixel border of padded with zeros
                'in_channel': The number of input channels.
                'out_channel': The number of output channels.
            initializer: Initializer class, to initialize weights
        """
        super(Convolution, self).__init__(name=name)
        self.conv_params = conv_params
        self.conv = conv(conv_params)

        self.trainable = True

        self.weights = initializer.initialize(
            (conv_params['out_channel'], conv_params['in_channel'], conv_params['kernel_h'], conv_params['kernel_w']))
        self.bias = np.zeros((conv_params['out_channel']))

        self.w_grad = np.zeros(self.weights.shape)
        self.b_grad = np.zeros(self.bias.shape)

        
    def update_output_shape(self, input_shape):
        in_batch, in_c, in_h, in_w = input_shape
        pad = self.conv_params['pad']
        stride = self.conv_params['stride']
        kernel_h = self.conv_params['kernel_h']
        kernel_w = self.conv_params['kernel_w']
        out_c = self.conv_params['out_channel']
        out_h =(in_h+pad+pad-kernel_h)//stride+1
        out_w =(in_w+pad+pad-kernel_w)//stride+1
        self.output_shape = (input_shape[0], out_c, out_h, out_w)

    def forward(self, input):
        output = self.conv.forward(input, self.weights, self.bias)
        return output

    def backward(self, out_grad, input):
        in_grad, self.w_grad, self.b_grad = self.conv.backward(
            out_grad, input, self.weights, self.bias)
        return in_grad

    def update(self, params):
        """Update parameters (self.weights and self.bias) with new params

        # Arguments
            params: dictionary, one key contains 'weights' and the other contains 'bias'

        # Returns
            none
        """
        for k, v in params.items():
            if 'weights' in k:
                self.weights = v
            else:
                self.bias = v

    def get_params(self, prefix):
        """Return parameters (self.weights and self.bias) as well as gradient (self.w_grad and self.b_grad)

        # Arguments
            prefix: string, to contruct prefix of keys in the dictionary (usually is the layer-ith)

        # Returns
            params: dictionary, store parameters of this layer, one key contains 'weights' and the other contains 'bias'
            grads: dictionary, store gradient of this layer, one key contains 'weights' and the other contains 'bias'

            None: if not trainable
        """
        if self.trainable:
            params = {
                prefix+':'+self.name+'/weights': self.weights,
                prefix+':'+self.name+'/bias': self.bias
            }
            grads = {
                prefix+':'+self.name+'/weights': self.w_grad,
                prefix+':'+self.name+'/bias': self.b_grad
            }
            return params, grads
        else:
            return None


class ReLU(Layer):
    def __init__(self, name='relu'):
        """Initialization
        """
        super(ReLU, self).__init__(name=name)
        self.relu = relu()

    def update_output_shape(self, input_shape):
        self.output_shape = input_shape

    def forward(self, input):
        """Forward pass

        # Arguments
            input: numpy array

        # Returns
            output: numpy array
        """
        output = self.relu.forward(input)
        return output

    def backward(self, out_grad, input):
        """Backward pass

        # Arguments
            out_grad: numpy array, gradient to output
            input: numpy array, same with forward input

        # Returns
            in_grad: numpy array, gradient to input 
        """
        in_grad = self.relu.backward(out_grad, input)
        return in_grad


class Pooling(Layer):
    def __init__(self, pool_params, name='pooling'):
        """Initialization

        # Arguments
            pool_params is a dictionary, containing these parameters:
                'pool_type': The type of pooling, 'max' or 'avg'
                'pool_h': The height of pooling kernel.
                'pool_w': The width of pooling kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The number of pixels that will be used to zero-pad the input in each x-y direction. Here, pad=2 means a 2-pixel border of padding with zeros.
        """
        super(Pooling, self).__init__(name=name)
        self.pool_params = pool_params
        self.pool = pool(pool_params)

    def update_output_shape(self, input_shape):
        in_batch, in_c, in_h, in_w = input_shape
        pool_height = self.pool_params['pool_height']
        pool_width = self.pool_params['pool_width']
        stride = self.pool_params['stride']
        pad = self.pool_params['pad']
        out_h = (in_h+pad+pad-pool_height)//stride+1
        out_w = (in_w+pad+pad-pool_width)//stride+1
        self.output_shape = (input_shape[0], in_c,out_h,out_w)

    def forward(self, input):
        output = self.pool.forward(input)
        return output

    def backward(self, out_grad, input):
        in_grad = self.pool.backward(out_grad, input)
        return in_grad


class Dropout(Layer):
    def __init__(self, rate, name='dropout', seed=None):
        """Initialization

        # Arguments
            rate: float [0, 1], the probability of setting a neuron to zero
            seed: int, random seed to sample from input, so as to get mask, which is convenient to check gradients. But for real training, it should be None to make sure to randomly drop neurons
        """
        super(Dropout, self).__init__(name=name)
        self.rate = rate
        self.seed = seed
        self.dropout = dropout(rate, self.training, seed)

    def update_output_shape(self, input_shape):
        self.output_shape = input_shape

    def forward(self, input):
        output = self.dropout.forward(input)
        return output

    def backward(self, out_grad, input):
        in_grad = self.dropout.backward(out_grad, input)
        return in_grad

    def set_mode(self, training):
        """Set the phrase/mode into training (True) or tesing (False)"""
        self.training = training
        self.dropout.training = training

class Flatten(Layer):
    def __init__(self, name='flatten', seed=None):
        """Initialization
        """
        super(Flatten, self).__init__(name=name)
        self.flatten = flatten()

    def update_output_shape(self, input_shape):
        shape = 1
        for i in input_shape:
            if i!=None:
                shape *= i
        self.output_shape = (input_shape[0], shape)

    def forward(self, input):
        """Forward pass

        # Arguments
            input: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            output: numpy array with shape (batch, in_channel*in_height*in_width)
        """
        output = self.flatten.forward(input)
        return output

    def backward(self, out_grad, input):
        """Backward pass

        # Arguments
            out_grad: numpy array with shape (batch, in_channel*in_height*in_width), gradient to output
            input: numpy array with shape (batch, in_channel, in_height, in_width), same with forward input

        # Returns
            in_grad: numpy array with shape (batch, in_channel, in_height, in_width), gradient to input 
        """
        in_grad = self.flatten.backward(out_grad, input)
        return in_grad

class SoftmaxCrossEntropy(Layer):
    def __init__(self, name='softmax', seed=None):
        """Initialization

        # Arguments
            seed: int, random seed to sample from input, so as to get mask, which is convenient to check gradients. But for real training, it should be None to make sure to randomly drop neurons
        """
        super(SoftmaxCrossEntropy, self).__init__(name=name)
        self.softmax_cross_entropy = softmax_cross_entropy()

    def update_output_shape(self, input_shape):
        self.output_shape = input_shape

    def forward(self, input, labels):
        output, probs = self.softmax_cross_entropy.forward(input, labels)
        return output, probs

    def backward(self, input, labels):
        in_grad = self.softmax_cross_entropy.backward(input, labels)
        return in_grad

