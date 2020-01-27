import numpy as np

# Attension:
# - Never change the value of input, which will change the result of backward


class operation(object):
    """
    Operation abstraction
    """

    def forward(self, input):
        """Forward operation, reture output"""
        raise NotImplementedError

    def backward(self, out_grad, input):
        """Backward operation, return gradient to input"""
        raise NotImplementedError


class relu(operation):
    def __init__(self):
        super(relu, self).__init__()

    def forward(self, input):
        output = np.maximum(0, input)
        return output

    def backward(self, out_grad, input):
        in_grad = (input >= 0) * out_grad
        return in_grad


class flatten(operation):
    def __init__(self):
        super(flatten, self).__init__()

    def forward(self, input):
        batch = input.shape[0]
        output = input.copy().reshape(batch, -1)
        return output

    def backward(self, out_grad, input):
        in_grad = out_grad.copy().reshape(input.shape)
        return in_grad


class matmul(operation):
    def __init__(self):
        super(matmul, self).__init__()

    def forward(self, input, weights):
        """
        # Arguments
            input: numpy array with shape (batch, in_features)
            weights: numpy array with shape (in_features, out_features)

        # Returns
            output: numpy array with shape(batch, out_features)
        """
        return np.matmul(input, weights)

    def backward(self, out_grad, input, weights):
        """
        # Arguments
            out_grad: gradient to the forward output of fc layer, with shape (batch, out_features)
            input: numpy array with shape (batch, in_features)
            weights: numpy array with shape (in_features, out_features)

        # Returns
            in_grad: gradient to the forward input with same shape as input
            w_grad: gradient to weights, with same shape as weights            
        """
        in_grad = np.matmul(out_grad, weights.T)
        w_grad = np.matmul(input.T, out_grad)
        return in_grad, w_grad


class add_bias(operation):
    def __init__(self):
        super(add_bias, self).__init__()

    def forward(self, input, bias):
        '''
        # Arugments
          input: numpy array with shape (batch, in_features)
          bias: numpy array with shape (in_features)

        # Returns
          output: numpy array with shape(batch, in_features)
        '''
        return input + bias.reshape(1, -1)

    def backward(self, out_grad, input, bias):
        """
        # Arguments
            out_grad: gradient to the forward output of fc layer, with shape (batch, out_features)
            input: numpy array with shape (batch, in_features)
            bias: numpy array with shape (out_features)
        # Returns
            in_grad: gradient to the forward input with same shape as input
            b_bias: gradient to bias, with same shape as bias
        """
        in_grad = out_grad
        b_grad = np.sum(out_grad, axis=0)
        return in_grad, b_grad


class fc(operation):
    def __init__(self):
        super(fc, self).__init__()
        self.matmul = matmul()
        self.add_bias = add_bias()

    def forward(self, input, weights, bias):
        """
        # Arguments
            input: numpy array with shape (batch, in_features)
            weights: numpy array with shape (in_features, out_features)
            bias: numpy array with shape (out_features)

        # Returns
            output: numpy array with shape(batch, out_features)
        """
        output = self.matmul.forward(input, weights)
        output = self.add_bias.forward(output, bias)
        # output = np.matmul(input, weights) + bias.reshape(1, -1)
        return output

    def backward(self, out_grad, input, weights, bias):
        """
        # Arguments
            out_grad: gradient to the forward output of fc layer, with shape (batch, out_features)
            input: numpy array with shape (batch, in_features)
            weights: numpy array with shape (in_features, out_features)
            bias: numpy array with shape (out_features)

        # Returns
            in_grad: gradient to the forward input of fc layer, with same shape as input
            w_grad: gradient to weights, with same shape as weights
            b_bias: gradient to bias, with same shape as bias
        """
        # in_grad = np.matmul(out_grad, weights.T)
        # w_grad = np.matmul(input.T, out_grad)
        # b_grad = np.sum(out_grad, axis=0)
        out_grad, b_grad = self.add_bias.backward(out_grad, input, bias)
        in_grad, w_grad = self.matmul.backward(out_grad, input, weights)
        return in_grad, w_grad, b_grad


class conv(operation):
    def __init__(self, conv_params):
        """
        # Arguments
            conv_params: dictionary, containing these parameters:
                'kernel_h': The height of kernel.
                'kernel_w': The width of kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The number of pixels padded to the bottom, top, left and right of each feature map. Here, pad = 2 means a 2-pixel border of padded with zeros
                'in_channel': The number of input channels.
                'out_channel': The number of output channels.
        """
        super(conv, self).__init__()
        self.conv_params = conv_params

    def forward(self, input, weights, bias):
        """
        # Arguments
            input: numpy array with shape (batch, in_channel, in_height, in_width)
            weights: numpy array with shape (out_channel, in_channel, kernel_h, kernel_w)
            bias: numpy array with shape (out_channel)

        # Returns
            output: numpy array with shape (batch, out_channel, out_height, out_width)
        """
        kernel_h = self.conv_params['kernel_h']  # height of kernel
        kernel_w = self.conv_params['kernel_w']  # width of kernel
        pad = self.conv_params['pad']
        stride = self.conv_params['stride']
        in_channel = self.conv_params['in_channel']
        out_channel = self.conv_params['out_channel']

        output = None

        #########################################
        # code here
        batch, in_c, in_h, in_w = input.shape
        out_h =(in_h+pad+pad-kernel_h)//stride+1
        out_w =(in_w+pad+pad-kernel_w)//stride+1
        weights_matrix = weights.reshape(weights.shape[0], -1) # shape (out_channel, in_channel x kernel_h x kernel_w)
        
        input_padded = np.pad(input,((0,0), (0,0), (pad,pad), (pad,pad)) ,'constant', constant_values=0)
        indices = [(h*stride, w*stride) for h in range(out_h) for w in range(out_w)]
        input_matrix = np.stack(
            [input_padded[:, :, x[0]:x[0]+kernel_h, x[1]:x[1]+kernel_w].reshape(batch, -1) for x in indices], 
            axis = -1
        )#shape = (batch, in_channel x kernel_h x kernel_w, out_h x out_w)

        output_matrix = np.matmul(weights_matrix, input_matrix) #shape = (batch, out_channel, out_h x out_w)
        output_matrix = output_matrix.reshape(batch,out_channel,out_h,out_w)
        output = output_matrix + bias.reshape(-1,1,1)
        #########################################

        return output

    def backward(self, out_grad, input, weights, bias):
        """
        # Arguments
            out_grad: gradient to the forward output of conv layer, with shape (batch, out_channel, out_height, out_width)
            input: numpy array with shape (batch, in_channel, in_height, in_width)
            weights: numpy array with shape (out_channel, in_channel, kernel_h, kernel_w)
            bias: numpy array with shape (out_channel)

        # Returns
            in_grad: gradient to the forward input of conv layer, with same shape as input
            w_grad: gradient to weights, with same shape as weights
            b_bias: gradient to bias, with same shape as bias
        """
        kernel_h = self.conv_params['kernel_h']  # height of kernel
        kernel_w = self.conv_params['kernel_w']  # width of kernel
        pad = self.conv_params['pad']
        stride = self.conv_params['stride']
        in_channel = self.conv_params['in_channel']
        out_channel = self.conv_params['out_channel']

        in_grad = None
        w_grad = None
        b_grad = None

        #########################################
        # code here
        batch, in_c, in_h, in_w = input.shape
        out_h = (in_h+pad+pad-kernel_h)//stride+1
        out_w = (in_w+pad+pad-kernel_w)//stride+1

        # b_grad
        b_grad = np.sum(out_grad, axis=(0,2,3))


        # w_grad
        out_grad_reshape = out_grad.reshape(out_grad.shape[0],out_grad.shape[1],-1) #shape (batch, out_channel, out_h x out_w)
        input_padded = np.pad(input,((0,0), (0,0), (pad,pad), (pad,pad)) ,'constant', constant_values=0)
        indices = [(h*stride, w*stride) for h in range(out_h) for w in range(out_w)]
        input_matrix = np.stack(
            [input_padded[:, :, x[0]:x[0]+kernel_h, x[1]:x[1]+kernel_w].reshape(batch, -1) for x in indices], 
            axis = -1
        )#shape = (batch, in_channel x kernel_h x kernel_w, out_h x out_w)
        w_grad = np.matmul(out_grad_reshape, np.transpose(input_matrix, axes=(0,2,1))) #shape = (batch, out_channel, in_channel x kernel_h x kernel_w)
        w_grad = np.sum(w_grad, axis=0).reshape(weights.shape) #shape=(out_channel, in_channel, kernel_h, kernel_w)

        # in_grad
        weights_matrix = weights.reshape(weights.shape[0], -1) # shape (out_channel, in_channel x kernel_h x kernel_w)
        out_grad_reshape_T = np.transpose(out_grad_reshape, axes = (0,2,1)) #shape (batch, out_h x out_w, out_channel)
        in_grad_hat = np.matmul(out_grad_reshape_T, weights_matrix) #shape = (batch, out_h x out_w, in_channel x kernel_h x kernel_w)
        in_grad = np.zeros((input.shape[0], input.shape[1], input.shape[2]+pad+pad, input.shape[3]+pad+pad), dtype = input.dtype)
        for h in range(out_h):
            for w in range(out_w):
                input_h = h*stride
                input_w = w*stride
                in_grad[:,:,input_h:input_h+kernel_h,input_w:input_w+kernel_w] += in_grad_hat[:,h*out_w+w].reshape(batch,in_channel,kernel_h,kernel_w)


        # weights_matrix = weights.reshape(weights.shape[0], -1) # shape (out_channel, in_channel x kernel_h x kernel_w)
        # in_grad_hat = np.matmul(weights_matrix.T, out_grad_reshape) #shape = (batch, in_channel x kernel_h x kernel_w, out_h x out_w)
        # in_grad = np.zeros((input.shape[0], input.shape[1], input.shape[2]+pad+pad, input.shape[3]+pad+pad), dtype = input.dtype)
        # # k = np.repeat(np.arange(in_c), kernel_h * kernel_w).reshape(-1, 1)
        # # i0 = np.repeat(np.arange(kernel_h), kernel_w)
        # # i0 = np.tile(i0, in_c)
        # # i1 = stride * np.repeat(np.arange(out_h), out_w)
        # # j0 = np.tile(np.arange(kernel_w), kernel_h * in_c)
        # # j1 = stride * np.tile(np.arange(out_w), out_h)
        # # i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        # # j = j0.reshape(-1, 1) + j1.reshape(1, -1)
        # # np.add.at(in_grad, (slice(None), k, i, j), in_grad_hat)

        in_grad = in_grad[:,:,pad:in_grad.shape[2]-pad, pad:in_grad.shape[3]-pad]  
        
        #########################################

        return in_grad, w_grad, b_grad


class pool(operation):
    def __init__(self, pool_params):
        """
        # Arguments
            pool_params: dictionary, containing these parameters:
                'pool_type': The type of pooling, 'max' or 'avg'
                'pool_h': The height of pooling kernel.
                'pool_w': The width of pooling kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The number of pixels that will be used to zero-pad the input in each x-y direction. Here, pad = 2 means a 2-pixel border of padding with zeros.
        """
        super(pool, self).__init__()
        self.pool_params = pool_params

    def forward(self, input):
        """
        # Arguments
            input: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            output: numpy array with shape (batch, in_channel, out_height, out_width)
        """
        pool_type = self.pool_params['pool_type']
        pool_height = self.pool_params['pool_height']
        pool_width = self.pool_params['pool_width']
        stride = self.pool_params['stride']
        pad = self.pool_params['pad']

        output = None

        #########################################
        # code here
        batch, in_c, in_h, in_w = input.shape
        out_h = (in_h+pad+pad-pool_height)//stride+1
        out_w = (in_w+pad+pad-pool_width)//stride+1
        input_padded = np.pad(input,((0,0), (0,0), (pad,pad), (pad,pad)) ,'constant', constant_values=0)
        output = np.zeros((batch,in_c,out_h,out_w),dtype = input.dtype)

        if pool_type == 'max':
            for h in range(out_h):
                for w in range(out_w):
                    input_h = h*stride
                    input_w = w*stride 
                    output[:,:,h,w]=np.amax(input_padded[:,:,input_h:input_h+pool_height, input_w:input_w+pool_width], axis=(2,3)) 
        elif pool_type == 'avg':
            for h in range(out_h):
                for w in range(out_w):
                    input_h = h*stride
                    input_w = w*stride 
                    output[:,:,h,w]=np.average(input_padded[:,:,input_h:input_h+pool_height, input_w:input_w+pool_width], axis=(2,3)) 
        else:
            raise ValueError('Doesn\'t support \'%s\' pooling.' %
                             pool_type)
        #########################################
        return output

    def backward(self, out_grad, input):
        """
        # Arguments
            out_grad: gradient to the forward output of conv layer, with shape (batch, in_channel, out_height, out_width)
            input: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            in_grad: gradient to the forward input of pool layer, with same shape as input
        """
        pool_type = self.pool_params['pool_type']
        pool_height = self.pool_params['pool_height']
        pool_width = self.pool_params['pool_width']
        stride = self.pool_params['stride']
        pad = self.pool_params['pad']

        in_grad = None

        #########################################
        # code here
        batch, in_c, in_h, in_w = input.shape
        batch2,out_c,out_h, out_w = out_grad.shape
        input_padded = np.pad(input,((0,0), (0,0), (pad,pad), (pad,pad)) ,'constant', constant_values=0)
        in_grad_padded = np.zeros_like(input_padded)

        if pool_type == 'max':
            for h in range(out_h):
                for w in range(out_w):
                    input_h = h*stride
                    input_w = w*stride 

                    maxInputIdxs = np.argmax(input_padded[:,:,input_h:input_h+pool_height, input_w:input_w+pool_width].reshape(batch,in_c,-1), axis=2) #shape (batch, in_c)
                    # maxInputHs = maxInputIdxs // pool_width + input_h
                    # maxInputWs = maxInputIdxs % pool_width + input_w
                    for sampleIdx, sample in enumerate(maxInputIdxs):
                        for channelIdx, channelMax in enumerate(sample):
                            in_grad_padded[sampleIdx,channelIdx, channelMax//pool_width+input_h, channelMax%pool_width+input_w] += out_grad[sampleIdx,channelIdx,h,w]
            in_grad = in_grad_padded[:,:,pad:in_grad_padded.shape[2]-pad, pad:in_grad_padded.shape[3]-pad] 
        elif pool_type == 'avg':
            # multiply is faster than divide so we precalculate:
            averagingFactor = 1/(pool_height*pool_width)
            for h in range(out_h):
                for w in range(out_w):
                    input_h = h*stride
                    input_w = w*stride 
                    in_grad_padded[:,:, input_h:input_h+pool_height, input_w:input_w+pool_width] += out_grad[:,:,h,w].reshape(batch,in_c,1,1)*averagingFactor
            in_grad = in_grad_padded[:,:,pad:in_grad_padded.shape[2]-pad, pad:in_grad_padded.shape[3]-pad]
        else:
            raise ValueError('Doesn\'t support \'%s\' pooling.' %
                             pool_type)
        
        #########################################

        return in_grad


class dropout(operation):
    def __init__(self, rate, training=True, seed=None):
        """
        # Arguments
            rate: float[0, 1], the probability of setting a neuron to zero
            training: boolean, apply this layer for training or not. If for training, randomly drop neurons, else DO NOT drop any neurons
            seed: int, random seed to sample from input, so as to get mask, which is convenient to check gradients. But for real training, it should be None to make sure to randomly drop neurons
            mask: the mask with value 0 or 1, corresponding to drop neurons (0) or not (1). same shape as input
        """
        self.rate = rate
        self.seed = seed
        self.training = training
        self.mask = None

    def forward(self, input):
        """
        # Arguments
            input: numpy array with any shape

        # Returns
            output: same shape as input
        """
        output = None
        if self.training:
            np.random.seed(self.seed)
            p = np.random.random_sample(input.shape)
            #########################################
            # code here
            output = np.array(input, copy=True)
            output[p<=self.rate] = 0
            output *= 1/(1-self.rate)
            self.mask = np.ones_like(input,dtype=int)
            self.mask[p<=self.rate] = 0
            #########################################
        else:
            output = input
        return output

    def backward(self, out_grad, input):
        """
        # Arguments
            out_grad: gradient to forward output of dropout, same shape as input
            input: numpy array with any shape
            mask: the mask with value 0 or 1, corresponding to drop neurons (0) or not (1). same shape as input

        # Returns
            in_grad: gradient to forward input of dropout, same shape as input
        """
        if self.training:
            #########################################
            # code here
            in_grad = np.array(out_grad, copy=True)
            in_grad[self.mask==0] = 0
            in_grad *= 1/(1-self.rate)
            #########################################
        else:
            in_grad = out_grad
        return in_grad


class softmax_cross_entropy(operation):
    def __init__(self):
        super(softmax_cross_entropy, self).__init__()

    def forward(self, input, labels):
        """
        # Arguments
            input: numpy array with shape (batch, num_class)
            labels: numpy array with shape (batch,)
            eps: float, precision to avoid overflow

        # Returns
            output: scalar, average loss
            probs: the probability of each category
        """
        # precision to avoid overflow
        eps = 1e-12

        batch = len(input)
        input_shift = input - np.max(input, axis=1, keepdims=True)
        Z = np.sum(np.exp(input_shift), axis=1, keepdims=True)

        log_probs = input_shift - np.log(Z+eps)
        probs = np.exp(log_probs)
        output = -1 * np.sum(log_probs[np.arange(batch), labels]) / batch if labels is not None else None
        return output, probs

    def backward(self, input, labels):
        """
        # Arguments
            input: numpy array with shape (batch, num_class)
            labels: numpy array with shape (batch,)
            eps: float, precision to avoid overflow

        # Returns
            in_grad: gradient to forward input of softmax cross entropy, with shape (batch, num_class)
        """
        # precision to avoid overflow
        eps = 1e-12

        batch = len(input)
        input_shift = input - np.max(input, axis=1, keepdims=True)
        Z = np.sum(np.exp(input_shift), axis=1, keepdims=True)
        log_probs = input_shift - np.log(Z+eps)
        probs = np.exp(log_probs)

        in_grad = probs.copy()
        in_grad[np.arange(batch), labels] -= 1
        in_grad /= batch
        return in_grad
