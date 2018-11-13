from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        out1, cache1 = affine_relu_forward(X, W1, b1) # affine - relu
        scores, cache2 = affine_forward(out1, W2, b2) # - affine
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        # Compute softmax loss
        data_loss, dout = softmax_loss(scores, y)
        
        # Add L2 regularization to the loss.
        reg_loss = 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        
        # Compute total loss
        loss = data_loss + reg_loss
        
        # Compute grads
        dout1, dW2, db2 = affine_backward(dout, cache2)
        dX, dW1, db1 = affine_relu_backward(dout1, cache1)
        
        # Add L2 regularization to the gradient.
        dW2 += self.reg * W2
        dW1 += self.reg * W1
        
        # Store grads in dictionary
        grads['W2'] = dW2
        grads['b2'] = db2
        grads['W1'] = dW1
        grads['b1'] = db1

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.hidden_dims = hidden_dims
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        
        num_hidden_layers = len(hidden_dims)
        
        for ctr in range(num_hidden_layers):
            
            layer_num = ctr + 1
            
            if layer_num == 1:
                self.params['W'+str(layer_num)] = weight_scale * np.random.randn(input_dim, hidden_dims[ctr])
                self.params['b'+str(layer_num)] = np.zeros(hidden_dims[ctr])
                
            else:
                self.params['W'+str(layer_num)] = weight_scale * np.random.randn(hidden_dims[ctr-1], hidden_dims[ctr])
                self.params['b'+str(layer_num)] = np.zeros(hidden_dims[ctr])

        if self.normalization=='batchnorm': # Initialise bacthnorm params

            for ctr in range(num_hidden_layers):

                layer_num = ctr + 1

                self.params['gamma'+str(layer_num)] = np.ones(hidden_dims[ctr]) # Init scale parm for batchnorm
                self.params['beta'+str(layer_num)] = np.zeros(hidden_dims[ctr]) # Init shift param for batchnorm
                
                
        self.params['W'+str(num_hidden_layers+1)] = weight_scale * np.random.randn(hidden_dims[num_hidden_layers-1], num_classes)
        self.params['b'+str(num_hidden_layers+1)] = np.zeros(num_classes)

        #print(self.params['W3'])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        
        num_hidden_layers = len(self.hidden_dims)
        
        af_out = [None] * (self.num_layers)
        af_cache = [None] * (self.num_layers)

        relu_out = [None] * (num_hidden_layers)
        relu_cache = [None] * (num_hidden_layers)

        drop_out = [None] * (num_hidden_layers+1)
        drop_cache = [None] * (num_hidden_layers+1)

        ctr = 0
        
        '''
        # Implementation using layer_utils without provision for batchnorm
        
        out = [None]*(num_hidden_layers+1)
        cache = [None]*(num_hidden_layers+1)
        
        out[ctr], cache[ctr] = affine_relu_forward(X, self.params['W'+str(ctr+1)], self.params['b'+str(ctr+1)])
        
        for ctr in range(num_hidden_layers-1):
            out[ctr+1], cache[ctr+1] = affine_relu_forward(out[ctr], self.params['W'+str(ctr+2)], self.params['b'+str(ctr+2)]) # (affine - relu)* (L-1)
        
        out[ctr+2], cache[ctr+2] = affine_forward(out[ctr+1], self.params['W'+str(ctr+3)], self.params['b'+str(ctr+3)]) # - affine
        scores = out[ctr+2]
        
        '''

        # Input affine layer is the same with/without batchnorm
        af_out[ctr], af_cache[ctr] = affine_forward(X, self.params['W'+str(ctr+1)], self.params['b'+str(ctr+1)])


        if self.normalization=='batchnorm': # Implementation with batchnorm

            bn_out = [None] * (num_hidden_layers)
            bn_cache = [None] * (num_hidden_layers)

            bn_out[ctr], bn_cache[ctr] = batchnorm_forward(af_out[ctr], self.params['gamma'+str(ctr+1)], self.params['beta'+str(ctr+1)], self.bn_params[ctr])
            relu_out[ctr], relu_cache[ctr] = relu_forward(bn_out[ctr])

            if self.use_dropout:
                drop_out[ctr], drop_cache[ctr] = dropout_forward(relu_out[ctr], self.dropout_param)
            else:
                drop_out[ctr] = relu_out[ctr]

            for ctr in range(num_hidden_layers-1):

                af_out[ctr+1], af_cache[ctr+1] = affine_forward(drop_out[ctr], self.params['W'+str(ctr+2)], self.params['b'+str(ctr+2)]) # (affine)* (L-1)
                bn_out[ctr+1], bn_cache[ctr+1] = batchnorm_forward(af_out[ctr+1], self.params['gamma'+str(ctr+2)], self.params['beta'+str(ctr+2)], self.bn_params[ctr+1])
                relu_out[ctr+1], relu_cache[ctr+1] = relu_forward(bn_out[ctr+1])
                if self.use_dropout:
                    drop_out[ctr+1], drop_cache[ctr+1] = dropout_forward(relu_out[ctr+1], self.dropout_param)
                else:
                    drop_out[ctr+1] = relu_out[ctr+1]

        else: # Implementation without batchnorm

            relu_out[ctr], relu_cache[ctr] = relu_forward(af_out[ctr])

            if self.use_dropout:
                drop_out[ctr], drop_cache[ctr] = dropout_forward(relu_out[ctr], self.dropout_param)
            else:
                drop_out[ctr] = relu_out[ctr]

            for ctr in range(num_hidden_layers-1):

                af_out[ctr+1], af_cache[ctr+1] = affine_forward(drop_out[ctr], self.params['W'+str(ctr+2)], self.params['b'+str(ctr+2)]) # (affine)* (L-1)
                relu_out[ctr+1], relu_cache[ctr+1] = relu_forward(af_out[ctr+1])

                if self.use_dropout:
                    drop_out[ctr+1], drop_cache[ctr+1] = dropout_forward(relu_out[ctr+1], self.dropout_param)
                else:
                    drop_out[ctr+1] = relu_out[ctr+1]



        # Final layer is the same with/without bacthnorm
        af_out[ctr+2], af_cache[ctr+2] = affine_forward(drop_out[ctr+1], self.params['W'+str(ctr+3)], self.params['b'+str(ctr+3)]) # - affine
        scores = af_out[ctr+2]

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        dout = [None] * (self.num_layers) # Because also needed for final affine layer
        da = [None] * (num_hidden_layers) # Only for affine - [batchnorm] - relu blocks; so L-1 times
        ddrop = [None] * (num_hidden_layers)

        # Compute softmax loss
        data_loss, dout[num_hidden_layers] = softmax_loss(scores, y)
        
        # Add L2 regularization to the loss.
        reg_factor = 0.0
        
        for ctr in range(num_hidden_layers):
            reg_factor += np.sum(np.square(self.params['W'+str(ctr+1)]))
        
        reg_loss = 0.5 * self.reg * reg_factor
        
        # Compute total loss
        loss = data_loss + reg_loss

        # Compute grads
        
        # Affine backward layer 0 it's the same with/without batchnorm
        dout[num_hidden_layers-1], grads['W'+str(num_hidden_layers+1)], grads['b'+str(num_hidden_layers+1)] = affine_backward(dout[num_hidden_layers], af_cache[num_hidden_layers])


        if self.normalization=='batchnorm': # Implementation with batchnorm

            db = [None]*(num_hidden_layers)

            for ctr in range(num_hidden_layers-1):

                if self.use_dropout:
                    ddrop[num_hidden_layers-1-ctr] = dropout_backward(dout[num_hidden_layers-1-ctr], drop_cache[num_hidden_layers-1-ctr])
                else:
                    ddrop[num_hidden_layers-1-ctr] = dout[num_hidden_layers-1-ctr]


                da[num_hidden_layers-1-ctr] = relu_backward(ddrop[num_hidden_layers-1-ctr], relu_cache[num_hidden_layers-1-ctr])
                db[num_hidden_layers-1-ctr], grads['gamma'+str(num_hidden_layers-ctr)], grads['beta'+str(num_hidden_layers-ctr)] = batchnorm_backward(da[num_hidden_layers-1-ctr], bn_cache[num_hidden_layers-1-ctr])
                dout[num_hidden_layers-2-ctr], grads['W'+str(num_hidden_layers-ctr)], grads['b'+str(num_hidden_layers-ctr)] = affine_backward(db[num_hidden_layers-1-ctr], af_cache[num_hidden_layers-1-ctr])

            if self.use_dropout:
                ddrop[num_hidden_layers-2-ctr] = dropout_backward(dout[num_hidden_layers-2-ctr], drop_cache[num_hidden_layers-2-ctr])
            else:
                ddrop[num_hidden_layers-2-ctr] = dout[num_hidden_layers-2-ctr]

            da[num_hidden_layers-2-ctr] = relu_backward(ddrop[num_hidden_layers-2-ctr], relu_cache[num_hidden_layers-2-ctr])
            db[num_hidden_layers-2-ctr], grads['gamma'+str(num_hidden_layers-1-ctr)], grads['beta'+str(num_hidden_layers-1-ctr)] = batchnorm_backward(da[num_hidden_layers-2-ctr], bn_cache[num_hidden_layers-2-ctr])
            dX, grads['W'+str(num_hidden_layers-1-ctr)], grads['b'+str(num_hidden_layers-1-ctr)] = affine_backward(db[num_hidden_layers-2-ctr], af_cache[num_hidden_layers-2-ctr])


        else:       # Implementation without batchnorm

            for ctr in range(num_hidden_layers-1):

                if self.use_dropout:
                    ddrop[num_hidden_layers-1-ctr] = dropout_backward(dout[num_hidden_layers-1-ctr], drop_cache[num_hidden_layers-1-ctr])

                else:
                    ddrop[num_hidden_layers-1-ctr] = dout[num_hidden_layers-1-ctr]

                da[num_hidden_layers-1-ctr] = relu_backward(ddrop[num_hidden_layers-1-ctr], relu_cache[num_hidden_layers-1-ctr])
                dout[num_hidden_layers-2-ctr], grads['W'+str(num_hidden_layers-ctr)], grads['b'+str(num_hidden_layers-ctr)] = affine_backward(da[num_hidden_layers-1-ctr], af_cache[num_hidden_layers-1-ctr])
        
            if self.use_dropout:
                ddrop[num_hidden_layers-2-ctr] = dropout_backward(dout[num_hidden_layers-2-ctr], drop_cache[num_hidden_layers-2-ctr])

            else:
                ddrop[num_hidden_layers-2-ctr] = dout[num_hidden_layers-2-ctr]

            da[num_hidden_layers-2-ctr] = relu_backward(ddrop[num_hidden_layers-2-ctr], relu_cache[num_hidden_layers-2-ctr])
            dX, grads['W'+str(num_hidden_layers-1-ctr)], grads['b'+str(num_hidden_layers-1-ctr)] = affine_backward(da[num_hidden_layers-2-ctr], af_cache[num_hidden_layers-2-ctr])

        
        # Add L2 regularization to the gradient.
        for ctr in range(num_hidden_layers):
            grads['W'+str(ctr+1)] += self.reg * self.params['W'+str(ctr+1)]
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
