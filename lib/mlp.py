"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""
__docformat__ = 'restructedtext en'


import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
import random
import time
import matplotlib.pyplot as plt

from logistic_sgd import LogisticRegression, load_data

theano.config.floatX = 'float32'

# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


# start-snippet-2
class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )
        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        # end-snippet-3

        # keep track of model input
        self.input = input


def test_mlp(base_learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


   """

    upsample_zero = True

    datasets = load_data(dataset, upsample_zero)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]


    # compute number of minibatches for training, validation and testing
    n_train_batches = len(train_set_x) / batch_size
    n_valid_batches = len(valid_set_x) / batch_size
    n_test_batches = len(test_set_x) / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    learning_rate = T.scalar('learning_rate')

    importance_weights = T.vector('importance_weights') #weights for importance sampling

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10
    )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        T.mean(classifier.negative_log_likelihood(y))
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    # end-snippet-4


    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[x,y],
        outputs=classifier.errors(y),
    )

    validate_model = theano.function(
        inputs=[x,y],
        outputs=classifier.errors(y),
    )

    # start-snippet-5
    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    #gparams = [T.grad(cost, param) for param in classifier.params]

    gparams = []

    gradient_norm = 0.0

    #Need to recompute gparams here.  Use gradient clipping.  

    for param in classifier.params:

        grad_param = theano.gradient.jacobian(importance_weights * 0.0 + batch_size * L2_reg * classifier.L2_sqr + classifier.negative_log_likelihood(y), param)

        #(N x h1 x h2) or (N x h1)

        if param.ndim == 2:
            gradient_norm_param = T.sum(T.sqr(grad_param), axis = (1,2), keepdims = True)
            gradient_norm += T.sum(gradient_norm_param, axis = (1,2))
        elif param.ndim == 1:
            gradient_norm_param = T.sum(T.sqr(grad_param), axis = 1, keepdims = True)
            gradient_norm += T.sum(gradient_norm_param, axis = 1)
        else:
            raise Exception("Param ndim invalid")

    gradient_norm = T.sqrt(gradient_norm)

    #generate updates


    for param in classifier.params:

        grad_param = theano.gradient.jacobian(classifier.negative_log_likelihood(y), param)

        if param.ndim == 2: 
            normalized_gradient = grad_param
        elif param.ndim == 1:
            normalized_gradient = grad_param
        else:
            raise Exception("param ndim invalid")

        gparams += [T.mean(normalized_gradient, axis = 0)]

    #Compute flattened update param over all instances and then check that L2 norm for each instance is 1.  

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[x, y, importance_weights, learning_rate],
        outputs=[cost, gradient_norm, normalized_gradient],
        updates=updates,
    )
    # end-snippet-5

    get_gradient_norm = theano.function(inputs = [x,y,importance_weights], outputs = [gradient_norm])

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    x_mb_train = []
    y_mb_train = []
    x_mb_valid = []
    y_mb_valid = []
    x_mb_test = []
    y_mb_test = []
    mb_indices = []
    importance_weights_mb = []

    minibatch_index = 0
 
    gradient_norm_lst = []
    sumGradientNorm = 0.0
    numGradientNorm = 0.0

    t0 = time.time()

    maxGradientNorm = 1.0
    gradientNorms = {}

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1

        indexLst = range(0, len(train_set_x))

        random.shuffle(indexLst)
        
        if len(gradient_norm_lst) > 0:
            #plt.hist(gradient_norm_lst, numpy.arange(-0.1, 100.0, 0.1))
            #plt.title("Distribution of L2 norm of gradient for epoch: " + str(epoch - 1))
            #plt.show()

            sorted_g_lst = numpy.asarray(list(reversed(sorted(gradient_norm_lst))))

            #plt.plot(sorted_g_lst.cumsum() * 1.0 / sorted_g_lst.sum())
            #plt.title("Cumulative Distribution of Gradient Norms for epoch: " + str(epoch - 1))
            #plt.show()

            gradient_norm_lst = []

        #eps 2.0 is good
        #High importance weight -> sample with low probability
        def calculateImportanceWeight(gnorm, max_gnorm, avg_gnorm, epsilon):
            sample_prob = (gnorm + 0.0001) / (max_gnorm + 0.0001)
            return 1.0 / sample_prob
            #return 1.0

        for i in indexLst:


            #doTrain could be based on uniform sampling
            doTrain = True


            if i in gradientNorms:
                imp_weight_instance = calculateImportanceWeight(gnorm = gradientNorms[i], max_gnorm = maxGradientNorm, avg_gnorm = sumGradientNorm * 1.0 / numGradientNorm, epsilon = 0.01)
            else:
                imp_weight_instance = 100.0

            if random.uniform(0,1) < 1.0 / imp_weight_instance: 
                doTrain = True
            else:
                doTrain = False

            if doTrain: 
                x_mb_train += [train_set_x[i]]
                y_mb_train += [train_set_y[i]]
                mb_indices += [i]
                importance_weights_mb += [imp_weight_instance]

            if len(gradientNorms) > 0:
                learning_rate = base_learning_rate
            else:
                learning_rate = base_learning_rate

            if len(x_mb_train) == batch_size: 
                minibatch_avg_cost, gradient_norm, ng = train_model(numpy.asarray(x_mb_train, dtype = 'float32'), numpy.asarray(y_mb_train, dtype = 'int32'), numpy.asarray(importance_weights_mb, dtype = 'float32'), numpy.asarray(learning_rate, dtype = 'float32'))

		print "ng shape", ng[0]
		print ng[1]
		print ng[2]

                if True:
                    if random.uniform(0,1) < 0.01 or sum(gradient_norm).tolist() == float('nan'):
                        for k in range(0, 1): 
                            print "================================================="
                            print sum(gradient_norm).tolist() == float('nan')
                            if mb_indices[k] in gradientNorms:
                                print "last grad norm", gradientNorms[mb_indices[k]]
                            print "gradient norm", gradient_norm[k]
                            print "class", y_mb_train[k]
                            print "sample prob", 1.0 / importance_weights_mb[k]
                    for k in range(0, batch_size):
                        gradient_norm_lst += [gradient_norm[k]]
                        gradientNorms[mb_indices[k]] = gradient_norm[k]
                        numGradientNorm += 1.0
                        sumGradientNorm += gradient_norm[k]

                minibatch_index += 1
                x_mb_train = []
                y_mb_train = []
                mb_indices = []
                importance_weights_mb = []

                iter = (epoch - 1) * n_train_batches + minibatch_index
    
                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = []
                
                    for j in range(0, len(valid_set_x)): 
                        x_mb_valid += [valid_set_x[j]]
                        y_mb_valid += [valid_set_y[j]]

                        if len(x_mb_valid) == batch_size: 
                            validation_losses += [validate_model(x_mb_valid, y_mb_valid)]

                            x_mb_valid = []
                            y_mb_valid = []

                    this_validation_loss = numpy.mean(numpy.asarray(validation_losses))

                    print "time", time.time() - t0
                    t0 = time.time()

                    print(
                        'epoch %i, minibatch %i/%i, iteration %i, validation error %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            iter,
                            n_train_batches,
                            this_validation_loss * 100.
                        )
                    )

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if (
                            this_validation_loss < best_validation_loss *
                            improvement_threshold
                        ):
                            patience = max(patience, iter * patience_increase)
    
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        test_losses = []
                    
                        for j in range(0, len(test_set_x)): 
                            x_mb_test += [test_set_x[j]]
                            y_mb_test += [test_set_y[j]]

                            if len(x_mb_test) == batch_size: 
                                test_losses += [test_model(x_mb_test, y_mb_test)]

                                x_mb_test = []
                                y_mb_test = []
                    
                    
                        test_score = numpy.mean(test_losses)
    
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))

                if patience <= iter:
                    done_looping = False
                    #break

        maxGradientNorm = max(gradientNorms.values())

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    test_mlp()