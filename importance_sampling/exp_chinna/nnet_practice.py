__author__ = 'chinna'
import theano
from theano import tensor as T
import numpy as np
import matplotlib.pyplot as plt
from theano import shared
from theano import function

import theano
from theano import tensor as T
import numpy as np
from load import mnist
from load import mnist_with_noise
#from foxhound.utils.vis import grayscale_grid_vis, unit_scale
from scipy.misc import imsave

nhidden = 1

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def sgd(cost, params, lr=0.05):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates

def model(X, w_h):
    h0 = T.nnet.sigmoid(T.dot(X, w_h[0]))
    pyx = T.nnet.softmax(T.dot(h0,w_h[1]))
    return pyx

trX, teX, trY, teY = mnist(ntrain=5000,ntest=200,onehot=True)


ntX,ntY,seq = mnist_with_noise([trX,trY],10)
print seq


X = T.fmatrix()
Y = T.fmatrix()
#grads = T.fvector()

w_h = [init_weights((784, 625)), init_weights((625, 10))]


py_x = model(X, w_h)
y_x = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
cost_for_norm = T.sum(T.nnet.categorical_crossentropy(py_x,Y))
params = w_h
updates = sgd(cost, params)
grads = T.grad(cost=cost,wrt=params)
grad_for_norm = T.grad(cost=cost_for_norm,wrt=params)

train = theano.function(inputs=[X, Y], outputs=[cost,grads[0],grads[1]], updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)
get_grad = theano.function(inputs=[X,Y],outputs=[grad_for_norm[0],grad_for_norm[1]], allow_input_downcast=True)

mb_size = 128
for i in range(10):
    grad_list = []
    for start, end in zip(range(0, len(trX), mb_size), range(mb_size, len(trX), mb_size)):
        cost,grads[0],grads[1] = train(trX[start:end], trY[start:end])
    print np.mean(np.argmax(teY, axis=1) == predict(teX))

noisy_grads = []
normal_grads = []
mb_size = 1
for i in seq:
    grad = get_grad(trX[i:i+1], trY[i:i+1])
    norm = np.linalg.norm(grad[0])
    if i < 500:
        noisy_grads.append(norm)
    else:
        normal_grads.append(norm)

#grad_list.append(np.linalg.norm(grads[0]))
print "noisy  : mean,var - " ,np.mean(noisy_grads),np.var(noisy_grads)
print "normal : mean,var - " , np.mean(normal_grads),np.var(normal_grads)
plt.plot(noisy_grads)
plt.plot(normal_grads)

plt.savefig('grad0.jpeg')


