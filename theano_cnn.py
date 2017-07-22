import theano
from theano import tensor as T
from theano.tensor.nnet import softmax, categorical_crossentropy
import numpy as np
from theano_helper import ConvPoolLayer, HiddenLayer, load_data, write_graph

rng = np.random.RandomState(23455)
batch_size = 128
num_classes = 10
epochs = 12
lr = 0.01
img_rows, img_cols = 28, 28
datasets = load_data('mnist.pkl.gz')
train_x, train_y = datasets[0]
valid_x, valid_y = datasets[1]
test_x, test_y = datasets[2]
n_train_batches = train_x.get_value(borrow=True).shape[0] // batch_size
n_test_batches = test_x.get_value(borrow=True).shape[0] // batch_size

x = T.matrix('x')
y = T.imatrix('y')
index = T.lscalar()  # index to a [mini]batch
layer0_input = x.reshape((batch_size, 1, img_rows, img_cols))

layer0 = ConvPoolLayer(rng,
                       input=layer0_input,
                       image_shape=(batch_size, 1, img_rows, img_cols),
                       filter_shape=(32, 1, 3, 3),
                       poolsize=(2, 2))

layer1_input = layer0.output.flatten(2)

layer1 = HiddenLayer(rng,
                     input=layer1_input,
                     n_in=32 * 13 * 13,
                     n_out=num_classes,
                     activation=softmax)

cost = T.mean(categorical_crossentropy(layer1.output, y))
acc = T.mean(T.eq(T.argmax(layer1.output,axis=1), T.argmax(y, axis=1)))
params = layer1.params + layer0.params
grads = T.grad(cost, params)
updates = [(param_i, param_i - lr * grad_i) for param_i, grad_i in zip(params, grads)]

train_model = theano.function([index], [cost, acc], updates=updates,
        givens={
        	x: train_x[index*batch_size:(index+1)*batch_size],
        	y: train_y[index*batch_size:(index+1)*batch_size]
        })

test_model = theano.function([index], [cost,acc],
        givens={
        	x: test_x[index*batch_size:(index+1)*batch_size],
        	y: test_y[index*batch_size:(index+1)*batch_size]
        })

write_graph(test_model, output_name='theano_cnn', output_format='png')

for epoch in range(epochs):
    for batch_idx in range(n_train_batches):
        ite = epoch * n_train_batches + batch_idx
        train_loss, train_acc = train_model(batch_idx)
        if ite % 50 == 0:
             test_res = np.array([test_model(i) for i in range(n_test_batches)])
             this_test_loss = np.mean(test_res[:,0])
             this_test_acc = np.mean(test_res[:,1])
             print 'epoch %d iteration %d train loss %f train acc %f test loss %f test acc %f' % \
                 (epoch+1, ite, train_loss, train_acc, this_test_loss, this_test_acc)
