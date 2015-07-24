import theano
from theano import tensor as T
import numpy as np

# training data
trX = np.linspace(-1,1,101)
trY = trX * np.random.randn(*trX.shape) * 0.33

X = T.scalar()
Y = T.scalar()

def model(X, w):
    return X * w

# initization
w = theano.shared(np.asarray(0., dtype=theano.config.floatX))
y = model(X,w)

cost = T.mean(T.sqr(y - Y))
gradient = T.grad(cost=cost, wrt=w)
updates = [[w, w - gradient * 0.01]]

train = theano.function(inputs=[X,Y], outputs=cost, updates=updates, allow_input_downcast=True)

for i in range(100):
    for x, y in zip(trX,trY):
        train1 = train(x,y)
        print train1
