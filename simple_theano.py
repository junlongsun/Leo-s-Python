import theano
from theano import tensor as T

# viriaable initization
a = T.scalar()
b = T.scalar()

y = a * b

multiply = theano.function(inputs=[a,b],outputs = y)

print multiply(1,2)
print multiply(3,3)