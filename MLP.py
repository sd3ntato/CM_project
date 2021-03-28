# general imports
from utils import myflatten, armijo_wolfe
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from IPython.display import clear_output

# activation functions
from numpy import tanh
ide = lambda x : np.copy(x)
relu = lambda x: x*(x > 0)
from scipy.special import softmax

# loss functions:
squared_error = lambda y,d:  np.linalg.norm( (y - d).flatten() ) ** 2 
cross_entropy = lambda y,d: -np.sum( d * np.log( y + np.finfo(float).eps ) ) 
MSE = lambda y,d: np.mean( np.square( y-d ) )

# norm-regularization (rewritten a naive version of numpy's l1-norm in order to give it an array of matrices of (possibly) different sizes and get back l1 norm of each matrix ) 
#l1 = lambda x: np.array( [ np.max(np.sum(np.abs(w), axis=0)) for w in x ] )
l1 = lambda x: np.linalg.norm( myflatten(x).reshape(-1), ord=1)

def derivative(f):
  """
  When f is an activation function, returns derivative w.r.t. potential of activation
  When f is a loss, returns derivative w.r.t. activation of last layer's units
  When f is norm, returns derivative w.r.t. weigts
  (When f is cross_entropy and activation of output units is softmax, maths say derivative of loss w.r.t potential is one )
  """
  if f == tanh:
    return lambda x: 1.0 - tanh(x)**2
  elif f == relu:
    return lambda x: 1*(x>=0)
  elif f == ide or f == softmax:
    return lambda x : x-x+1
  elif f == squared_error or f==MSE or f == cross_entropy:
    return lambda d,y: y-d 
  elif f==l1:
    return lambda x: np.array( [ np.sign(w) for w in x ] )

def get_f(f):
  """
  activation function: string->function
  """ 
  if f=='relu':
    return relu
  elif f=='tanh':
    return tanh

def get_f_out(f):
  """
  activation function of output units: string->function
  """ 
  if f=='ide':
    return ide
  elif f=='softmax':
    return softmax

def get_loss(f):
  """
  loss function: string->function
  """ 
  if f=='squared_error':
    return squared_error
  elif f=='MSE':
    return MSE
  elif f=='cross_entropy':
    return cross_entropy

def get_regularization(r):
  if r=='l1':
    return l1

def get_error(f):
  """
  error function: string->function
  """ 
  if f=='MSE':
    return MSE
  elif f=='cross_entropy':
    return cross_entropy

class MLP():

  def __init__(self, Nh=[10], Nu=1, Ny=1, f='tanh', f_out='ide' , w_range=.7, w_scale=4, loss='MSE', regularization='l1', error='MSE'):
    """
    Nh: number of hidden units for each layer. it is supposed to be an array [Nh_1, Nh_2, ... , Nh_l] s.t. each element Nh_i is the number of hidden units in layer i
    Nu: number of input units
    Ny: number of output units
    f: activation function of hidden units
    f_out: activation function of output units
    w_range: initial range of values for entries in weight matrices
    w_sclae: initial number of decimals of values for entries in weight matrices
    loss: loss function: l(y,d) 
    regularization: type of norm regularization: a penality term is added to loss function O(w)
    error: error function for assessment of training/validation/test error
    """ 
      
    if loss == 'cross_entropy':
      assert f_out == 'softmax', 'if using cross-entropy loss, must use softmax as output activation function'

    f = get_f(f)
    f_out = get_f_out(f_out)
    loss = get_loss(loss)
    omega = get_regularization(regularization)
    error = get_error(error)

    Nl = len(Nh)
    self.Nl = Nl # number of hidden layers
    self.Nu = Nu # number of input units
    self.Ny = Ny # number of output units
    self.Nh = Nh # array [Nh_1, Nh_2, ... , Nh_l] s.t. each element Nh_i is the number of hidden units in layer i

    self.f = [ ide ] + ( [f] * Nl ) + [ f_out ] # [f_in, f, f, ..., f, f, f_out] f[m](a[m]) array of activation functions, f[i] is activation function at layer i
    self.df = [ derivative(f) for f in self.f] # df[m](v[m]) array of derivative of activation functions w.r.t. their input
    self.w = np.array( [None]*(Nl+1), dtype=object ) # weight matrices, w[i] contains weights of connections between units at layer i and units at layer i+1

    self.l = loss # loss function,
    self.dl = derivative(loss) # its derivative w.r.t. activation of last layer's units
    self.o = omega # penality term of loss function given by selected norm of weights
    self.do = derivative(omega) # derivative of penality term for regularization
    self.error = error # error function, to asses test/training/validation error

    # a[m+1] = f[m]( w[m] @ a[m] ) a[m] = (Nh,1) a[m+1] = (Nh,1) w[m] = (Nh,Nh)
    self.w[0] = np.round( ( 2*np.random.rand( Nh[0], Nu+1 ) -1 )*w_range, w_scale )# pesi input-to-primo-layer, ultima colonna e' bias. w[i,j] in [-1,1]
    for i in range(1, Nl):
      self.w[i] = np.round( ( 2*np.random.rand( Nh[i], Nh[i-1] + 1 )-1 )*w_range, w_scale )# pesi layer-to-layer, ultima colonna e' bias
    self.w[Nl] = np.round( ( 2*np.random.rand( Ny, Nh[Nl-1] + 1) -1 )*w_range, w_scale )# pesi ultimo-layer-to-output, ultima colonna e' bias

  def forward_pass(self, u:np.ndarray ): 
    """
    compute activations and activation potentials
    """
    Nl = self.Nl
    v = [None]*(Nl+2) # potenziali attivazione v[m]
    a = [None]*(Nl+2) # activations a[m] = f[m](v[m])

    # reshape input if needed
    if not u.shape == (self.Nu,1): 
      u = u.reshape((self.Nu,1))

    # compute activation and potentials for units in each layer
    v[0] = u
    a[0] = u # activation of input units is external input
    for m in range(self.Nl+1): 
      v[m+1] = self.w[m] @ np.vstack((a[m],1)) # To simulate bias activation, i add to activations a 1 at the bottom
      a[m+1] = self.f[m+1](v[m+1])
    return a,v

  def backward_pass(self, y, a, v): 
    """
    given activations and potentials compute error-propagation-coefficents
    """
    Nl=self.Nl

    d = [None]*(self.Nl+2) # error-propagation-coefficents d[m]

    # reshape desired-output if needed
    if not y.shape == (self.Ny,1):
      y = y.reshape((self.Ny,1))

    # compute error-propagation-coefficents for units in each layer
    d[Nl+1] = self.dl( y , a[Nl+1]) * self.df[Nl+1](v[Nl+1]) # error-propagation-coefficents of output units
    for m in range(Nl,-1,-1):
      d[m] = ( np.delete( self.w[m].T, -1, 0) @ d[m+1] ) * self.df[m](v[m])  # must get row (column) of bias weights out of the computation of propagation coefficents

    return d

  def compute_gradient_p(self,p): 
    """
    compute gradient of loss w.r.t activations, over pattern p
    """
    Nl = self.Nl

    # pattern is composed of input and relative desired output
    x,y = p

    # compute activations and potentials
    a, v = self.forward_pass( x ) 

    # compute error-propagation-coefficents
    d = self.backward_pass( y, a, v ) 

    # compute gradient for each layer. To simulate bias activation, i add to activations a 1 at the bottom
    grad = [ ( d[m+1] @ np.vstack( ( a[m], 1 ) ).T ) for m in range(Nl+1) ]

    return np.array(grad)

  def compute_gradient(self, train_x, train_y, epsilon):
    N = np.size(train_x,axis=0) 
    compute_partial_gradient = self.compute_gradient_p # function for computing paryial gradient on pattern on pattern p
    return sum( map( compute_partial_gradient, zip( train_x,train_y ) ) )/N + epsilon * self.do(self.w)  # function for computing gradient of loss function over all patterns


  def momentum_train(self, train_x:np.ndarray, train_y:np.ndarray, alpha=1e-01, beta=5e-02, epsilon=0, tresh=1e-02, max_epochs=300,reset=True ):
    """
      trains the network using classical momentum
      alpha: learning rate
      beta: acceleration coefficent 
      epsilon: regularization coefficent
      tresh: treshold to exit main loop of training
      max_epochs: maximum number of epochs to be done
    """
    # stuff for statistics computation
    grad_norms = []
    errors = []
    def statistics(gradient_norm, X, Y):
      grad_norms.append( gradient_norm )
      e = self.test_loss(X,Y,epsilon)
      errors.append( e )
      print(gradient_norm, e )
      clear_output(wait=True)

    # number of patterns in training set, epochs of training currently executed
    epoch = 0

    # functions for gradient computation
    compute_gradient = self.compute_gradient

    # prevous velocity (v_t) for momentum computation. (placeholder)
    old_d = np.array( [ np.zeros(self.w[i].shape) for i in range(self.Nl+1) ] ,dtype=object) 

    # main loop: compute velocity and update weights
    gradient_norm = np.inf # placeholder for gradient norm
    init_gradient_norm = np.linalg.norm( myflatten( compute_gradient( train_x,train_y, epsilon ) ) ) # initial norm of the gradient, to be used for stoppin criterion
    while( ( gradient_norm / init_gradient_norm ) > tresh and epoch < max_epochs ):

      if reset and epoch % 50 == 0:
        print('reset')
        old_d = np.array( [ np.zeros(self.w[i].shape) for i in range(self.Nl+1) ] ,dtype=object) 

      # compute gradient ( \Nabla loss(w_t) ) and its norm
      g = compute_gradient( train_x, train_y, epsilon )
      gradient_norm = np.linalg.norm( myflatten(g) ) # generally gradient is a tensor/matrix. To compute its norm i flatten it in order to make it become a vector

      #compute velocity d
      d = -alpha*g + beta*old_d

      # update weights and prevoius velocity
      self.w = self.w + d
      old_d = d

      # update epochs counter and collect statistics
      epoch +=1; statistics(gradient_norm / init_gradient_norm,train_x,train_y)
    
    return grad_norms, errors
    

  def supply(self, u):
    """
    Supply an input to this network. The network computes its internal state and otuput of the network is activation of the last layer's units.
    u: input pattern
    returns output of the network given the supplied pattern
    """
    a = [None]*(self.Nl+2) # attivazioni a[m] = f[m](v[m])

    # reshape input if needed
    if not u.shape == (self.Nu,1):
      u = u.reshape((self.Nu,1))
    
    # calculate activation of units in each layer
    a[0] = u
    for m in range(self.Nl+1):
      a[m+1] = self.f[m+1](  self.w[m] @ np.vstack(( a[m], 1)) )
    return np.copy(a[self.Nl+1])

  def supply_sequence(self,U):
    """
    given sequence of input patterns, computes sequence of relative network's outputs.
    complied version of 
      return [float(self.predict(u)) for u in tx]
    U: sequence of input patterns.
    """
    # calculate sequence of outputs of the network when provided when given sequence of inputs
    sup = self.supply
    return np.array(list( map( lambda u : sup(u) , U ) ))
  
  def test_error(self, X, Y):
    outs = self.supply_sequence(X)
    return self.error(outs, Y.reshape(outs.shape))
  
  def test_loss(self, X, Y, epsilon):
    outs = self.supply_sequence(X)
    return self.l(outs, Y.reshape(outs.shape)) + epsilon * self.o(self.w)

