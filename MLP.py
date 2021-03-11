#http://machinelearningmechanic.com/deep_learning/2019/09/04/cross-entropy-loss-derivative.html
#https://deepnotes.io/softmax-crossentropy

# general imports
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from IPython.display import clear_output

# activation functions
from numpy import tanh
from scipy.special import softmax
ide = lambda x : np.copy(x)
relu = lambda x: x*(x > 0)
#softmax = lambda x: np.exp(x - logsumexp(x, keepdims=True)) # implementazione scipy special

# loss functions:
squared_error = lambda y,d:  np.linalg.norm(y - d) ** 2 # categorical cross-entropy
cross_entropy = lambda y,d: -np.sum( d * np.log( y + np.finfo(float).eps ) )
MSE = lambda x,y: np.mean( np.square( x-y ) )

def derivative(f):
  """
  When f is an activation function, returns derivative w.r.t. potential of activation
  When f is a loss, returns derivative w.r.t. activation
  When f is cross_entropy and activation of output units is softmax, maths say derivative of loss w.r.t potential is one returned
  """
  if f == tanh:
    return lambda x: 1.0 - tanh(x)**2
  elif f == relu:
    return lambda x: 1*(x>=0)
  elif f == ide or f == softmax:
    return lambda x : x-x+1
  elif f == squared_error or f == cross_entropy:
    return lambda d,y: y-d 

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
  elif f=='cross_entropy':
    return cross_entropy

def get_error(f):
  """
  error function: string->function
  """ 
  if f=='MSE':
    return MSE
  elif f=='cross_entropy':
    return cross_entropy

class MLP():

  def __init__(self, Nh=[10], Nu=1, Ny=1, f='tanh', f_out='ide' , w_range=.7, w_scale=2, loss='squared_error', error='MSE'):
    """
    Nh: number of hidden units for each layer
    Nu: number of input units
    Ny: number of output units
    f: activation function of hidden units
    f_out: activation function of output units
    w_range: initial range of values for entries in weight matrices
    w_range: initial number of decimals of values for entries in weight matrices
    loss: loss functions
    error: error function
    """ 
      
    if loss == 'cross_entropy':
      assert f_out == 'softmax', 'if using cross-entropy loss, must use softmax as output activation function'

    f = get_f(f)
    f_out = get_f_out(f_out)
    loss = get_loss(loss)
    error = get_error(error)

    Nl = len(Nh)
    self.Nl = Nl # numero layer
    self.Nu = Nu # unita' input
    self.Ny = Ny # unita' output
    self.Nh = Nh # unita' interne

    self.f = [ ide ] + ( [f] * Nl ) + [ f_out ] #[f_in, f,f,f,f ,f_out] f[m](a[m])
    self.df = [ derivative(f) for f in self.f] # df[m](v[m])
    self.w = np.array( [None]*(Nl+1), dtype=object ) # matrici dei pesi 

    self.l = loss # funzione loss (y-d)**2
    self.dl = derivative(loss) # (y-d)
    self.error = error

    # a[m+1] = f[m]( w[m]*a[m] ) a[m] = (Nh,1) a[m+1] = (Nh,1) w[m] = (Nh,Nh)
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
    a = [None]*(Nl+2) # attivazioni a[m] = f[m](v[m])

    # reshape input if needed
    if not u.shape == (self.Nu,1): 
      u = u.reshape((self.Nu,1))

    # compute activation and potentials for units in each layer
    v[0] = u
    a[0] = u # activation of input units is external input
    for m in range(self.Nl+1): 
      v[m+1] =  np.dot( self.w[m] , np.vstack((a[m],1)) ) # activation of bias units is always 1
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

    # calculate error-propagation-coefficents for units in each layer
    d[Nl+1] = self.dl( y , a[Nl+1]) * self.df[Nl+1](v[Nl+1]) # error-propagation-coefficents of output units
    for m in range(Nl,-1,-1):
      d[m] =  np.dot(  np.delete( self.w[m].T , -1, 0)  , d[m+1]  ) * self.df[m](v[m])  # must get row (column) of bias weights out of the computation of propagation coefficents

    return d

  def compute_gradient_p(self,p): 
    """
    compute gradient over pattern p
    """
    Nl = self.Nl

    # pattern is composed of input and relative desired output
    x,y = p

    # compute activations and potentials
    a, v = self.forward_pass( x ) 

    # compute error-propagation-coefficents
    d = self.backward_pass( y, a, v ) 

    #compute gradient for each layer. To siumlate bias activation, i add to activations a 1 at the bottom
    grad = [ np.dot( d[m+1] , np.vstack( ( a[m], 1 ) ).T ) for m in range(Nl+1) ]

    return np.array(grad)

  def momentum_train(self, train_x:np.ndarray, train_y:np.ndarray, epsilon, mu, tresh=1e-02, max_epochs=300 ):
    """
      trains the network using classical momentum
    """
    # stuff for statistics computation
    grad_norms = []
    errors = []
    def statistics(gradient_norm, X, Y):
      grad_norms.append( gradient_norm )
      e = self.test(X,Y)
      errors.append( e )
      print(gradient_norm, e )
      clear_output(wait=True)

    # number of patterns in training set, epochs of training currently executed
    N = np.size(train_x,axis=0) 
    epoch = 0

    # functions for gradient computation
    compute_partial_gradient = self.compute_gradient_p # function for computing paryial gradient on pattern on pattern p
    compute_gradient = lambda train_x, train_y: sum( map( compute_partial_gradient, zip( train_x,train_y ) ) )/N # function for computing gradient of loss function over all patterns

    # prevous velocity (v_t) for momentum computation
    old_v = np.array( [ np.zeros(self.w[i].shape) for i in range(self.Nl+1) ] ,dtype=object) 

    # main loop: compute velocity and update weights
    gradient_norm = np.inf
    while( gradient_norm > tresh and epoch < max_epochs ):

      # compute gradient ( \Nabla l(w_t) ) and its "norm"
      g = compute_gradient( train_x,train_y ); gradient_norm = np.linalg.norm( np.hstack( [ g[i].flatten() for i in range(len(g)) ] ) )

      #compute velocity v_{t+1}
      v = mu * old_v - epsilon * g

      # update weights and prevoius velocity
      self. w = self.w + v
      old_v = v

      # update epochs counter and collect statistics
      epoch +=1; statistics(gradient_norm,train_x,train_y)
    
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
      a[m+1] = self.f[m+1]( np.dot( self.w[m] , np.vstack((a[m],1)) ) )
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
  
  def test(self, X, Y):
    outs = self.supply_sequence(X).flatten()
    return self.error(outs, Y.flatten())

