import numpy as np
import cvxpy as cp

def to_categorical(y, num_classes=None, dtype='float32'): # code from keras implementation: keras.utils.to_categorical
  """Converts a class vector (integers) to binary class matrix.
  E.g. for use with categorical_crossentropy.
  Args:
      y: class vector to be converted into a matrix
          (integers from 0 to num_classes).
      num_classes: total number of classes. If `None`, this would be inferred
        as the (largest number in `y`) + 1.
      dtype: The data type expected by the input. Default: `'float32'`.
  Returns:
      A binary matrix representation of the input. The classes axis is placed
      last.
  """
  y = np.array(y, dtype='int')
  input_shape = y.shape
  if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
    input_shape = tuple(input_shape[:-1])
  y = y.ravel()
  if not num_classes:
    num_classes = np.max(y) + 1
  n = y.shape[0]
  categorical = np.zeros((n, num_classes), dtype=dtype)
  categorical[np.arange(n), y] = 1
  output_shape = input_shape + (num_classes,)
  categorical = np.reshape(categorical, output_shape)
  return categorical

def myflatten(g):
  return np.hstack( [ g[i].flatten() for i in range(len(g)) ] )

def deflatten(n,x):
  """
    given weight in flat form, returns weights in "tensor" form
  """
  w = np.array( [None]*(n.Nl+1), dtype=object )

  w[0] = x[ : n.Nh[0] * (n.Nu+1) ].reshape( n.Nh[0], n.Nu+1 )
  x = x[n.Nh[0]*(n.Nu+1):]
  for i in range(1, n.Nl):
      w[i] = x[ : n.Nh[i] * (n.Nh[i-1] + 1) ].reshape( n.Nh[i], n.Nh[i-1] + 1)
      x = x[n.Nh[i] * (n.Nh[i-1] + 1):]
  w[n.Nl] = x.reshape(n.Ny, n.Nh[n.Nl-1] + 1)

  # print( myflatten(n.w) - myflatten( deflatten( n, myflatten( n.w ) ) ) )

  return w

def phi(alpha, n, d, train_x, train_y, epsilon):
  """
    phi(alpha) = f( x + alpha * d ) = loss( n_{w+alpha*d} )
  """
  # copy the current weights of the netwrork
  w = np.copy(n.w)

  # use a network with temporarily updated weights
  n.w += alpha * d

  # compute loss of the modified network, a.k.a phi(alpha)
  outs = n.supply_sequence(train_x).reshape(train_y.shape)
  phi_alpha = n.l(outs, train_y) + np.linalg.norm(myflatten(n.w), ord=1)# f of x plus alpha d

  # compute derivative/gradient of loss of the modified net
  g = n.compute_gradient( train_x, train_y ) + epsilon * n.do(n.w) 

  # compute actual value of derivative of phi(alpha)
  phi_prime_alpha = np.dot( myflatten(g), myflatten(d) )

  # reset weights
  n.w = w

  return phi_alpha, phi_prime_alpha

def armijo_wolfe(n, alpha, d, train_x, train_y, epsilon, m1, m2, tau):
  """
   In realta' questa e' una versione becera del backtracking, se voglio tenerla cosi dovrei tenere solo la condizione armijo col maggiore stretto
  """
  phi_zero, phi_prime_zero = phi(0, n, d, train_x, train_y, epsilon)
  phi_alpha, phi_prime_alpha = phi(alpha, n, d, train_x, train_y, epsilon)

  n_iter=0
  while not ( phi_alpha <= phi_zero + m1 * alpha * phi_prime_zero and abs(phi_prime_alpha) <= - m2 * phi_prime_zero ) and n_iter<10:
    alpha *= tau # decrease tau
    phi_alpha, phi_prime_alpha = phi(alpha, n, d, train_x, train_y, epsilon)
    n_iter+=1
  
  print( 'alpha found at itration ',n_iter,' , ',alpha)
  return alpha

def proximal_bundle_method(n, mu, epsilon, m1, reg_param, train_x, train_y):
  from numpy.linalg import norm
  N = n.numero_parametri # numero parametri 

  def f(x): 
    """
    this computes loss and its derivative/gradient given weights
    """

    # copy the current weights of the netwrork
    w = np.copy(n.w)

    # use a network with temporarily updated weights
    n.w = deflatten(n,x) # questo x arriva piatto e va rimesso in forma di tensore

    # compute loss of the modified network
    outs = n.supply_sequence(train_x).reshape(train_y.shape)
    f_x = n.l(outs, train_y) + np.linalg.norm(myflatten(n.w), ord=1)# f of x plus alpha d

    # compute derivative/gradient of loss of the modified net
    g_x = n.compute_gradient( train_x, train_y ) + reg_param * n.do(n.w) 

    # reset weights
    n.w = w

    return f_x, g_x

  x_bar = np.copy(n.w)
  f_x_bar, g_x_bar = f(x_bar)
  bundle = [ ( x_bar, f_x_bar, g_x_bar ) ]

  def solve_quadratic():
    v = cp.Variable(1,1)
    x = cp.Variable(N,1)
    objective = cp.Minimize( v + mu * norm( x - x_bar )**2  )
    constraints = [ v >= f_i + g_i.T @ ( x - x_i ) for f_i, g_i, x_i in bundle  ]
    prob = cp.Problem(objective, constraints)
    _ = prob.solve()
    return x.value, v.value

  while(True):
    x_star, f_bundle_x_star = solve_quadratic()
    if mu * norm(x_star-x_bar) <= epsilon:
      break
    
    f_x_star, g_x_star = f(x_star)
    if f_x_star <= f_x_bar + m1*( f_bundle_x_star - f_x_bar ):
      x_bar = x_star
      f_x_bar, g_x_bar = f(x_bar)
      
    bundle.append( x_star, f_x_star, g_x_star )
