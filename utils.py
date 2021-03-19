import numpy as np
import cvxpy as cp
from IPython.display import clear_output

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
  return np.hstack( [ g[i].flatten() for i in range(len(g)) ] ).reshape(-1,1)

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
  phi_alpha = n.test_loss(train_x,train_y) + epsilon * np.linalg.norm( myflatten(n.w).reshape(-1), ord=1 )

  # compute derivative/gradient of loss of the modified net
  g = n.compute_gradient( train_x, train_y ) + epsilon * n.do(n.w) 

  # compute actual value of derivative of phi(alpha)
  phi_prime_alpha =  myflatten(g).T @ myflatten(d) 

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

def proximal_bundle_method(n, train_x, train_y, reg_param=1e-04, m1 = 5e-02, epsilon=1e-03, mu=.1, max_epochs=100):
  from numpy.linalg import norm

  def f(x): 
    """
    this computes loss and its derivative/gradient given weights
    """

    # copy the current weights of the netwrork
    w = np.copy(n.w)

    # use a network with temporarily updated weights
    n.w = deflatten(n,x) # questo x arriva piatto e va rimesso in forma di tensore

    # compute loss of the modified network
    f_x = n.test_loss(train_x,train_y) + reg_param * np.linalg.norm( myflatten(n.w).reshape(-1), ord=1 )

    # compute derivative/gradient of loss of the modified net
    g_x = myflatten( n.compute_gradient( train_x, train_y ) + reg_param * n.do(n.w) )

    # reset weights
    n.w = w

    return f_x, g_x

  # function and data structure for statistics computation
  grad_norms = []
  errors = []
  def statistics(gradient_norm, X, Y):
    grad_norms.append( gradient_norm )
    e = n.test_loss(X,Y)
    errors.append( e )
    print(gradient_norm, e, mu )
    clear_output(wait=True)

  # initial point
  x_bar = myflatten( np.copy(n.w) )
  N = len(x_bar) # numero parametri 

  # value of function and its gradient on initial point
  f_x_bar, g_x_bar = f(x_bar)

  # initialize bundle
  bundle = [ ( x_bar, f_x_bar, g_x_bar ) ]

  # function for solving master problem
  def solve_master(x_bar,bundle):
    v = cp.Variable((1,1))
    x = cp.Variable((N,1))
    objective = cp.Minimize( v + mu * 0.5 * cp.atoms.norm( (x-x_bar).flatten() )**2  )
    constraints = [ v >= f_i + g_i.T @ ( x - x_i ) for x_i, f_i, g_i in bundle  ]
    problem = cp.Problem(objective, constraints)
    try:
      problem.solve()
    except:
      print('solver failed')
      return None, None
    print(problem.status)
    return x.value, v.value

  # main loop
  n_epoch = 0
  while(True):
    # compute current x_star, optimal value of the bundle function
    x_star, f_bundle_x_star = solve_master(x_bar,bundle)

    #if not isinstance(x_star,np.ndarray):
    if x_star is None:
      print('unable to solve master problem')
      return x_star, grad_norms, errors

    # if i get a solution close to the one i had, i can quit 
    if mu * np.linalg.norm(x_star-x_bar) <= epsilon or n_epoch > max_epochs:
      print( mu * np.linalg.norm(x_star-x_bar) )
      break
    
    # Null step/Serious step decision: i compute f on current x_star and see if it is better than my current x_bar
    f_x_star, g_x_star = f(x_star)
    if f_x_star <= f_x_bar + m1 * ( f_bundle_x_star - f_x_bar ):
      # if step is serious i change my current x_bar and recompute function value and its gradient
      x_bar = x_star
      f_x_bar, g_x_bar = f(x_bar)

      mu = mu * 0.7

      n.w = deflatten(n,x_bar)

      print('SS')
      statistics(np.linalg.norm(g_x_bar), train_x, train_y)
    else:
      print('NS')
      statistics(np.linalg.norm(g_x_bar), train_x, train_y)
      mu = mu * 1.3

    bundle.append( (x_star, f_x_star, g_x_star) )
    n_epoch += 1
  
  return deflatten(n,x_bar), grad_norms, errors
