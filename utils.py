import numpy as np

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

myflatten = lambda g: np.hstack( [ g[i].flatten() for i in range(len(g)) ] )

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