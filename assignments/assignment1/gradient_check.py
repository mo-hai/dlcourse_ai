import numpy as np
from typing import Union


def check_gradient(f, x: np.array, delta: float = 1e-5, tol: float = 1e-4):
    '''
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:
      bool indicating whether gradients match or not
    '''
    
    assert isinstance(x, np.ndarray)
    assert x.dtype == float
    
    orig_x = x.copy()
    fx, analytic_grad = f(x)
    assert np.all(np.isclose(orig_x, x, tol)), "Functions shouldn't modify input variables"

    assert analytic_grad.shape == x.shape
    analytic_grad = analytic_grad.copy()

    numeric_grad = np.zeros_like(x)  # Initialize gradient matrix
    assert numeric_grad.shape == x.shape

    # We will go through every dimension of x and compute numeric derivative for it
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index

        # sample = x[idx[0]].copy()
        tmp_val = x[idx]
        # Calculate f(x + delta)
        x[idx] = tmp_val + delta
        fx1 = f(x)[0]
        # Calculate f(x - delta)
        x[idx] = tmp_val - delta
        fx2 = f(x)[0]
        # Restore the original value of x[idx]
        x[idx] = tmp_val

        # Calculate the gradient for element at index idx
        numeric_grad[idx] = (fx1 - fx2) / (2 * delta)

        numeric_grad_at_idx = numeric_grad[idx]
        analytic_grad_at_idx = analytic_grad[idx]

        # TODO compute value of numeric gradient of f to idx
        if not np.isclose(numeric_grad_at_idx, analytic_grad_at_idx, tol):
            print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (idx, analytic_grad_at_idx, numeric_grad_at_idx))
            return False

        it.iternext()

    print("Gradient check passed!")
    return True
