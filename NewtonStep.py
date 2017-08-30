'''
	Author: Mohit Sharma
	Email: sharm267@purdue.edu
'''

import numpy as np
import numpy.linalg as linalg
import scipy.sparse as sparse
'''
	For a system of non-linear equations, return the newton step.
'''
class NewtonStep:

	def __init__(self,LP,curr_pt,eta,gamma):
		self.LP = LP
		self.curr_pt = curr_pt

		self._x = None
		self._y = None
		self._s = None
		self._tau = None
		self._kappa = None


		# Add checks for the values of eta and gamma
		self.eta = eta
		self.gamma = gamma
	
		# This functions calculates all the steps and assign them to class variables
		self.compute()

	@property
	def x(self):
		return self._x

	@property 
	def y(self):
		return self._y

	@property
	def s(self):
		return self._s

	@property
	def tau(self):
		return self._tau

	@property
	def kappa(self):
		return self._kappa

	'''
		Building a dense K matrix and converting back to csc_matrix format in the end. 

		TODO:
		Find a consistent way to construct sparse matrix out of sparse block matrices.
	'''	

	def compute_step(self):
		
		X = np.diag(self.curr_pt.x.transpose()[0])
		S = np.diag(self.curr_pt.s.transpose()[0])


		'''
			Construct the matrix K 

			K = [ -X^(-1)S 	A^T ; A 	0]
		'''

		X_inv  = linalg.inv(X)
		
		row1 = np.bmat([-1*(linalg.inv(X))*S,self.LP.A.toarray().transpose()])
		row2 = np.bmat([self.LP.A.toarray(),np.zeros((self.LP.A.toarray().shape[0],self.LP.A.toarray().shape[0]))])
 	 
		K = sparse.csc_matrix(np.bmat([[row1],[row2]]))
		
		# Assert that K is non-singular
		rank = linalg.matrix_rank(K)
		assert( rank == K.shape[0])


		return K
