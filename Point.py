'''
	Author: Mohit Sharma
	email: sharm267@purdue.edu
'''

import numpy as np
import copy

'''
	For Homogeneous Linear Feasibility model (HFM) a point represents a tuple (x,tau,y,s,kappa)

	If initial point is not specified, assign default (e,1,0,e,1)

	Point Class checks the strict feasibility of all the points
'''
class Point(object):

	def __init__(self,n,m,x=None,tau=None,y=None,s=None,kappa=None):
		
		assert( n >=1 )
		assert( m >=1 )

		self._n = n
		self._m = m
		self._x = x if x is not None else np.ones((n,1))
		self._tau = tau if tau is not None else 1
		self._y = y if y is not None else np.zeros((m,1))
		self._s = s if s is not None else np.ones((n,1))
		self._kappa = kappa if kappa is not None else 1

	@property 
	def x(self):
		return self._x
	
	@property 
	def tau(self):
		return self._tau

	@property 
	def y(self):
		return self._y

	@property 
	def s(self):
		return self._s

	@property
	def kappa(self):
		return self._kappa

	@x.setter
	def x(self,val):
		assert(np.all(val > 0))
		self._x = val

	@tau.setter
	def tau(self,val):
		assert(val > 0)
		self._tau = val

	@y.setter
	def y(self,val):
		self._y = val

	@s.setter
	def s(self,val):
		assert(np.all(val>0))
		self._s = val

	@kappa.setter
	def kappa(self,val):
		assert(val > 0)
		self._kappa = val
	
	def copy(self):
		return copy.deepcopy(self)

	'''
		Increment the current point in the direction of step 
		by the amount specified in step_size
	'''
	def update(self,curr_step,curr_step_size):

		if curr_step_size.same_step:
			alpha_p = np.min([curr_step_size.alpha_p,curr_step_size.alpha_d])
			alpha_d = alpha_p
			alpha_t = alpha_p
			alpha_k = alpha_p
		else:
			alpha_p = curr_step_size.alpha_p
			alpha_d = curr_step_size.alpha_d

			tau_p = self.tau + alpha_p*curr_step.dtau
			tau_d = self.tau + alpha_d*curr_step.dtau

			if tau_p < tau_d:
				alpha_t = alpha_p
				alpha_k = alpha_d
			else:
				alpha_t = alpha_d
				alpha_k = alpha_p

		tau_p = self.tau + curr_step_size.alpha_p*curr_step.dtau
		tau_d = self.tau + curr_step_size.alpha_d*curr_step.dtau

		self.x = self.x + alpha_p*curr_step.dx
		self.y = self.y + alpha_d*curr_step.dy
		self.tau = self.tau + alpha_t*curr_step.dtau
		self.s = self.s + alpha_d*curr_step.ds
		self.kappa = self.kappa + alpha_k*curr_step.dkappa

		# self.x = self.x + curr_step_size.alpha_p*curr_step.dx
		# self.y = self.y + curr_step_size.alpha_d*curr_step.dy
		# self.tau = self.tau + curr_step_size.alpha_t*curr_step.dtau
		# self.s = self.s + curr_step_size.alpha_d*curr_step.ds
		# self.kappa = self.kappa + curr_step_size.alpha_k*curr_step.dkappa


		# import pdb; pdb.set_trace()

		# Further Update for x, y and z to get solution to original LP, not the Homogeneous one
		self.x = np.true_divide(self.tau,tau_p)*self.x
		self.y = np.true_divide(self.tau,tau_d)*self.y
		self.s = np.true_divide(self.tau,tau_d)*self.s