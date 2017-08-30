'''
	Author: Mohit Sharma
	email: sharm267@purdue.edu
'''

import numpy as np

'''
	For Homogeneous Linear Feasibility model (HFM) a point represents a tuple (x,tau,y,s,kappa)

	If initial point is not specified, assign default (e,1,0,e,1)
'''
class Point:

	def __init__(self,n,m,x=None,tau=None,y=None,s=None,kappa=None):
		
		assert( n >=1 )
		assert( m >=1 )

		self.x = x if x is not None else np.ones((n,1))
		self.tau = tau if tau is not None else 1
		self.y = y if y is not None else np.zeros((m,1))
		self.s = s if s is not None else np.ones((n,1))
		self.kappa = kappa if kappa is not None else 1

		assert(self.x.shape == (n,1))
		assert(self.y.shape == (m,1))
		assert(self.s.shape == (n,1))
		
		assert(self.tau > 0)
		assert(self.kappa >0)

		assert(np.all(self.x > 0))
		assert(np.all(self.s>0))