'''
	Author: Mohit Sharma
	Email: sharm267@purdue.edu
'''

import numpy as np
import numpy.linalg as linalg
import scipy.sparse as sparse
from SymmetricSystemSolver import SymmetricSystemSolver
from Step import Step

'''
	For a system of non-linear equations, return the newton step.
'''
class NewtonStep:

	def __init__(self,LP,curr_pt,eta,gamma,residual_step=None):

		self.step = Step()


		self.LP = LP
		self.curr_pt = curr_pt

		# Add checks for the values of eta and gamma
		self.eta = eta
		self.gamma = gamma



		# Residuals at current point
		self.r_p = self.curr_pt.tau*self.LP.b - self.LP.A.dot(self.curr_pt.x)
		self.r_d = self.curr_pt.tau*self.LP.c - self.LP.A.transpose().dot(self.curr_pt.y) - self.curr_pt.s


		self.r_g = self.curr_pt.kappa + np.dot(self.LP.c.transpose(),self.curr_pt.x) - np.dot(self.LP.b.transpose(),self.curr_pt.y)

		# complementarity-gap at current point
		self.mu = (np.dot(self.curr_pt.x.transpose(),self.curr_pt.s) + self.curr_pt.kappa*self.curr_pt.tau)/(self.LP.shape[1] + 1)
	
		self.r_xs = np.dot(-1*np.diag(self.curr_pt.x.transpose()[0]),self.curr_pt.s) + self.gamma*self.mu*np.ones((self.LP.shape[1],1))
		self.r_tk = -1*self.curr_pt.tau*self.curr_pt.kappa + self.gamma*self.mu

		# Add the residual term
		self.r_xs = self.r_xs + np.dot(np.diag(residual_step.dx.transpose()[0]),residual_step.ds) if residual_step is not None else self.r_xs
		self.r_tk = self.r_tk + residual_step.dtau* residual_step.dkappa if residual_step is not None else self.r_tk 

		# This functions calculates all the steps and assign them to class variables
		self.compute_step()

	'''

		TODO:
		Find a consistent way to construct sparse matrices directly
	'''	

	def compute_step(self):
		
		X = sparse.diags([self.curr_pt.x.transpose()[0]],[0])
		S = sparse.diags([self.curr_pt.s.transpose()[0]],[0])

		X_inv  = sparse.linalg.inv(X)
	

		'''
			Solve K (p;q) = (c;b)
			
			where  K = [[K1 K2],[K3 0]]

		'''
		K1 = (-1*X_inv).dot(S)
		K2 = self.LP.A.transpose()
		K3 = self.LP.A

		r1 = self.LP.c
		r2 = self.LP.b

		sol1 = SymmetricSystemSolver(K1,K2,K3,r1,r2)

		r1 = self.r_d - X_inv.dot(self.r_xs)
		r2 = self.r_p

		sol2 = SymmetricSystemSolver(K1,K2,K3,r1,r2)

		self.calculate_tau_step(sol1,sol2)
		self.calculate_primal_steps(sol1,sol2)
		self.calculate_dual_steps(sol1,sol2)

	def calculate_tau_step(self,sol1,sol2):
		num = self.curr_pt.tau*self.r_g + self.r_tk + self.curr_pt.tau*np.dot(self.LP.c.transpose(),sol2.u) - self.curr_pt.tau*np.dot(self.LP.b.transpose(),sol2.v)
		den = self.curr_pt.kappa - self.curr_pt.tau*np.dot(self.LP.c.transpose(),sol1.u) + self.curr_pt.tau*np.dot(self.LP.b.transpose(),sol1.v)

		d_tau = num / den

		self.step.dtau = d_tau

	def calculate_primal_steps(self,sol1,sol2):

		dx = sol2.u + sol1.u*self._dtau
		dy = sol2.v + sol1.v*self._dtau

		self.step.dx = dx
		self.step.dy = dy

	def calculate_dual_steps(self,sol1,sol2):
		X = np.diag(self.curr_pt.x.transpose()[0])
		X_inv = linalg.inv(X)

		S = np.diag(self.curr_pt.s.transpose()[0])

		ds = np.dot(X_inv,(self.r_xs - np.dot(S,self._dx)))
		dkappa = (self.r_tk - self.curr_pt.kappa*self._dtau)/self.curr_pt.tau

		self.step.ds = ds
		self.step.dkappa = dkappa
