'''
	Author: Mohit Sharma
	Email: sharm267@purdue.edu
'''

import numpy as np
import numpy.linalg as linalg
import scipy.sparse as sparse
from SymmetricSystemSolver import SymmetricSystemSolver
from Step import Step
import util

'''
	For a system of non-linear equations, return the newton step.
'''
class NewtonStep(Step):

	def __init__(self,LP,curr_pt,eta,gamma,residual_step=None):

		self._LP = LP
		self._curr_pt = curr_pt

		# Add checks for the values of eta and gamma
		self._eta = eta
		self._gamma = gamma



		# Residuals at current point
		self._r_p = self._curr_pt.tau*self._LP.b - self._LP.A.dot(self._curr_pt.x)
		self._r_d = self._curr_pt.tau*self._LP.c - self._LP.A.transpose().dot(self._curr_pt.y) - self._curr_pt.s


		self._r_g = self._curr_pt.kappa + util.dot(self._LP.c,self._curr_pt.x) - util.dot(self._LP.b,self._curr_pt.y)

		# complementarity-gap at current point
		self._mu = np.true_divide(util.dot(self._curr_pt.x,self._curr_pt.s) + self._curr_pt.kappa*self._curr_pt.tau,self._LP.shape[1] + 1)
		self._r_xs = np.dot(-1*np.diag(self._curr_pt.x.transpose()[0]),self._curr_pt.s) + self._gamma*self._mu*np.ones((self._LP.shape[1],1))
		self._r_tk = -1*self._curr_pt.tau*self._curr_pt.kappa + self._gamma*self._mu

		# Add the residual term
		self._r_xs = self._r_xs - np.dot(np.diag(residual_step.dx.transpose()[0]),residual_step.ds) if residual_step is not None else self._r_xs
		self._r_tk = self._r_tk - residual_step.dtau* residual_step.dkappa if residual_step is not None else self._r_tk 

		# This functions calculates all the steps and assign them to class variables
		self.compute_step()

	'''

		TODO:
		Find a consistent way to construct sparse matrices directly
	'''	

	def compute_step(self):
		
		X = sparse.diags([self._curr_pt.x.transpose()[0]],[0])
		S = sparse.diags([self._curr_pt.s.transpose()[0]],[0])

		X_inv  = sparse.linalg.inv(X)
	

		'''
			Solve K (p;q) = (c;b)
			
			where  K = [[K1 K2],[K3 0]]

		'''
		K1 = (-1*X_inv).dot(S)
		K2 = self._LP.A.transpose()
		K3 = self._LP.A

		r1 = self._LP.c
		r2 = self._LP.b

		sol1 = SymmetricSystemSolver(K1,K2,K3,r1,r2)

		r1 = self._r_d - X_inv.dot(self._r_xs)
		r2 = self._r_p

		sol2 = SymmetricSystemSolver(K1,K2,K3,r1,r2)

		self.calculate_tau_step(sol1,sol2)
		self.calculate_primal_steps(sol1,sol2)
		self.calculate_dual_steps(sol1,sol2)

	def calculate_tau_step(self,sol1,sol2):
		num = self._curr_pt.tau*self._r_g + self._r_tk + self._curr_pt.tau*util.dot(self._LP.c,sol2.u) - self._curr_pt.tau*util.dot(self._LP.b,sol2.v)
		den = self._curr_pt.kappa - self._curr_pt.tau*util.dot(self._LP.c,sol1.u) + self._curr_pt.tau*util.dot(self._LP.b,sol1.v)

		d_tau = num / den

		self.dtau = d_tau

	def calculate_primal_steps(self,sol1,sol2):

		dx = sol2.u + sol1.u*self.dtau
		dy = sol2.v + sol1.v*self.dtau

		self.dx = dx
		self.dy = dy

	def calculate_dual_steps(self,sol1,sol2):
		X = sparse.diags([self._curr_pt.x.transpose()[0]],[0])
		X_inv = sparse.linalg.inv(X)

		S = sparse.diags([self._curr_pt.s.transpose()[0]],[0])

		ds = X_inv.dot(self._r_xs - S.dot(self.dx))
		
		dkappa = (self._r_tk - self._curr_pt.kappa*self.dtau)/self._curr_pt.tau

		self.ds = ds
		self.dkappa = dkappa
