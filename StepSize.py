'''
	Author: Mohit Sharma
	Email: sharm267@purdue.edu
'''

import numpy as np
import util
'''
	For any step at any point, find the step size by the heuristic mentioned in Section 4.5, X. Xu et al, A simplified
	self-dual LP algorithm
'''
class StepSize(object):

	def __init__(self,curr_pt,step):

		# Primal Step Size
		self._alpha_p = None
		#Dual Step Size 
		self._alpha_d = None

		# Which alpha (primal or dual) to use for Tau
		self._alpha_t = None

		# Which alpha (primal or dual) to use for kappa
		self._alpha_k = None

		# Are we taking same step-size for primal and dual vars
		self._same_step = None

		#---------------------#
		#  Private Variables  #
		#---------------------#

		self._curr_pt = curr_pt

		self._step = step

		# Taken from X. Xu et al
		self._beta = 0.99995

		self.compute_step_size()

	@property 
	def alpha_p(self):
		return self._alpha_p

	@property 
	def alpha_d(self):
		return self._alpha_d

	@property 
	def alpha_t(self):
		return self._alpha_t

	@property
	def alpha_k(self):
		return self._alpha_k

	@property 
	def same_step(self):
		return self._same_step

	@alpha_p.setter
	def alpha_p(self,val):
		self._alpha_p = val

	@alpha_d.setter
	def alpha_d(self,val):
		self._alpha_d = val

	@alpha_t.setter
	def alpha_t(self,val):
		self._alpha_t = val

	@alpha_k.setter
	def alpha_k(self,val):
		self._alpha_k = val

	@same_step.setter
	def same_step(self,val):
		self._same_step = val

	def compute_step_size(self):
		keep_x = np.where(self._step.dx <0)[0]
		alpha_x = np.min(np.true_divide(-1*self._curr_pt.x[keep_x],self._step.dx[keep_x]))

		keep_s = np.where(self._step.ds <0)[0]
		alpha_s = np.min(np.true_divide(-1*self._curr_pt.s[keep_s],self._step.ds[keep_s]))


		alpha_tau = (-1*self._curr_pt.tau / self._step.dtau) if self._step.dtau < 0 else 1
		alpha_kappa = (-1*self._curr_pt.kappa / self._step.dkappa) if self._step.dkappa < 0 else 1
		

		import pdb; pdb.set_trace()
		
		# Assert that these are indeed the maximum step sizes you can take
		assert(np.all((self._curr_pt.x + alpha_x*self._step.dx) >=0))
		assert(np.all((self._curr_pt.s + alpha_s*self._step.ds) >=0))
		assert((self._curr_pt.tau + alpha_tau*self._step.dtau) >=0)
		assert((self._curr_pt.kappa + alpha_kappa*self._step.dkappa) >=0)

		# import pdb; pdb.set_trace()

		assert(alpha_x > 0)
		assert(alpha_s > 0)
		assert(alpha_tau > 0)
		assert(alpha_kappa > 0)

		self.alpha_p = self._beta*np.min([alpha_x,alpha_tau,alpha_kappa,1])
		self.alpha_d = self._beta*np.min([alpha_s,alpha_tau,alpha_kappa,1])

		self.same_step = self.is_same_step()

		# if self.same_step:
		# 	print "Same Step to be taken"
		# else:
		# 	print "Different Steps to be taken"

		# Figure out which step size to take for tau and kappa
		if not self.same_step:

			# Pick alpha which leads to smaller tau, use the same alpha for kappa

			tau_p = self._curr_pt.tau + self.alpha_p*self._step.dtau
			tau_d = self._curr_pt.tau + self.alpha_d*self._step.dtau

			self.alpha_t = self.alpha_p if tau_p < tau_d else self.alpha_d

			# alpha_k is the other one

			self.alpha_k = self.alpha_p if self.alpha_t == self.alpha_d else self.alpha_d

	def is_same_step(self):
		if (self.alpha_p > self.alpha_d) and (util.dot(self._curr_pt.s,self._step.dx) <= 0) :
			return False
		if (self.alpha_p < self.alpha_d) and (util.dot(self._curr_pt.x,self._step.ds) <= 0):
			return False
		return True
