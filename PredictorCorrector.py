'''
	Author: Mohit Sharma
	Email: sharm267@purdue.edu
'''

from NewtonStep import NewtonStep
from Step import Step
from StepSize import StepSize
import numpy as np
import math
import util

'''
	For the current point, completes the predictor step and the corrector step and returns the 
	step for primal and dual variables
'''
class PredictorCorrector(Step):

	def __init__(self,LP,curr_pt):
		self.LP = LP
		self._curr_pt = curr_pt

		self._gamma = None
		self._eta = None

		self.beta_1 = 0.1

		step = self.predictor_corrector()
		super(PredictorCorrector,self).__init__(dx=step.dx,dy=step.dy,ds=step.ds,dtau=step.dtau,dkappa=step.dkappa)


	@property 
	def gamma(self):
		return self._gamma

	@property
	def eta(self):
		return self._eta 

	@gamma.setter
	def gamma(self,val):
		self._gamma = val

	@eta.setter
	def eta(self,val):
		self._eta = val


	'''
		Predictor steps get the Newton Step for gamma = 0 and eta = 1
	'''
	def predictor_corrector(self):
		self.gamma = 0
		self.eta = 1

		predictor_step = NewtonStep(self.LP,self._curr_pt,self.eta,self.gamma)

		# Get the step lengths for primal and dual and calculate the alpha and eta parameters for the corrector step

		step_size = StepSize(self._curr_pt,predictor_step)

		self.set_corrector_params(step_size,predictor_step)

		corrector_step = NewtonStep(self.LP,self._curr_pt,self.eta,self.gamma,residual_step=predictor_step)

		return corrector_step

	'''
		Set the values of Gamma and Eta for the corrector step
	'''
	def set_corrector_params(self,step_size,predictor_step):

		# alpha_t = np.min((step_size.alpha_p,step_size.alpha_d)) if step_size.same_step else step_size.alpha_t
		# alpha_k = np.min((step_size.alpha_p,step_size.alpha_d)) if step_size.same_step else step_size.alpha_k

		s1 = util.dot((self._curr_pt.x + step_size.alpha_p*predictor_step.dx),self._curr_pt.s + step_size.alpha_d*predictor_step.ds)
		s2 = (self._curr_pt.tau + step_size.alpha_t*predictor_step.dtau)*(self._curr_pt.kappa + step_size.alpha_k*predictor_step.dkappa)

		mu_a = s1 + s2
		mu_k = util.dot(self._curr_pt.x,self._curr_pt.s) + self._curr_pt.tau*self._curr_pt.kappa

		# Calculate the Gamma and Eta parameters

		if mu_a/mu_k <= 0.01:
			self.gamma = math.pow(mu_a/mu_k,2)
		else:
			self.gamma = np.min([0.1,np.max([math.pow(mu_a/mu_k,3),0.0001])])

		self.eta = 1 - self.gamma

	'''
	Get the maximum step sizes for each variable without violating the non-negativity constraints
	'''
	def compute_non_negative_step_size(self):
		keep_x = np.where(self._step.dx <=0)[0]
		alpha_x = np.min(np.true_divide(self._curr_pt.x[keep_x],self._step.dx[keep_x]))

		keep_s = np.where(self._step.ds <=0)[0]
		alpha_s = np.min(np.true_divide(self._curr_pt.s[keep_s],self._step.ds[keep_s]))

		alpha_tau = -1*self._curr_pt.tau / self._step.dtau
		alpha_kappa = -1*self._curr_pt.kappa / self._step.dkappa

		assert(alpha_x>0)
		assert(alpha_s>0)
		assert(alpha_tau>0)
		assert(alpha_kappa >0)

		self.alpha_p = np.min([alpha_x,alpha_tau,alpha_kappa])
		self.alpha_d = np.min([alpha_s,alpha_tau,alpha_kappa])

	def get_alpha_max(self,step):
		z = np.vstack((self.curr_pt.x,self._curr_pt.tau,self._curr_pt.s,self._curr_pt.kappa))
		d = np.vstack((step.dx,step.dtau,step.ds,step.dkappa))

		assert(z.shape == d.shape)

		keep = np.where(d <= 0)[0]

		alpha_max = np.min(np.true_divide(z[keep],d[keep]))

		assert(alpha_max >= 0)

		return alpha_max
