'''
	Author: Mohit Sharma
	Email: sharm267@purdue.edu
'''

class Solution():

	'''
		The Solution format is the same as scipy.optimize.linprog

		TODO: 
		1. Modify this to be the Solution class for the IP method (Currently only set up for simplex)
	'''
	def __init__(self,x,fun,slack,success,status,nit,message):
		self.status_dict = {
			0 : "Optimization terminated successfully"
			1 : "Iteration limit reached"
			2 : "Problem appears to be infeasible"
			3 : "Problem appears to be unbounded"
		}

		assert(status in self.status_dict)

		self.x = x
		self.fun = fun
		self.slack = slack
		self.success = success
		self.nit = nit
		self.message = self.status_dict[status]