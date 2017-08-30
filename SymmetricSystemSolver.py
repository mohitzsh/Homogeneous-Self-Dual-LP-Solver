'''
	Author: Mohit Sharma
	Email: sharm267@purdue.edu
'''
from scipy.sparse.linalg import spsolve

'''
	Solves Ax = b where 
	A is symmetric (Potentially Sparse also)

	List of Methods Available:
	1> Standard solver (Use as default)
	2> Method based on Normal Equation Systems
'''
class SymmetricSystemSolver:

	def __init__(self,A,b,method = None):
		self.A = A
		self.b = b

		self.sol = self.solve(method)

	def solve(self,method):
		if method is None:
			return self.default_solver()

	def default_solver(self):
		return spsolve(self.A,self.b)
