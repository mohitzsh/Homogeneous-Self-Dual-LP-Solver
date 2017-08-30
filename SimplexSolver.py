'''
	Author: Mohit Sharma
	Email: sharm267@purdue.edu
'''

import scipy.optimize.linprog as lp
import numpy as np

import Solution.Solution

'''
	Takes in LP in original form as well as standard form
	and returns the optimial value and the optimizer.
	
	Use scipy.optimize.linprog

	NOTE: The SLP would require transformation of variable of optimization to get optimizer
	for original problem.
'''

class SimplexSolver():

	def __init__(self,LP,SLP):
		self.LP = LP
		self.SLP = SLP

	'''
		Solves LP in the UF LP form
		min c^T x
		s.t Ax = b
		    lo <= x <= hi   
	'''
	def solveLP(self):
		res = linprog(LP.c,A_eq = LP.A,b_eq=LP.b,bounds=np.column_stack((LP.lo,LP.hi)),options={"disp": True})


	'''
		Solves LP in the standard form

		min c^T x
		s.t. Ax = b
			 x>=0
	'''
	def solveSLP(self):
