'''
	Author: Mohit Sharma
	Email: sharm267@purdue.edu
'''

import scipy.optimize as opt
import numpy as np
from scipy.sparse import csc_matrix

#Define infinity
_INF = 1.00000000e+308
_NEGINF = -1*float("inf")

'''
	Convert the LP into standard form
	min c^T x
	s.t. Ax = b
	and  x >=0
'''
class SLP():

	def __init__(self, LP):

		self.LP = LP

		self.A = self.modified_A()
		self.c = self.modified_c()
		self.b = self.modified_b()

		self.n = self.nvars()
		self.m = self.neqs()

	def nvars(self):
		return self.LP.A.shape[1]
	def neqs(self):
		return self.LP.A.shape[0]

	def divide_mtrx(self):
		hi_ind = np.where(self.LP.hi != _INF)
		lo_ind = np.where(self.LP.lo != 0)

		# Indices where x has both lower and upper bound
		hi_lo_ind = np.intersect1d(hi_ind,lo_ind)

		# Indices where x only has an upper bound
		hi_ind = np.setdiff1d(hi_ind,hi_lo_ind)

		# Indices where x only has a lower bound
		lo_ind = np.setdiff1d(lo_ind,hi_lo_ind)

		# Indices where x is unbounded on both ends
		unbnd_ind = np.setdiff1d(np.arange(self.LP.A.shape[1]),np.union1d(np.union1d(hi_ind,lo_ind),hi_lo_ind))

		return hi_ind, lo_ind, hi_lo_ind, unbnd_ind

	'''
		Assuming for now that the problem is already in standard form
	'''
	def modified_A(self):
		# hi_ind, lo_ind, hi_lo_ind, unbnd_ind = self.divide_mtrx()

		# # Make A efficient for column slicing
		# # Don't use A for any other purpose, can be highly inefficient
		# A = csc_matrix(self.LP.A)

		# A_u = A[:,unbnd_ind]
		# A_hi = A[:,hi_ind]
		# A_lo = A[:,lo_ind]
		# A_hi_lo = A[:,hi_lo_ind]

		# print "A_u", A_u.shape
		# print "A_hi", A_hi.shape
		# print "A_lo", A_lo.shape
		# print "A_hi_lo", A_hi_lo.shape
		# row1 = np.bmat([A_u,-1*A_u,A_hi,-1*A_hi,A_lo,A_hi_lo,
		# 	np.zeros(A_hi.shape),np.zeros(A_hi_lo.shape)])

		# row2 = np.bmat([np.zeros(A_u.shape),np.zeros(A_u.shape),
		# 	np.zeros(A_hi.shape),np.zeros(A_hi.shape),np.zeros(A_lo.shape),
		# 	np.eye(A_hi_lo.shape),np.zeros(A_hi.shape),np.eye(A_hi_lo.shape)])

		# row3 = np.bmat([np.zeros(A_u.shape),np.zeros(A_u.shape),
		# 	np.eye(A_hi.shape),-1*np.eye(A_hi.shape),
		# 	np.zeros(A_lo.shape),np.zeros(A_hi_lo.shape),
		# 	np.eye(A_hi.shape),np.zeros(A_hi_lo.shape)])
		
		# return np.bmat([row1,row2,row3])

		return self.LP.A


	def modified_c(self):
		# hi_ind, lo_ind, hi_lo_ind, unbnd_ind = self.divide_mtrx()
		# c_u = self.LP.c[unbnd_ind,:]
		# c_hi = self.LP.c[hi_ind,:]
		# c_lo = self.LP.c[lo_ind,:]
		# c_hi_lo = self.LP.c[hi_lo_ind,:]
		# c = np.bmat[[c_u],[-1*c_u],[c_hi],[-1*c_i], [c_lo],[c_hi]]
		# return c

		return self.LP.c

	def modified_b(self):

		return self.LP.b

	def solve(self):
		res = opt.linprog(self.c,A_eq=self.A.todense(),b_eq = self.b,options={"disp": True})
		return res


