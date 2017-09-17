'''
	Author: Mohit Sharma
	Email: sharm267@purdue.edu
'''
import scipy.sparse as sparse
import numpy as np

'''
	Implements solver for Ax = b

	Method based on Normal Equation Systems
		- Returns (u;v) such that 
			[-X^-1*S  A^T](u;v) = r1
			[A 0](u;v) = r2


	NOTE: K1,K2,K3 are expected to be csc_matrix matrices

'''
class SymmetricSystemSolver(object):

	def __init__(self,K1,K2,K3,r1,r2):

		# Primary Vas
		self._u = None
		self._v = None

		# Helper Vars
		self.K1 = sparse.csc_matrix(K1)
		self.K2 = sparse.csc_matrix(K2)
		self.K3 = sparse.csc_matrix(K3)
		self.r1 = sparse.csc_matrix(r1)
		self.r2 = sparse.csc_matrix(r2)

		self.solve()

	@property 
	def u(self):
		return self._u

	@property 
	def v(self):
		return self._v

	def solve(self):

		# print "K1: ", self.K1.shape
		# print "K2: ", self.K2.shape
		# print "K3: ", self.K3.shape

		# print "r1: ", self.r1.shape
		# print "r2: ", self.r2.shape
		
		D = -1*self.K1
		D_inv = sparse.linalg.inv(D)
		M = self.K3.dot(D_inv).dot(self.K2)

		# print "Type of M: ", type(M)

		b = self.r2 + self.K3.dot(D_inv).dot(self.r1)
		v = sparse.linalg.spsolve(M,b,permc_spec="MMD_ATA").reshape((self.K3.shape[0],1))
		u = D_inv.dot(self.K2.dot(v) - self.r1)

		self._u = np.asarray(u)
		self._v = np.asarray(v)

