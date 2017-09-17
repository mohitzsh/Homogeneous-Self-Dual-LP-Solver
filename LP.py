'''
	Author: Mohit Sharma
	email: sharm267@purdue.edu
'''

import numpy as np 
import scipy.io as scio
import os
import scipy.optimize as opt
import scipy.sparse as sparse


'''
	Read the LP in the following form:
	
	min c^T x 
	s.t Ax = b 
		x<= hi 
		x>= lo
'''
class LP(object):

	'''
	Possible TODOs:
	1. Add check for A to be sparse
	2. Add method to ensure that only one of 'sprs_mtrx' or 'A,b,c,hi,lo' is given

	Current Implementation includes:
	1. Read the matrix from the UFL sparse matrix repository


	####################################################
	# USING MM (Matric Market) format to read the data #
	####################################################
	The data dir should look like this
	lp_agg/
		lp_agg.mtx
		lp_agg_b.mtx 
		lp_agg_c.mtx 
		lp_agg_hi.mtx 
		lp_agg_lo.mtx 
		lp_agg_z0.mtx 
	'''
	def __init__(self, base_dir, mtrx_name = "lp_agg"):
		self.base_dir = base_dir
		self.mtrx_name ='lp_agg'
		self.A = sparse.csc_matrix(self.mtrx_read(''))
		self.b = self.mtrx_read('_b')
		self.c = self.mtrx_read('_c')
		self.hi = self.mtrx_read('_hi')
		self.lo = self.mtrx_read('_lo')

		self.shape = self.A.shape

	def mtrx_read(self,suff):
		path = os.path.join(self.base_dir,self.mtrx_name,self.mtrx_name + suff + '.mtx')
		return scio.mmread(path)

	'''
		Solve the LP using scipy.optimize.linprog

		NOTE:
		scipy.optimize.linprog doesn't work with sparse matrices. Convert to dense to check the result

	'''
	def solve(self):
		bounds = np.column_stack((self.lo,self.hi))
		res = opt.linprog(np.reshape(self.c,(self.c.shape[0])),A_eq = self.A.todense(),b_eq=self.b,method='interior point',options={"disp": True})

		return res
