'''
	Author: Mohit Sharma
	email: sharm267@purdue.edu
'''

import numpy as np 
import scipy.io as scio
import os

#Define initinity
_INF = float("inf")
_NEGINF = -1*float("inf")


# Store the Linear Program
class LP():

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
		self.A = self.mtrx_read('')
		self.b = self.mtrx_read('_b')
		self.c = self.mtrx_read('_c')
		self.hi = self.mtrx_read('_hi')
		self.lo = self.mtrx_read('_lo')

	def mtrx_read(self,suff):
		path = os.path.join(self.base_dir,self.mtrx_name,self.mtrx_name + suff + '.mtx')
		return scio.mmread(path)


if __name__ == "__main__":
	myLP = LP(base_dir = os.path.join(os.environ['HOME'],'InteriorPointMethods','data'), mtrx_name = 'data')
