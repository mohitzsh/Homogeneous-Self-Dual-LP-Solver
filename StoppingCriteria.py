'''
	Author: Mohit Sharma
	Email: sharm267@purdue.edu
'''

import numpy as np
import pdb
import util
'''
	Using the simple heuristics from X. Xu et al, A simplified self-dual LP algorithm
'''

def is_optimal(LP,curr_pt,start_pt,tol1,tol2,tol3):

	# Check the duality gap
	
	cond1 = np.true_divide(np.absolute(util.dot(LP.c,curr_pt.x) - util.dot(LP.b,curr_pt.y)), curr_pt.tau + np.absolute(util.dot(LP.b,curr_pt.y))) < tol1
	# Check primal infeasibility

	rp = curr_pt.tau*LP.b - LP.A.dot(curr_pt.x)

	cond2 = np.true_divide(np.linalg.norm(rp),curr_pt.tau + np.linalg.norm(curr_pt.x)) < tol1

	# Check dual infeasibility

	rd = curr_pt.tau*LP.c - LP.A.transpose().dot(curr_pt.y) - curr_pt.s

	cond3 = np.true_divide(np.linalg.norm(rd), curr_pt.tau + np.linalg.norm(curr_pt.s)) < tol1


	# Check for infeasibility


	mu_k = np.true_divide(util.dot(curr_pt.x,curr_pt.s) + curr_pt.tau*curr_pt.kappa,LP.shape[1]+1)
	mu_0 = np.true_divide(util.dot(start_pt.x,start_pt.s) + start_pt.tau*start_pt.kappa,LP.shape[1]+1)

	cond4 = np.true_divide(mu_k,mu_0) < tol2

	cond5 = np.true_divide(curr_pt.tau*start_pt.kappa,curr_pt.kappa*start_pt.tau) < tol3

	optimal = cond1 and cond2 and cond3

	infeasible = cond4 and cond5

	converge = optimal or infeasible

	return {
		"converge": converge,
		"optimal" : optimal,
		"infeasible": infeasible
	}
