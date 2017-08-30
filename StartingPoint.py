'''
	Author: Mohit Sharma
	email: sharm267@purdue.edu
'''

import numpy as np
import Point
'''
	Holds the initial value for the primal and dual variables.

	TODO:

	The simple starting point scheme suggested in the paper is already handled in the Point Class.
	Use this if you need a fancy heuristic to initialize the starting point
'''
class StartingPoint(Point):

	def __init__(self):
		