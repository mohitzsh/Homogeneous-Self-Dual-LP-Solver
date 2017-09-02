'''
	Author: Mohit Sharma
	Email: sharm267@purdue.edu
'''

class Step:

	def __init__(self,dx=None,dy=None,ds=None,dtau=None,dkappa=None):
		self._dx = dx
		self._dy = dy
		self._ds = ds
		self._dtau = dtau
		self._dkappa = dkappa

	@property 
	def dx(self):
		return self._dx

	@property 
	def dy(self):
		return self._dy

	@property 
	def ds(self):
		return self._ds

	@property
	def dtau(self):
		return self._dtau

	@property 
	def dkappa(self):
		return self._dkappa

	@dx.setter
	def dx(self,val):
		return self._dx = val

	@dy.setter
	def dy(self,val):
		return self._dy = val

	@ds.setter
	def ds(self,val):
		return self._ds = val

	@dtau.setter
	def dtau(self,val):
		self._dtau = val

	@dkappa.setter
	def dkappa(self,val):
		self._dkappa = val

