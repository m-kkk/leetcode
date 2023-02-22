# segment tree implementation with lazy propagation
"""
Space complexity: O(N)
calculation time complexity: O(logN)
single modification time complexity: O(logN)
range modification time complexity: O(logN)
create time complexity: O(N*logN)
"""
import math
class seg_tree(object):
	def __init__(self,nums):
		n = len(nums)
		self.tree = [0]*(2**(int(math.log(n,2))+2))
		self.lazy = [0]*(2**(int(math.log(n,2))+2))
		self.create_tree(0,n-1,1,nums)

	def create_tree(self,start,end,ti,nums):
		if start==end:
			print(start,end,ti)
			self.tree[ti] = nums[start]
		else:
			mid = (start+end)//2
			self.create_tree(start,mid,ti*2,nums)
			self.create_tree(mid+1,end,ti*2+1,nums)
			self.tree[ti] = self.tree[ti*2] + self.tree[ti*2+1]

	def query_sum(self,start,end,l,r,ti):
		if l>end or r<start:
			return 0
		if self.lazy[ti] != 0:
			self.tree[ti] += (end-start+1)*lazy[ti]
			if start!=end:
				lazy[ti*2] = lazy[ti]
				lazy[ti*2+1] = lazy[ti]
			lazy[ti] = 0

		if l<=start and r>=end:
			return self.tree[ti]
		mid = (start+end)//2
		left = self.query_sum(start,mid,l,r,ti*2)
		right = self.query_sum(mid+1,end,l,r,ti*2+1)
		return left+right

	def update_single(self,ti,start,end,i,val):
		if i<start or i >end:
			return
		else:
			self.tree[ti]+=val
			if start==end:
				return
			mid = (start+end)//2
			self.update_single(ti*2,start,mid,i,val)
			self.update_single(ti*2+1,mid+1,end,i,val)

	def update(self,n,i,val):
		self.update_single(1,0,n-1,i,val)
	def get_sum(self,l,r,n):
		return self.query_sum(0,n-1,l,r,1)

	def update_range(self,n,l,r,val):
		self.update_range_lazy_propagation(1,0,n-1,l,r,val)


	def update_range_lazy_propagation(self,ti,start,end,l,r,val):
		if self.lazy[ti]!=0:
			self.tree[ti] += (end-start+1)*self.lazy[ti]
			if start!=end:
				self.lazy[ti*2] += self.lazy[ti]
				self.lazy[ti*2+1] += self.lazy[ti]
			self.lazy[ti] = 0
		#out of range
		if l>end or r<start:
			return
		#range is with in the segment 
		if start>=l and end<=r:
			self.tree[ti] += (end-start+1)*val
			if start!=end:
				self.lazy[ti*2] += val
				self.lazy[ti*2+1] += val
			return
		mid = (start+end)//2
		self.update_range_lazy_propagation(ti*2,start,mid,l,r,val)
		self.update_range_lazy_propagation(ti*2+1,mid+1,end,l,r,val)
		self.tree[ti] = self.tree[ti*2]+self.tree[ti*2+1]


st =  seg_tree([1,2,3,4,5,6,7])
print(st.tree)
print(st.get_sum(0,5,7))
st.update_range(7,0,4,3)
print(st.tree)
print(st.get_sum(0,5,7))
st.update_range(7,0,4,3)
st.update_range(7,0,4,-3)
print(st.get_sum(0,5,7))
st.update_range(7,0,4,2)
print(st.get_sum(0,5,7))