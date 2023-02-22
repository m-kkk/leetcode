#Fenwick Tree implementation
#2017/4/11
"""
a fenwick tree or a binary index tree is a data structure 
to calculation a prefix sum of an array and maintain the value.
It provide efficient calculation and modification
Space complexity: O(N)
calculation time complexity: O(logN)
modification time complexity: O(logN)
create time complexity: O(N*logN)
"""

class FenwickTree(object):
	def __init__(self, input_array):
		self.length=len(input_array)+1
		self.fenwich_tree=[0]*self.length
		self.createTree(input_array)

	def createTree(self, input_array):
		#tree start with index 1 and has length n+1,

		for i in range(len(input_array)):
			self.updateTree(i,input_array[i])

	def updateTree(self, index, value):
		#start from index+1 adding numbers until index out of range.
		#note that value is the difference after modifying 0->2 value=2 if 2->0 value=-2
		tree_index=index+1
		while tree_index<self.length:
			self.fenwich_tree[tree_index]+=value
			tree_index=self.getNext(tree_index)

	def getNext(self, index):
		#adding the right most 1 to index
		return index+(index & -index)

	def getPrev(self, index):
		#removing the right most 1 to index
		return index-(index & -index)

	def getSum(self, index):
		#start from index+1 adding the number to the sum until reach 0 in tree.
		tree_index=index+1
		res=0
		while tree_index>0:
			res+=self.fenwich_tree[tree_index]
			tree_index=self.getPrev(tree_index)
		return res

def test():
	l=[1,2,3,4,5,6,7,8,9,10]
	tr=FenwickTree(l)
	for i in range(len(l)):
		assert(tr.getSum(i)==sum(l[:i+1]))
	tr.updateTree(3,-1)
	print("pass calculation")
	for i in range(len(l)):
		if i<3:
			assert(tr.getSum(i)==sum(l[:i+1]))
		else:
			assert(tr.getSum(i)==sum(l[:i+1])-1)
	print("pass modification")

if __name__ == "__main__":
	test()


