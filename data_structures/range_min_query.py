# segment tree implementation
"""
Space complexity: O(N)
calculation time complexity: O(logN)
modification time complexity: O(logN)
create time complexity: O(N*logN)
"""
class tree_node(object):
	def __init__(self,val,start,end):
		self.val = val
		self.start = start
		self.end = end
		self.left = None
		self.right = None


class seg_tree(object):
	def __init__(self,L):
		n = len(L)
		self.root = self.create_tree(0,n-1,L)

	def create_tree(self,start,end,L):
		root = tree_node(0,start,end)
		if start == end:
			root.val = L[start]
			return root
		mid = (start+end)//2
		root.left = self.create_tree(start,mid,L)
		root.right = self.create_tree(mid+1,end,L)
		root.val = min(root.left.val,root.right.val)
		return root

	def getMin(self,i,j):
		return self.getMin_help(i,j,self.root)

	def getMin_help(self,start,end,root):
		if start<=root.start and end>=root.end:
			return root.val
		if start>root.end or end<root.start:
			return None
		left = self.getMin_help(start,end,root.left)
		right = self.getMin_help(start,end,root.right)
		if left and right:
			return min(left,right)
		else:
			return left if left else right



	def update(self,i,value):
		self.update_help(self.root,i,value)

	def get_value(self,i,root):
		if root.start>i or root.end<i:
			return 0
		if root.start==root.end:
			if root.start==i:
				return root.val
		else:
			return self.get_value(i,root.left)+self.get_value(i,root.right)

	def update_help(self,root,i,value):
		if root.start>i or root.end<i:
			return 
		if value <= root.val:
			root.val = value
		if root.start==root.end:
			if root.start==i:
				root.val = value
			return
		self.update_help(root.left,i,value)
		self.update_help(root.right,i,value)


if __name__=="__main__":
	L = list(range(1,11))
	tree = seg_tree(L)
	for i in range(10):
		assert(tree.get_value(i,tree.root) == L[i])
	print("Pass get value")
	print(tree.getMin(4,7))
	tree.update(4,-1)
	print(tree.getMin(0,4))

	
	
