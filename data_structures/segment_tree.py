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
		root.val = root.left.val+root.right.val
		return root

	def getSum(self,i,j):
		return self.getSum_help(i,j,self.root)

	def getSum_help(self,start,end,root):
		if start<=root.start and end>=root.end:
			return root.val
		if start>root.end or end<root.start:
			return 0
		return self.getSum_help(start,end,root.left)+self.getSum_help(start,end,root.right)

	def update(self,i,value):
		prev = self.get_value(i,self.root)
		dif = value-prev
		self.update_help(self.root,i,dif)

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
		root.val = root.val+value
		if root.start==root.end:
			return
		self.update_help(root.left,i,value)
		self.update_help(root.right,i,value)


if __name__=="__main__":
	tree = seg_tree([1,2,3,4,5,6,7,8])
	print(tree.getSum(4,7))
	print(tree.get_value(5,tree.root))
	tree.update(5,1)
	print(tree.get_value(5,tree.root))
	print(tree.getSum(0,3))
