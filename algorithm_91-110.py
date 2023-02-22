#91. Decode Ways
class Solution(object):
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        if s=="" or int(s[0])==0:
            return 0
        dp=[0]*(len(s)+1)
        dp[0],dp[1]=1,1
        for i in range(1,len(s)):
            if not s[i].isdigit():
                return False
            else:
                val=int(s[i-1:i+1])
                if val<=26 and val>9:
                    dp[i+1]+=dp[i-1]
                if int(s[i])!=0:
                    dp[i+1]+=dp[i]
        return dp[len(s)]

solution=Solution()
print(solution.numDecodings("123712"))

#92. Reverse Linked List II(Medium)
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
    def __str__(self):
        return str(self.val)
L=ListNode(1)
L.next=ListNode(2)
L.next.next=ListNode(3)
class Solution(object):
    def reverseBetween(self, head, m, n):
        """
        :type head: ListNode
        :type m: int
        :type n: int
        :rtype: ListNode
        """
        res=ListNode(-1)
        res.next=head
        prev=res
        for i in range(1,n+1):
            if i<m:
                prev=prev.next
            if i==m:
                start=head
            if i==n:
                end=head
            head=head.next
        while prev.next!=end:
            temp=start.next
            start.next=temp.next
            temp.next=prev.next
            prev.next=temp
        return res.next
solution=Solution()
print(solution.reverseBetween(L,2,3).next.next)

#93. Restore IP Addresses
class Solution(object):
    def restoreIpAddresses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        ips=[]
        self.serch(0,s,ips,"")
        return ips
    def serch(self,i,s,ips,ip):
        if i==4:
            if s=="":
                ips.append(ip[1:])
            return
        for j in range(1,4):
            if j<=len(s):
                if int(s[0:j])<=255:
                    self.serch(i+1,s[j:],ips,ip+"."+s[0:j])
                if s[0]=="0":
                    break


#94. Binary Tree Inorder Traversal(Medium)
#solution1:recursive
class Solution(object):
    def inorderTraversal(self, root):
        if root==None:
            return []
        res=[]
        if root.left!=None:
            res+=(self.inorderTraversal(root.left))
        res.append(root.val)
        if root.right!=None:
            res+=(self.inorderTraversal(root.right))
        return res
    #solution2:nonrecursive
    def inorderTraversal(self, root):
        if root==None:
            return []
        stack=[]
        res=[]
        while(stack!=[] or root!=None):
            if root!=None:
                stack.append(root)
                root=root.left
            else:
                root=stack.pop()
                res.append(root.val)
                root=root.right
        return res


class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
    def __repr__(self):
        if not self.left and not self.right:
            return str(self.val)
        elif self.left and not self.right:
            return str(self.val)+"left:"+str(self.left.val)
        elif self.right and not self.left:
            return str(self.val)+"right:"+str(self.right.val)
        else:
            return str(self.val)+"left:"+str(self.left.val)+"right:"+str(self.right.val)

#95.Unique Binary Search Trees II(Medium)
class Solution(object):
    def generateTrees(self, t,n):
        """
        :type n: int
        :rtype: List[TreeNode]
        """
        if n==0:
            return []
        return self.dfs(t,n)
    def dfs(self,start,end):
        if start>end:
            return [None]
        else:
            res=[]
            for i in range(start,end+1):
                lefts=self.dfs(start,i-1)
                rights=self.dfs(i+1,end)
                for p in lefts:
                    for q in rights:
                        t=TreeNode(i)
                        t.left=p
                        t.right=q
                        res.append(t)
            return res
print('------------------------')
s = Solution()
a = s.generateTrees(1,3)

def print_node(r):
    res=[]
    if r==None:
        return [None]
    res+=[r.val]
    res+=print_node(r.left)
    res+=print_node(r.right)
    return res

print([print_node(b) for b in a])
#96. Unique Binary Search Trees
class Solution(object):
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        dp=[1,1,2]
        i=3
        while(i<=n):
            res=0
            for j in range(i):
                res+=dp[j]*dp[i-j-1]
            dp.append(res)
            i+=1
        return dp[n]

#97. Interleaving String(Hard)
class Solution(object):
    def isInterleave(self, s1, s2, s3):
        """
        :type s1: str
        :type s2: str
        :type s3: str
        :rtype: bool
        """
        if len(s1)+len(s2)!=len(s3):
            return False
        dp=[[False for i in range(len(s2)+1)] for i in range(len(s1)+1)]
        dp[0][0]=True
        for i in range(len(s1)):
            dp[i+1][0]=(dp[i][0] and s1[:i+1]==s3[:i+1])
        for j in range(len(s2)):
            dp[0][j+1]=(dp[0][j] and s2[:j+1]==s3[:j+1])
        for i in range(len(s1)):
            for j in range(len(s2)):
                if s1[i]==s3[i+j+1]:
                    dp[i+1][j+1]|=dp[i][j+1]
                if s2[j]==s3[i+j+1]:
                    dp[i+1][j+1]|=dp[i+1][j]
        return dp[len(s1)][len(s2)]

    def isInterleave(self, s1, s2, s3):
        """
        :type s1: str
        :type s2: str
        :type s3: str
        :rtype: bool
        """
        if len(s1) + len(s2) != len(s3):
            return False
        i, j, p = 0, 0, 0 # i, j, p are pointers for s1, s2 and s3
        while i >= 0 and j < len(s2):
            if i < len(s1) and s3[p] == s1[i]: # always choose the first string
                i, p = i + 1, p + 1
            elif s3[p] == s2[j]: # if the first string doesn't match, we choose the second string
                j, p = j + 1, p + 1
            else: 
            # if choosing the first string was wrong in previous steps
            # we retrospect with choosing the second string in previous steps
                i, j = i - 1, j + 1
        #when we finish s1 or s2, check if the rest of the other string match the rest of s3
        return s1[i:] + s2[j:] == s3[p:] and i >= 0

    def isInterleave(self, s1, s2, s3):
        """
        :type s1: str
        :type s2: str
        :type s3: str
        :rtype: bool
        """
        def dfs(i, j, k):
            if (i, j, k) not in memo:
                memo[(i, j, k)] = k>=l3 or (i<l1 and s3[k]==s1[i] and dfs(i+1,j,k+1)) or (j<l2 and s3[k]==s2[j] and dfs(i,j+1,k+1))
            return memo[(i, j, k)]
        l1, l2, l3, memo = len(s1), len(s2), len(s3), {}
        if l3 != l1 + l2: return False
        return dfs(0, 0, 0)

#98. Validate Binary Search Tree
class Solution(object):
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        return self.is_bst(root,None,None)
    def is_bst(self,root,lower,upper):
        if root==None:
            return True
        p=root.val
        return (lower==None or lower<p) and (upper==None or upper>p) and (self.is_bst(root.left,lower,p)) and (self.is_bst(root.right,p,upper))
solution=Solution()
# print(solution.isInterleave("ab","bc","babc"))

#99. Recover Binary Search Tree
class Solution(object):
    def recoverTree(self, root):
        """
        :type root: TreeNode
        :rtype: void Do not return anything, modify root in-place instead.
        """
        self.first=None
        self.second=None
        self.last=None
        self.traversal(root)
        self.first.val,self.second.val=self.second.val,self.first.val

    def traversal(self,root):
        if root==None:
            return
        if root.left==None and self.last==None:
            self.last=root
        self.traversal(root.left)
        if self.first==None and root.val<self.last.val:
            self.first=self.last
        if self.first!=None and root.val<self.last.val:
            self.second=root
        self.last=root
        self.traversal(root.right)

#100. Same Tree(Easy)
class Solution(object):
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        if p==None and q==None:
            return True
        elif p!=None and q!=None:
            if p.val==q.val and self.isSameTree(p.left,q.left) and self.isSameTree(p.right,q.right):
                return True
            else:
                return False
        else:
            return False

#101. Symmetric Tree
class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if root==None:
            return True
        return isSym(root.left,root.right)
    def isSym(self,left,right):
        if left==None and right==None:
            return True
        if left!=None and right==None:
            return False
        if left==None and right!=None:
            return False
        return left.val==right.val and self.isSym(left.left,right.right) and self.isSym(left.right,right.left)

#102. Binary Tree Level Order Traversal(Medium)
class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if root==None:
            return []
        res=[]
        queue=[root]
        while(queue!=[]):
            next_q=[]
            cur=[]
            for node in queue:
                cur.append(node.val)
                if node.left!=None:
                    next_q.append(node.left)
                if node.right!=None:
                    next_q.append(node.right)
            res.append(cur)
            queue=next_q
        return res




#103. Binary Tree Zigzag Level Order Traversal(Medium)
class Solution(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if root==None:
            return None
        res=[]
        level=0
        queue=[root]
        while(queue!=[]):
            next_q=[]
            cur=[]
            for node in queue:
                cur.append(node.val)
                if level%2==1:
                    if node.left!=None:
                        next_q.append(node.left)
                    if node.right!=None:
                        next_q.append(node.right)
                else:
                    if node.right!=None:
                        next_q.append(node.right)
                    if node.left!=None:
                        next_q.append(node.left)
            next_q=next_q[::-1]
            res.append(cur)
            queue=next_q
            level+=1
        return res

#104. Maximum Depth of Binary Tree(Easy)
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root==None:
            return 0
        return max(self.maxDepth(root.left),self.maxDepth(root.right))+1

#105. Construct Binary Tree from Preorder and Inorder Traversal(Medium)
class Solution(object):
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        if preorder==[]:
            return None
        root=TreeNode(preorder[0])
        root_ind=inorder.index(preorder[0])
        root.left=self.buildTree(preorder[1:root_ind+1],inorder[:root_ind])
        root.right=self.buildTree(preorder[root_ind+1:],inorder[root_ind+1:])
        return root

#106. Construct Binary Tree from Inorder and Postorder Traversal(Medium)
class Solution(object):
    def buildTree(self, inorder, postorder):
        """
        :type inorder: List[int]
        :type postorder: List[int]
        :rtype: TreeNode
        """
        root=TreeNode(postorder[-1])
        root_ind=inorder.index(postorder[-1])
        root.left=self.buildTree(inorder[:root_ind],postorder[:root_ind])
        root.right=self.buildTree(inorder[root_ind+1:],postorder[root_ind:-1])
        return root

#107. Binary Tree Level Order Traversal II(Easy)
class Solution(object):
    def levelOrderBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if root==None:
            return []
        res=[]
        queue=[root]
        while(queue!=[]):
            next_q=[]
            cur=[]
            for node in queue:
                cur.append(node.val)
                if node.left!=None:
                    next_q.append(node.left)
                if node.right!=None:
                    next_q.append(node.right)
            res.append(cur)
            queue=next_q
        return res[::-1]


#108. Convert Sorted Array to Binary Search Tree(Easy)
class Solution(object):
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        if nums==[]:
            return None
        mid=len(nums)//2
        root=TreeNode(nums[mid])
        root.left=self.sortedArrayToBST(nums[:mid])
        root.right=self.sortedArrayToBST(nums[mid+1:])
        return root


#109. Convert Sorted List to Binary Search Tree(Medium)
class Solution(object):
    def sortedListToBST(self, head):
        """
        :type head: ListNode
        :rtype: TreeNode
        """
        if head==None:
            return None
        if head.next==None:
            return TreeNode(head.val)
        mid=TreeNode(-1)
        mid.next=head
        end=head
        while  end!=None and end.next!=None:
            mid=mid.next
            end=end.next.next
        temp=mid.next
        mid.next=None
        root=TreeNode(temp.val)
        root.left=self.sortedListToBST(head)
        root.right=self.sortedListToBST(temp.next)
        return root

#110. Balanced Binary Tree(Easy)
class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        self.ba=True
        self.height(root)
        return self.ba
    def height(self,root):
        if root==None:
            return 0
        left_h=self.height(root.left)
        right_h=self.height(root.right)
        if abs(left_h-right_h)>1:
            self.ba=False
        return max(left_h,right_h)+1
