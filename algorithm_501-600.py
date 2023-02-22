# 518. Coin Change 2(Medium)
class Solution(object):
    def change(self, amount, coins):
        """
        :type amount: int
        :type coins: List[int]
        :rtype: int
        """
        if not coins:
            return 0 if amount else 1
        dp = [[0]*(amount+1) for _ in range(len(coins))]
        for i in range(len(coins)):
            dp[i][0] = 1
            for j in range(1,amount+1):
                dp[i][j] = dp[i-1][j] + dp[i][j-coins[i]] if j-coins[i]>=0 else dp[i-1][j]
        
        return dp[-1][-1]

    #Optimize space usage to 1-D dp
    def change(self, amount, coins):
        """
        :type amount: int
        :type coins: List[int]
        :rtype: int
        """
        dp = [0]*(amount+1)
        dp[0]=1
        for coin in coins:
            for i in range(coin,amount+1):
                dp[i] += dp[i-coin]
        return dp[-1]



#525. Contiguous Array(Medium)
class Solution:
    def findMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        res,c_sum = 0,0
        hsh={}
        for i,n in enumerate(nums):
            c_sum = c_sum+1 if n else c_sum-1
            c = hsh.get(c_sum,None)
            if c_sum not in hsh:
                hsh[c_sum] = i
            if not c_sum:
                res=i+1
            if c!=None:
                res=max(res,i-c)
        return res


#560. Subarray Sum Equals K(Medium)
class Solution:
    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        count,c_sum=0,0
        hsh = {}
        for i in range(len(nums)):
            c_sum+=nums[i]
            if c_sum==k:
                count+=1
            count+=hsh.get(c_sum-k,0)
            if c_sum in hsh:
                hsh[c_sum]+=1
            else:
                hsh[c_sum]=1
        return count

#572. Subtree of Another Tree(Easy)
class Solution:
    def isSubtree(self, s, t):
        """
        :type s: TreeNode
        :type t: TreeNode
        :rtype: bool
        """
        def same_tree(s,t):
            if s is None and t is None:
                return True
            elif s and t:
                return same_tree(s.left,t.left) and same_tree(s.right,t.right) and (s.val==t.val)
            else:
                return False
        if not s:
            return False
        if s.val==t.val:
            if same_tree(s,t):
                return True
        if self.isSubtree(s.left,t):
            return True
        if self.isSubtree(s.right,t):
            return True
        return False

#589. N-ary Tree Preorder Traversal(Easy)

# Definition for a Node.
class Node(object):
    def __init__(self, val, children):
        self.val = val
        self.children = children

class Solution(object):
    def preorder(self, root):
        """
        :type root: Node
        :rtype: List[int]
        iteratively
        """
        res = []
        stack = [root]
        if not root:
            return res
        while stack:
            node = stack.pop()
            res.append(node.val)
            while node.children:
                stack.append(node.children.pop())
        return res