#265. Paint House II(Hard)
class Solution:
    def minCostII(self, costs):
        """
        :type costs: List[List[int]]
        :rtype: int
        """
        if not costs:return 0
        first,second = float('inf'),float('inf')
        for j in range(len(costs[0])):
            if costs[0][j] < first:
                first,second = costs[0][j],first
            elif costs[0][j] < second:
                second = costs[0][j]
        for i in range(1,len(costs)):
            left_first,left_second=float('inf'),float('inf')
            for j in range(len(costs[i])):
                if costs[i-1][j] == first:
                    costs[i][j] += first
                else:
                    costs[i][j] += second
                if costs[i][j] < left_first:
                    left_first,left_second = costs[i][j],left_first
                elif costs[i][j] < left_second:
                    left_second = costs[i][j]
            first,second = left_first,left_second
        return first







#289. Game of Life(M)
class Solution:
    def gameOfLife(self, board):
        """
        :type board: List[List[int]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        def valid(board,i,j):
            if i<0 or i>=len(board):
                return False
            if j<0 or j>=len(board[0]):
                return False
            return True
        def check(board,x,y):
            res = []
            for i in range(-1,2):
                for j in range(-1,2):
                    if i==0 and j==0:continue
                    if valid(board,x+i,y+j):
                        res.append(board[x+i][y+j])
            return res
        n,m = len(board),len(board[0])
        lives,dead = [],[]
        for i in range(n):
            for j in range(m):
                neighbors = check(board,i,j)
                if board[i][j]==0:
                    if neighbors.count(1)==3:
                        lives.append((i,j))
                    else:
                        dead.append((i,j))
                if board[i][j]==1:
                    if neighbors.count(1)<2:
                        dead.append((i,j))
                    elif neighbors.count(1)>3:
                        dead.append((i,j))
                    else:
                        lives.append((i,j))
        for points in lives:
            board[points[0]][points[1]]=1
        for points in dead:
            board[points[0]][points[1]]=0  



#300.Longest Increasing Subsequence(M)
class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:return 0
        dp = [0]*len(nums)
        dp[0]=1
        for i in range(1,len(nums)):
            max_length=0
            for j in range(i):
                if nums[i]>nums[j]:
                    max_length = max(max_length,dp[j]+1)
            dp[i] = max(max_length,1)
        # print(dp)
        return max(dp)

class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        def search(LIS,num):
        	left,right=0,len(LIS)-1
        	if right<0:return -1
        	while left<right:
        		mid = (left+right)//2
        		if LIS[mid]<num:
        			left=mid+1
        		else:
        			right=mid
        	if right == len(LIS)-1 and LIS[right]<num:
        		return -1
        	return right
        LIS = []
        for n in nums:
        	pos = search(LIS,n)
        	if pos == -1:
        		LIS.append(n)
        	else:
        		LIS[pos]=n
        return len(LIS)
s = Solution()
print(s.lengthOfLIS([100,1,2,98,3,5,-1,11]))

#307. Range Sum Query - Mutable(Medium)
class NumArray:

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        if not nums:
            self.seg_tree=[0]
            self.create_tree(0,0,0,[0])
            return
        self.nums = nums
        n = len(nums)
        height = math.ceil(math.log(n,2))+1
        self.seg_tree = [0]*int(2**height-1)
        self.create_tree(0,n-1,0,nums)
        
    def create_tree(self,start,end,i,nums):
        if start == end :
            self.seg_tree[i] = nums[start]
            return nums[start]
        mid = (start+end)//2
        self.seg_tree[i] = self.create_tree(start,mid,i*2+1,nums)+self.create_tree(mid+1,end,i*2+2,nums)
        return self.seg_tree[i]

    def update(self, i, val):
        """
        :type i: int
        :type val: int
        :rtype: void
        """
        dif = val-self.nums[i]
        self.update_helper(0,len(self.nums)-1,i,0,dif)
    
    def update_helper(self,start,end,i,tree_index,dif):
        if i < start or i > end:
            return
        self.seg_tree[tree_index] = self.seg_tree[tree_index] + dif
        if start == end:
            if start == i:
                self.nums[i]+=dif
            return
        mid = (start+end)//2
        self.update_helper(start,mid,i,tree_index*2+1,dif)
        self.update_helper(mid+1,end,i,tree_index*2+2,dif)

    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        return self.getsum(0,len(self.nums)-1,i,j,0)
    
    def getsum(self,start,end,i,j,tree_index):
        if start>j or end<i:
            return 0
        if start>=i and end<=j:
            return self.seg_tree[tree_index]
        mid = (start+end)//2
        return self.getsum(start,mid,i,j,tree_index*2+1)+self.getsum(mid+1,end,i,j,tree_index*2+2)

# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# obj.update(i,val)
# param_2 = obj.sumRange(i,j)

#325. Maximum Size Subarray Sum Equals k(Medium)
class Solution:
    def maxSubArrayLen(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        hsh={}
        c_sum,res=0,0
        for i,n in enumerate(nums):
            c_sum+=n
            if c_sum==k:
                res=max(res,i+1)
            c = hsh.get(c_sum-k,None)
            if c != None:
                res=max(res,i-c)
            if c_sum not in hsh:
                hsh[c_sum] = i
        return res


#328.Odd Even Linked List(M)
class Solution(object):
    def oddEvenList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next:return head
        odd = head
        even = head.next
        head2 = even
        while odd.next or even.next:
            if odd and odd.next:
                odd.next = odd.next.next
                if odd.next:
                    odd = odd.next
            if even and even.next:
                even.next = even.next.next
                if even.next:
                    even = even.next
        odd.next = head2
        return head

class Solution(object):
    def oddEvenList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next:return head
        odd = head
        even = head.next
        while even and even.next:
            p = even.next
            even.next = p.next
            p.next = odd.next
            odd.next = p
            odd = odd.next
            even = even.next
        return head

#340. Longest Substring with At Most K Distinct Characters
class Solution:
    def lengthOfLongestSubstringKDistinct(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        """
        mapping = {}
        left = 0
        res = 0
        for i,c in enumerate(s):
            if c in mapping:
                mapping[c]+=1
            else:
                mapping[c]=1
            while len(mapping)>k:
                mapping[s[left]]-=1
                if mapping[s[left]]==0:
                    del mapping[s[left]]
                left+=1
            res = max(res,i-left+1)
        return res

#344. Reverse String
class Solution:
    def reverseString(self, s):
        """
        :type s: str
        :rtype: str
        """
        return s[::-1]