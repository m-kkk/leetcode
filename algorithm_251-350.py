#252. Meeting Rooms(Easy)
# Definition for an interval.
class Interval:
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e

class Solution:
    def canAttendMeetings(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: bool
        """
        intervals = sorted(intervals,key=lambda x:x.start)
        for i in range(1,len(intervals)):
            if intervals[i].start < intervals[i-1].end:
                return False
        return True


#253. Meeting Rooms II(Medium)
import heapq
class Solution:
    def minMeetingRooms(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: int
        """
        if not intervals:return 0
        intervals.sort(key=lambda x:x.start)
        heap = []
        res = 0
        for meeting in intervals:
            if heap and meeting.start>=heap[0]:
                heapq.heappop(heap)
            heapq.heappush(heap,meeting.end)
        return len(heap)

#254. Factor Combinations(Medium)
class Solution:
    def getFactors(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        def help(n,k):
            res = []
            for i in range(k,int(n**0.5)+1):
                if n%i == 0:
                    res.append([i,n//i])
                    for item in help(n//i,i):
                        res.append([i]+item)
            return res
        return help(n,2)

solution = Solution()
print(solution.getFactors(16))

#257. Binary Tree Paths(Easy)
class Solution:
    def binaryTreePaths(self, root):
        """
        :type root: TreeNode
        :rtype: List[str]
        """
        if not root:
            return []
        def helper(current,node,res):
            if node.left:
                helper(current+str(node.val)+"->",node.left,res)
            if node.right:
                helper(current+str(node.val)+"->",node.right,res)
            if not node.left and not node.right:
                res.append(current+str(node.val))
        res = []
        helper("",root,res)
        return res


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

#268. Missing Number(Easy)
class Solution(object):
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        res = len(nums)
        for i in range(len(nums)):
            res = res^i^nums[i]
        return res

#273. Integer to English Words(Hard)
"""This is ridiculous！！
No wonder why American kid can't do arithmetic 
"""
class Solution:
    def numberToWords(self, num):
        """
        :type num: int
        :rtype: str
        """
        def one(num):
            switcher = {
                1: 'One',
                2: 'Two',
                3: 'Three',
                4: 'Four',
                5: 'Five',
                6: 'Six',
                7: 'Seven',
                8: 'Eight',
                9: 'Nine'
            }
            return switcher.get(num)

        def two_less_20(num):
            switcher = {
                10: 'Ten',
                11: 'Eleven',
                12: 'Twelve',
                13: 'Thirteen',
                14: 'Fourteen',
                15: 'Fifteen',
                16: 'Sixteen',
                17: 'Seventeen',
                18: 'Eighteen',
                19: 'Nineteen'
            }
            return switcher.get(num)
        
        def ten(num):
            switcher = {
                2: 'Twenty',
                3: 'Thirty',
                4: 'Forty',
                5: 'Fifty',
                6: 'Sixty',
                7: 'Seventy',
                8: 'Eighty',
                9: 'Ninety'
            }
            return switcher.get(num)
        

        def two(num):
            if not num:
                return ''
            elif num < 10:
                return one(num)
            elif num < 20:
                return two_less_20(num)
            else:
                tenner = num // 10
                rest = num - tenner * 10
                return ten(tenner) + ' ' + one(rest) if rest else ten(tenner)
        
        def three(num):
            hundred = num // 100
            rest = num - hundred * 100
            if hundred and rest:
                return one(hundred) + ' Hundred ' + two(rest) 
            elif not hundred and rest: 
                return two(rest)
            elif hundred and not rest:
                return one(hundred) + ' Hundred'
        
        billion = num // 1000000000
        million = (num - billion * 1000000000) // 1000000
        thousand = (num - billion * 1000000000 - million * 1000000) // 1000
        rest = num - billion * 1000000000 - million * 1000000 - thousand * 1000
        
        if not num:
            return 'Zero'
        
        result = ''
        if billion:        
            result = three(billion) + ' Billion'
        if million:
            result += ' ' if result else ''    
            result += three(million) + ' Million'
        if thousand:
            result += ' ' if result else ''
            result += three(thousand) + ' Thousand'
        if rest:
            result += ' ' if result else ''
            result += three(rest)
        return result


#287. Find the Duplicate Number(Medium)
class Solution:
    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums)<2:
            return -1
        slow=nums[0]
        fast=nums[slow]
        while (slow != fast):
            slow = nums[slow]
            fast = nums[nums[fast]]
        fast = 0
        while slow!=fast :
            slow = nums[slow]
            fast = nums[fast]
        return slow
solution = Solution()
print("-------------")
print(solution.findDuplicate([1,2,3,1]))

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


# 295. Find Median from Data Stream(Hard)
class MedianFinder(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.hi = []
        self.lo = []
        

    def addNum(self, num):
        """
        :type num: int
        :rtype: void
        """
        heapq.heappush(self.lo,num)
        heapq.heappush(self.hi,-heapq.heappop(self.lo))
        if len(self.hi)>len(self.lo):
            heapq.heappush(self.lo,-heapq.heappop(self.hi))
        # print(self.lo,self.hi)

    def findMedian(self):
        """
        :rtype: float
        """
        if len(self.lo)>len(self.hi):
            return self.lo[0]
        else:
            return (self.lo[0]-self.hi[0])/2


# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()


# 297. Serialize and Deserialize Binary Tree(Hard)
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        res = []
        def helper(root,res):
            if not root:
                res.append('None')
                return
            res.append(str(root.val))
            helper(root.left,res)
            helper(root.right,res)
        helper(root,res)
        return ",".join(res)
            
    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        s = data.split(',')
        def helper(s):
            if s[0]=='None':
                s.pop(0)
                return None
            root = TreeNode(int(s[0]))
            s.pop(0)
            root.left=helper(s)
            root.right=helper(s)
            return root
        return helper(s)

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
# print(s.lengthOfLIS([100,1,2,98,3,5,-1,11]))

#303. Range Sum Query - Immutable(Easy)
class NumArray:

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        if not nums:
            self.cache = []
            return
        self.cache = [0]*(len(nums)+1)
        for i in range(1,len(nums)+1):
            self.cache[i] = self.cache[i-1]+nums[i-1]
        

    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        if not self.cache:
            return 0
        return self.cache[j+1]-self.cache[i]


#304. Range Sum Query 2D - Immutable(Medium)
class NumMatrix:

    def __init__(self, matrix):
        """
        :type matrix: List[List[int]]
        """
        if not matrix or not matrix[0]:
            self.cache= None
            return
        self.cache = [[0]*(len(matrix[0])+1) for i in range(len(matrix)+1)]
        for i in range(1,len(matrix)+1):
            self.cache[i][1] = self.cache[i-1][1] + matrix[i-1][0]
        for j in range(2,len(matrix[0])+1):
            self.cache[1][j] = self.cache[1][j-1] + matrix[0][j-1]
        for i in range(1,len(matrix)+1):
            for j in range(1,len(matrix[0])+1):
                self.cache[i][j] = self.cache[i-1][j] + self.cache[i][j-1] - self.cache[i-1][j-1] + matrix[i-1][j-1]
    def sumRegion(self, row1, col1, row2, col2):
        """
        :type row1: int
        :type col1: int
        :type row2: int
        :type col2: int
        :rtype: int
        """
        if not self.cache:
            return 0
        return self.cache[row2+1][col2+1] - self.cache[row2+1][col1] - self.cache[row1][col2+1] + self.cache[row1][col1]

n = NumMatrix([[3,0,1,4,2],
               [5,6,3,2,1],
               [1,2,0,1,5],
               [4,1,0,1,7],
               [1,0,3,0,5]])

print(n.sumRegion(*[2,1,4,3]))
print(n.sumRegion(*[1,1,2,2]))
print(n.sumRegion(*[1,2,2,4]))



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

#314. Binary Tree Vertical Order Traversal(Medium)
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def verticalOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        import collections
        nodes = collections.defaultdict(list)
        if not root:
            return []
        queue = [(root,0)]
        neg,pos=0,0
        while queue:
            node,i = queue.pop(0)
            if i<neg:
                neg=i
            if i>pos:
                pos=i
            nodes[i].append(node.val)
            if node.left:
                queue.append((node.left,i-1))
            if node.right:
                queue.append((node.right,i+1))
            
        return [nodes[i] for i in range(neg,pos+1)]

#316. Remove Duplicate Letters(Hard)
class Solution:
    def removeDuplicateLetters(self, s):
        """
        :type s: str
        :rtype: str
        """
        last_positions = {c:i for i,c in enumerate(s)}
        stack = []
        seen = set()
        for i,c in enumerate(s):
            if c not in seen:
                while stack and c<stack[-1] and last_positions[stack[-1]]>i:
                    seen.remove(stack[-1])
                    stack.pop()
                stack.append(c)
                seen.add(c)
        return "".join(stack)


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


#328.Odd Even Linked List(Medium)
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

# 338. Counting Bits(Medium)
class Solution(object):
    def countBits(self, num):
        """
        :type num: int
        :rtype: List[int]
        """
        res = [0] * (num+1)
        for i in range(1,num+1):
            res[i] = res[(i&(i-1))] + 1# or i-i&(-i)
        return res


#340. Longest Substring with At Most K Distinct Characters(Hard)
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
