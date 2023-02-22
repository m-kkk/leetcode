#81. Search in Rotated Sorted Array II(Medium)
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: bool
        """
        if nums==[]:
            return False
        else:
            if nums[0]==target:
                return True
            elif nums[0]>target:
                hi=len(nums)-1
                while hi>=0 and nums[hi]<=nums[0]:
                    if nums[hi]==target:
                        return True
                    hi-=1
                return False
            else:
                lo=0
                while lo<=len(nums)-1 and nums[lo]>=nums[0]:
                    if nums[lo]==target:
                        return True
                    lo+=1
                return False
# Remove Duplicates from Sorted List II(Meduim)
class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head==None or head.next==None:
            return head
        temp=ListNode(0)
        temp.next=head
        res=temp
        current=head
        while temp!=None and temp.next!=None:
            if current.val==current.next.val:
                while current.next!=None and current.val==current.next.val:
                    current=current.next
                temp.next=current.next
                current=temp.next
            else:
                temp=temp.next
                current=temp.next
        return res.next

#83. Remove Duplicates from Sorted List(Easy)
class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head==None:
            return head
        res=ListNode(0)
        res.next=head
        temp=head
        while head.next!=None:
            if head.val==head.next.val:
                while temp.next!=None and temp.val==temp.next.val:
                    temp=temp.next
                head.next=temp.next
            else:
                head=head.next
                temp=temp.next
        return res.next

#84. Largest Rectangle in Histogram(Hard)
class Solution(object):
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        stack=[]
        lo,max_are=0,0
        heights.append(0)
        while lo<len(heights):
            print(stack)
            if stack==[] or heights[lo]>=heights[stack[-1]]:
                stack.append(lo)
                lo+=1
            else:
                t=stack.pop()
                if stack==[]:
                    max_are=max(max_are,lo*heights[t])
                else:
                    max_are=max(max_are,(lo-stack[-1]-1)*heights[t])
        heights.pop()
        return max_are

solution=Solution()
print(solution.largestRectangleArea([2,5,3,15,2]))

#85.Maximal Rectangle(Hard)
class Solution(object):
    def maximalRectangle(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        max_rec=0
        heights=[0 for i in range(len(matrix[0]))]
        for lines in matrix:
            heights_n = [int(i) for i in lines]
            for i in range(len(heights_n)):
                if heights_n[i]==0:
                    heights[i]=0
                else:
                    heights[i]+=heights_n[i]
            max_rec=max(max_rec,largestRectangleArea(heights))
        return max_rec
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        stack=[]
        lo,max_are=0,0
        heights.append(0)
        while lo<len(heights):
            # print(stack)
            if stack==[] or heights[lo]>=heights[stack[-1]]:
                stack.append(lo)
                lo+=1
            else:
                t=stack.pop()
                if stack==[]:
                    max_are=max(max_are,lo*heights[t])
                else:
                    max_are=max(max_are,(lo-stack[-1]-1)*heights[t])
        heights.pop()
        return max_are

#86. Partition List(Medium)
class Solution(object):
    def partition(self, head, x):
        """
        :type head: ListNode
        :type x: int
        :rtype: ListNode
        """
        res=ListNode(-1)
        temp=ListNode(0)
        temp.next=head
        cur=res
        ne=temp
        while temp!=None:
            if temp.next!=None and temp.next.val<x:
                cur.next=temp.next
                temp.next=temp.next.next
                cur=cur.next
            else:
                temp=temp.next
        cur.next=ne.next
        return res

#87. Scramble String(Hard)
class Solution(object):
    # def isScramble(self, s1, s2):
    #     """
    #     :type s1: str
    #     :type s2: str
    #     :rtype: bool
    #     """
    #     if len(s1)!=len(s2): return False
    #     if s1==s2: return True
    #     l1=list(s1); l2=list(s2)
    #     l1.sort();l2.sort()
    #     if l1!=l2: return False
    #     l=len(s1)
    #     dp=[[[False for i in range(l+1)]for i in range(l)]for i in range(l)]
    #     for i in range(l):
    #         for j in range(l):
    #             dp[i][j][1]=(s1[i]==s2[j])
    #     for k in range(2,l+1):
    #         for i in range(l-k+1):
    #             for j in range(l-k+1):
    #                 for p in range(1,k):
    #                     dp[i][j][k]=(dp[i][j][k] or ((dp[i][j][p] and dp[i+p][j+p][k-p])or(dp[i][j+k-p][p] and dp[i+p][j][k-p])))
    #     return dp[0][0][l]

    def isScramble(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        if len(s1)!=len(s2): return False
        if s1==s2: return True
        l1=list(s1); l2=list(s2)
        l1.sort();l2.sort()
        if l1!=l2: return False
        for i in range(1,len(s1)):
            if self.isScramble(s1[:i],s2[:i]) and self.isScramble(s1[i:],s2[i:]):
                return True
            if self.isScramble(s1[:i],s2[len(s2)-i:]) and self.isScramble(s1[i:],s2[:len(s2)-i]):
                return True
        return False

solution=Solution()
# print(solution.isScramble("rgtae","great"))
#88. Merge Sorted Array(Easy)
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void Do not return anything, modify nums1 in-place instead.
        """
        mergelist=[]
        i,j=0,0
        while i<m and j<n:
            if nums1[i]<nums2[j]:
                mergelist.append(nums1[i])
                i+=1
            else:
                mergelist.append(nums2[j])
                j+=1
        mergelist+=nums1[i:m]
        mergelist+=nums2[j:n]
        for i in range(m+n):
            print(i)
            nums1[i]=mergelist[i]

#89. Gray Code(Medium)
class Solution(object):
    def grayCode(self, n):
        """
        :type n: int
        :rtype: List[int]
        """
        if n==0:
            return [0]
        else:
            res=[]
            prev=self.grayCode(n-1)
            for nums in prev:
                res=[nums+pow(2,n-1)]+res
            return prev+res
s=Solution()
print(s.grayCode(3))

#90 Subsets II(Medium)
class Solution(object):
    def subsetsWithDup(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums=sorted(nums)
        if nums==[]:
            return [[]]
        else:
            res=[]
            subsets=self.subsetsWithDup(nums[1:])
            res+=subsets
            for subs in subsets:
                if [nums[0]]+subs not in res:
                    res.append([nums[0]]+subs)
            return res

class Solution(object):
    def subsetsWithDup(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        ans = []
        nums.sort()
        def backTracking(nums, index, current, ans):
            ans.append(current)
            for i in range(index,len(nums)):
                target = current+[nums[i]]
                if i>index and nums[i-1]==nums[i]:
                    continue
                backTracking(nums,i+1,target,ans)
        backTracking(nums,0,[],ans)
        return ans

