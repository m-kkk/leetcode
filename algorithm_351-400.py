#352. Data Stream as Disjoint Intervals(Hard)
# Definition for an interval.
class Interval:
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e
    def __repr__(self):
        return "[%d,%d]"%(self.start,self.end)


class SummaryRanges:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.data = []

    def b_search_left(self,val):
        left,right = 0,len(self.data)
        while left<right:
            mid = (left+right)//2
            if self.data[mid].start < val:
                left = mid+1
            else:
                right = mid
        return left

    def addNum(self, val):
        """
        :type val: int
        :rtype: void
        """
        item = Interval(val,val)
        ind = self.b_search_left(val)
        self.data = self.data[0:ind]+[item]+self.data[ind:]
        self.merge(ind)

        
    def merge(self, i):
        if len(self.data)<=1:
            return
        if i+1<len(self.data) and (self.data[i].end + 1 >= self.data[i+1].start): 
            self.data[i+1].start = min(self.data[i].start,self.data[i+1].start)
            self.data = self.data[:i]+self.data[i+1:]
        if i-1>=0 and (self.data[i].start - 1 <= self.data[i-1].end):
            self.data[i-1].end = max(self.data[i].end,self.data[i-1].end)
            self.data = self.data[:i]+self.data[i+1:]

    def getIntervals(self):
        """
        :rtype: List[Interval]
        """
        return self.data

c = SummaryRanges()
for i in [6,6,0,4,8,7,6,4,7,5]:
    c.addNum(i)
print(c.getIntervals())

#377. Combination Sum IV(Medium)
class Solution:
    def combinationSum4(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        nums.sort()
        dp = [0]*(target+1)
        #dp[i] represent how many combinations picked from nums to sum to i
        dp[0] = 1
        # 1 way to sum=0 ,pick nothing
        for i in range(1,target+1):
            for n in nums:
                if i<n:
                    break
                dp[i]+=dp[i-n]
        return dp[-1]

#387. First Unique Character in a String(Easy)
class Solution:
    def firstUniqChar(self, s):
        """
        :type s: str
        :rtype: int
        """
        t = {}
        for c in s:
            if c in t:
                t[c]+=1
            else:
                t[c]=1
        for i,c in enumerate(s):
            if t[c] == 1:
                return i
        return -1

#392. Is Subsequence(Medium)
class Solution:
    def isSubsequence(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        p=0
        if not s:
            return True
        for c in t:
            if c==s[p]:
                p+=1
            if p==len(s):
                return True
        return False


#395.Maximum Product of Word Lengths(Meduim)
class Solution:
    def longestSubstring(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        """
        if not s:return 0
        cal = self.calc_repeat(s)
        l = 0
        breaks = [-1]
        for i,c in enumerate(s):
            if cal[c]<k:
                breaks.append(i)
        breaks.append(len(s))
        if len(breaks)==2:return len(s)
        for i in range(1,len(breaks)):
            l = max(l,self.longestSubstring(s[breaks[i-1]+1:breaks[i]],k))
        return l
        
    def calc_repeat(self,s):
        res={}
        for c in s:
            if c not in res:
                res[c]=0
            res[c]+=1
        return res
    
            
