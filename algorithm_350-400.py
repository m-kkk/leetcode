 #387. First Unique Character in a String
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
    
            
