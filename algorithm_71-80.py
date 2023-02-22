#71. Simplify Path(Medium)
class Solution(object):
    def simplifyPath(self, path):
        """
        :type path: str
        :rtype: str
        """
        roots=path.split("/")
        res=[]
        for i in range(len(roots)):
            if roots[i]=="":
                continue
            elif roots[i]==".":
                continue
            elif roots[i]=="..":
                res=res[:-1]
            else:
                res.append(roots[i])
        return "/"+"/".join(res)
#72. Edit Distance(Hard)
class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        m,n=len(word1),len(word2)
        # if m==0 or n==0:
        #     return m+n
        dp=[[0]*(n+1) for i in range(m+1)]
        print(dp)
        for i in range(m+1):
            for j in range(n+1):
                if i==0:
                    dp[i][j]=j
                elif j==0:
                    dp[i][j]=i
                elif word1[i-1]==word2[j-1]:
                    dp[i][j]=dp[i-1][j-1]
                elif word1[i-1]!=word2[j-1]:
                    dp[i][j]=min(dp[i-1][j-1],dp[i][j-1],dp[i-1][j])+1
        return dp[-1][-1]

#73. Set Matrix Zeroes(Medium)
class Solution(object):
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        m,n=len(matrix),len(matrix[0])
        rows,cols=set(),set()
        for i in range(m):
            for j in range(n):
                if matrix[i][j]==0:
                    rows.add(i)
                    cols.add(j)
                if (i in rows) or (j in cols):
                    matrix[i][j]=0
#74. Search a 2D Matrix
class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        m=len(matrix)
        if not m: return False
        n=len(matrix[0])
        if not n:return False
        lo,hi=0,m
        while lo<hi:
            mid=lo+(hi-lo)//2
            if matrix[mid][-1]<target:
                lo=mid+1
            elif matrix[mid][-1]>target:
                hi=mid
            else:
                return True
        aim=lo
        if aim>=m:return False
        if matrix[aim][0]>target:
            return False
        else:
            lo,hi=0,n
            
            while lo<hi:
                mid=lo+(hi-lo)//2
                if matrix[aim][mid]<target:
                    lo=mid+1
                elif matrix[aim][mid]>target:
                    hi=mid
                else:
                    return True
        return False

#75. Sort Colors(Medium)
class Solution(object):
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        red,white,blue=0,0,0
        for i in range(len(nums)):
            if nums[i]==0:
                red+=1
            elif nums[i]==1:
                white+=1
            elif nums[i]==2:
                blue+=1
        for i in range(len(nums)):
            if i<red:
                nums[i]=0
            elif i<red+white:
                nums[i]=1
            else:
                nums[i]=2
#76. Minimum Window Substring(Hard)    
class Solution(object):
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        if t=="":
            return ""
        min_len=0
        start,end,count=0,0,0
        m,n=len(s),len(t)
        dic,current={},{}
        expand=True
        res=""
        for c in t:
            if c in dic:
                dic[c]+=1
            else:
                dic[c]=1
        while not(end==m and count<n):
            # print(count,end,m,n)
            if expand:
                if s[end] in dic:
                    if s[end] in current:
                        current[s[end]]+=1
                    else:
                        current[s[end]]=1
                    if current[s[end]]<=dic[s[end]]:
                        count+=1
                if count>=n:
                    expand=False
                    if end-start+1<min_len or res=="":
                        min_len=end-start+1
                        res=s[start:end+1]
                if expand:
                    end+=1
            else:
                if s[start] in dic:
                    current[s[start]]-=1
                    if current[s[start]]<dic[s[start]]:
                        count-=1
                start+=1
                if count<n:
                    expand=True
                else:
                    if end-start+1<min_len:
                        min_len=end-start+1
                        res=s[start:end+1]
                if expand:
                    end+=1
        return res

#77. Combinations(Medium)
class Solution(object):
    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        #Recursive
        if k==0:
            return [[]]
        else:
            if n==k:
                return [[i for i in range(1,n+1)]]
            res=self.combine(n-1,k)
            next_comb=self.combine(n-1,k-1)
            for combines in next_comb:
                res.append(combines+[n])
            return res
    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        #DFS
        def dfs(i,current,res):
            if len(current)==k:
                res.append(current)
                return
            for m in range(i+1,n+1):
                dfs(m,current+[m],res)
        res=[]
        dfs(0,[],res)
        return res

#78. Subsets(Medium)
class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = [[]]
        for n in nums:
            nlev=[]
            for subset in res:
                new = subset+[n]
                nlev.append(new)
            res+=nlev
        return res

#79. Word Search(Medium)
class Solution(object):
    seen=set()
    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        rows,cols=len(board),len(board[0])
        for row in range(rows):
            for col in range(cols):
                self.seen=set()
                if self.existIn(board,word,row,col):
                    return True
        return False
    def isLegal(self,board,row,col):
        rows,cols=len(board),len(board[0])
        if row<0 or row>=rows:
            return False
        if col<0 or col>=cols:
            return False
        if (row,col) in self.seen:
            return False
        return True
    def existIn(self,board,word,row,col):
        if not self.isLegal(board,row,col):
            return False
        else:
            if board[row][col]==word:
                return True
            else:
                if board[row][col]!=word[0]:
                    return False
                else:
                    self.seen.add((row,col))
                    if self.existIn(board,word[1::],row,col+1) or self.existIn(board,word[1::],row,col-1) or self.existIn(board,word[1::],row-1,col) or self.existIn(board,word[1::],row+1,col):
                        return True
                    else:
                        self.seen.remove((row,col))
                        return False
solution=Solution()
board=[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]
#80. Remove Duplicates from Sorted Array II(Medium)
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        res=[]
        jump=False
        for i in range(len(nums)):
            if i==0:
                res.append(nums[i])
            else:
                if nums[i-1]==nums[i] and not jump:
                    res.append(nums[i])
                    jump=True
                if nums[i-1]!=nums[i]:
                    jump=False
                    res.append(nums[i])
        nums[0:len(res)]=res
        return len(res)
solution=Solution()
print(solution.removeDuplicates([1,1,1,1,2,2,3]))