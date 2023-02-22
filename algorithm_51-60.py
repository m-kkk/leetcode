#51. N-Queens(Hard)
class Solution(object):
    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        self.length=n
        self.res=[]
        self.solution=[-1]*n
        self.solve(0)
        return self.res
    def isLegal(self,row,col):
        for qcol in range(col):
            qrow = self.solution[qcol]
            if ((qrow == row) or
                (qrow+qcol == row+col) or
                (qrow-qcol == row-col)):
                return False
        return True
    def solve(self,col):
        if col == self.length:
            board =[['.' for i in range(self.length)]for i in range(self.length)]
            for i in range(self.length):
                board[i][self.solution[i]]='Q'
                board[i]="".join(board[i])
            self.res.append(board)
        else:
            for row in range(self.length):
                if self.isLegal(row,col):
                    self.solution[col]=row
                    self.solve(col+1)
                    self.solution[col]=-1
solution=Solution()
# print(solution.solveNQueens(4))

#52. N-Queens II(Hard)
class Solution(object):
    def totalNQueens(self, n):
        self.res=0
        self.length=n
        self.solution=[-1]*n
        self.solve(0)
        return self.res
    def isLegal(self,row,col):
        for qcol in range(col):
            qrow = self.solution[qcol]
            if ((qrow == row) or
                (qrow+qcol == row+col) or
                (qrow-qcol == row-col)):
                return False
        return True
    def solve(self,col):
        if col == self.length:
            self.res+=1
        else:
            for row in range(self.length):
                if self.isLegal(row,col):
                    self.solution[col]=row
                    self.solve(col+1)
                    self.solution[col]=-1

#53. Maximum Subarray(Medium)
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        max_sum=None
        cur_sum=0
        for i in range(len(nums)):
            if cur_sum<0:
                cur_sum =0
            cur_sum+=nums[i]
            if max_sum==None:
                max_sum=cur_sum
            else:
                max_sum=max(max_sum,cur_sum)
        return max_sum

#54. Spiral Matrix(Medium)
class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        if matrix==[]:
            return []
        self.seen=[]
        res=[]
        dirs = [(0,1),(1,0),(0,-1),(-1,0)]
        run,self.rows,self.cols=0,len(matrix),len(matrix[0])
        row,col=0,0
        while(len(res)<self.rows*self.cols):
            if self.isLegal(row,col):
                self.seen.append((row,col))
                res.append(matrix[row][col])
            else:
                row-=dirs[run%4][0]
                col-=dirs[run%4][1]
                run+=1
            row+=dirs[run%4][0]
            col+=dirs[run%4][1]
        return res
    def isLegal(self,row,col):
        if row<0 or row>=self.rows:
            return False
        if col<0 or col>=self.cols:
            return False
        if (row,col) in self.seen:
            return False
        return True
solution=Solution()

#55. Jump Game(Meduim)
class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        if(len(nums)==1):
            return True
        max_jump=0
        lo=0
        while(lo<=max_jump and lo<len(nums)):
            max_jump=max(nums[lo]+lo,max_jump)
            lo+=1
        print(max_jump)
        if max_jump<len(nums)-1:
            return False
        return True

#56. Merge Intervals(Medium)
class Interval(object):
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e
class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        intervals=sorted(intervals,key= lambda x:x.start)
        for i in range(1,len(intervals)):
            if intervals[i].start<=intervals[i-1].end:
                intervals[i].start=intervals[i-1].start
                intervals[i].end=max(intervals[i].end,intervals[i-1].end)
                intervals[i-1]=None
        res=[]
        for i in range(len(intervals)):
            if intervals[i]!=None:
                res.append(intervals[i])
        return res

#57. Insert Interval(Hard)
class Solution(object):
    def insert(self, intervals, newInterval):
        """
        :type intervals: List[Interval]
        :type newInterval: Interval
        :rtype: List[Interval]
        """
        res=[]
        intervals.append(newInterval)
        intervals=sorted(intervals,key= lambda x:x.start)
        for i in range(len(intervals)):
            if res==[] or intervals[i].start>res[-1].end:
                res.append(intervals[i])
            else:
                res[-1].end=max(res[-1].end,intervals[i].end)
        return res
        
#58. Length of Last Word(Easy)
class Solution(object):
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        s=s.strip()
        s=s.split(" ")
        return len(s[-1])

#59. Spiral Matrix II(Medium)
class Solution(object):
    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        res=[[0]*n for i in range(n)]
        dirs = [(0,1),(1,0),(0,-1),(-1,0)]
        for i in range(n):
            res[0][i]=i+1
        row,col,target=0,n-1,n+1
        n-=1
        run=1
        while(n>0):
            for j in range(2):
                for i in range(n):
                    row+=dirs[run%4][0]
                    col+=dirs[run%4][1]
                    res[row][col]=target
                    target+=1
                run+=1
            n-=1
        return res
solution=Solution()
#60. Permutation Sequence(Medium)
class Solution(object):
    def getPermutation(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: str
        """
        s=[str(i) for i in range(1,n+1)]
        res=[]
        m=n
        k=k-1
        while(len(res)<m):
            i=k//self.factorial(n-1)
            k=k%self.factorial(n-1)
            res.append(s[i])
            s.remove(s[i])
            n=n-1
        return "".join(res)
    def factorial(self,n):
        if n == 0:
            return 1
        else:
            return n*self.factorial(n-1)
solution=Solution()
# print(solution.factorial(5))
print(solution.getPermutation(4,3))

