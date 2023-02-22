#31. Next Permutation(Medium)
class Solution(object):
    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        hi=len(nums)-1
        while(hi!=0):
            if nums[hi]>nums[hi-1]:
                break
            hi-=1
        if(hi==0):
            nums[:] = nums[::-1]
            return
        piv=nums[hi-1]
        target=len(nums)-1
        while(nums[target]<=piv):
            target-=1
        nums[hi-1],nums[target]=nums[target],nums[hi-1]
        nums[hi:] = nums[hi:][::-1]

solution=Solution()
# sample = [0,1,2,5,3,3,0]
sample=[3,2,1,0]
solution.nextPermutation(sample)

#32 Longest Valid Parentheses(Hard)
class Solution(object):
    #Stack
    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        if s=="":
            return 0
        valid=None
        max_length=0
        len_count=0
        pair_length=0
        stack=[]
        for i in range(len(s)):
            if s[i]=="(":
                stack.append(i)
            else:
                if stack==[]:
                    len_count=0
                else:
                    match=stack[-1]
                    stack=stack[0:len(stack)-1]
                    if stack==[]:
                        pair_length=i-match+1
                        len_count+=pair_length
                        pair_length=len_count
                    else:
                        pair_length=i-stack[-1]
                    max_length=max(max_length,pair_length)
        return max_length

    #Dynamic Programming
    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        if s=="":
            return 0
        dp=[0]*len(s)
        res=0
        for i in reversed(range(len(s)-1)):
            if s[i]=="(":
                j=i+1+dp[i+1]
                if j<len(s) and s[j]==")":
                    dp[i]=2+dp[i+1]
                    if j+1<len(s):
                        dp[i]+=dp[j+1]
                else:
                    dp[i]=0
            else:
                dp[i]=0
            res=max(res,dp[i])
        return res
solution=Solution()
print(solution.longestValidParentheses("()())(()()()())"))
#33. Search in Rotated Sorted Array(Hard)
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        if nums==[]:
            return -1
        if target>=nums[0]:
            lo=0
            while(lo<len(nums) and nums[lo]>=nums[0]):
                if(nums[lo]==target):
                    return lo
                lo+=1
            return -1
        else:
            hi=len(nums)-1
            while(hi>=0 and nums[hi]<=nums[0]):
                print(hi,nums[hi])
                if(nums[hi]==target):
                    return hi
                hi-=1
            return -1

#34. Search for a Range(Medium)
class Solution(object):
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        find=False
        lo,hi=0,len(nums)
        res=[]
        while(lo<hi):
            mid=lo+(hi-lo)//2
            if(nums[mid]==target):
                find=True
                if(mid==0 or nums[mid-1]<target):
                    res.append(mid)
                    break
                else:
                    hi=mid
            elif(nums[mid]<target):
                lo=mid+1
            elif(nums[mid]>target):
                hi=mid
        print(lo,hi)
        if(find==False):
            return [-1,-1]
        lo,hi=0,len(nums)
        while(lo<hi):
            mid=lo+(hi-lo)//2
            if(nums[mid]==target):
                find=True
                if(mid==len(nums)-1 or nums[mid+1]>target):
                    res.append(mid)
                    break
                else:
                    lo=mid+1
            elif(nums[mid]<target):
                lo=mid+1
            elif(nums[mid]>target):
                hi=mid

        return res

solution=Solution()
# print(solution.searchRange([1,1,1,2,2,2,2,4,5,6,7],3))

#35. Search Insert Position(Medium)
class Solution(object):
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        lo,hi=0,len(nums)
        while(lo<hi):
            mid=lo+(hi-lo)//2
            if nums[mid]==target:
                return mid
            elif(nums[mid]>target):
                hi=mid
            elif(nums[mid]<target):
                lo=mid+1
        return lo
#36. Valid Sudoku
class Solution(object):
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        rows=[set() for i in range(9)]
        cols=[set() for i in range(9)]
        grids=[set() for i in range(9)]
        for r in range(9):
            for c in range(9):
                if(board[r][c] == "."):
                    continue
                if board[r][c] in rows[r]:
                    return False
                if board[r][c] in cols[c]:
                    return False
                g = r//3*3+c//3
                if board[r][c] in grids[g]:
                    return False
                rows[r].add(board[r][c])
                cols[c].add(board[r][c])
                grids[g].add(board[r][c])
        return True
board=[".87654321","2........","3........","4........","5........","6........","7........","8........","9........"]
solution=Solution()
# print(solution.isValidSudoku(board))

#37. Sudoku Solver(Hard)
class Solution(object):
    def isValid(self, board, x, y):
        g = x//3*3+y//3
        for c in range(9):
            if(board[x][c]==board[x][y] and c!=y):
                return False
        for r in range(9):
            if(board[r][y]==board[x][y] and r!=x):
                return False
        for i in range(9):
            row=g//3*3+i//3
            col=g%3*3+i%3
            if(board[x][y]==board[row][col] and row!=x and col!=y):
                return False
        return True
    def solveSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        self.solve(board,0,0)
    def solve(self,board,x,y):
        if y==9:
            y=0
            x+=1
        if x>=9:
            return True
        if(board[x][y]!="."):
            return self.solve(board,x,y+1)
        for i in range(9):
            board[x][y]=str(i)
            if self.isValid(board,x,y) and self.solve(board,x,y+1):
                return True
            else:
                board[x][y]="."
        return False

board=["..9748...","7........",".2.1.9...","..7...24.",".64.1.59.",".98...3..","...8.3.2.","........6","...2759.."]
board_l= [list(board[i]) for i in range(9)]
# board_l.append(0)
#38. Count and Say(Easy)
class Solution(object):
    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        if n==1:
            return "1"
        res="1"
        count=1
        while n>1:
            temp=""
            for i in range(len(res)):
                if(i!=0 and res[i]==res[i-1]):
                    count+=1
                if(i!=0 and res[i]!=res[i-1]):
                    temp+=str(count)+res[i-1]
                    count=1
                if(i==len(res)-1):
                    temp+=str(count)+res[i]
            n-=1
            count=1
            res=temp
        return res
solution=Solution()
# print(solution.countAndSay(5))

#39. Combination Sum(Medium)
class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        return self.serch(candidates,target,0,[])

    def serch(self,candidates,target,start,current):
        res=[]
        init_len = len(current)
        for i in range(start,len(candidates)):
            if candidates[i] == target:
                current=current+[candidates[i]]
                res.append(current)
            elif candidates[i]<target:
                current=current+[candidates[i]]
                next_res=self.serch(candidates,target-candidates[i],i,current)
                if(next_res!=[]):
                    res+=next_res
            current=current[0:init_len]
        return res

    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        def dfs(current,nums,k,res,target):
            if sum(current)==target:
                res.append(current)
                return
            if sum(current)>target:
                return
            for i in range(k,len(nums)):
                dfs(current+[nums[i]],nums,i,res,target)
        res = []
        dfs([],candidates,0,res,target)
        return res

#40. Combination Sum II(Medium)
class Solution(object):
    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        return self.serch(sorted(candidates),target,0,[])

    def serch(self,candidates,target,start,current):
        res=[]
        init_len = len(current)
        for i in range(start,len(candidates)):
            if candidates[i] == target:
                current=current+[candidates[i]]
                if(current not in res):
                    res.append(current)
            elif candidates[i]<target:
                current=current+[candidates[i]]
                next_res=self.serch(candidates,target-candidates[i],i+1,current)
                if(next_res!=[]):
                    for element in next_res:
                        if element not in res:
                            res.append(element)
            current=current[0:init_len]
        return res

    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        nums = sorted(candidates)
        def dfs(current,nums,target,res):
            if target==0:
                res.append(current)
                return
            if target<0:
                return 
            for i in range(len(nums)):
                if nums[i] > target:
                    return
                if i!=0 and nums[i-1]==nums[i]:
                    continue
                dfs(current+[nums[i]],nums[i+1:],target-nums[i],res)
        res = []
        dfs([],nums,target,res)
        return res
