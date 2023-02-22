#111. Minimum Depth of Binary Tree(Easy)
class Solution(object):
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root==None:
            return 0
        left_h=self.minDepth(root.left)
        right_h=self.minDepth(root.right)
        if left_h*right_h==0:
            return left_h+right_h+1
        else:
            return min(left_h,right_h)+1

#112. Path Sum(Easy)
class Solution(object):
    def hasPathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: bool
        """
        if root==None:
            return False
        if root.left==None and root.right==None:
            return sum==root.val
        return self.hasPathSum(root.left,sum-root.val) or self.hasPathSum(root.right,sum-root.val)

#113. Path Sum II(Medium)
class Solution(object):
    def pathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: List[List[int]]
        """
        if root==None:
            return []
        self.res=[]
        self.dfs(root,sum,[])
        return self.res

    def dfs(self,root,sum,path):
        if root==None:
            return None
        if root.val==sum and root.left==None and root.right==None:
            path.append(root.val)
            res=[path[i] for i in range(len(path))]
            self.res.append(res)
            path.pop()
        else:
            path.append(root.val)
            self.dfs(root.left,sum-root.val,path)
            self.dfs(root.right,sum-root.val,path)
            path.pop()

#114. Flatten Binary Tree to Linked List(Medium)
class Solution(object):
    def flatten(self, root):
        """
        :type root: TreeNode
        :rtype: void Do not return anything, modify root in-place instead.
        """
        if root==None:
            return
        self.flatten(root.left)
        self.flatten(root.right)
        if root.left==None:
            return
        p=root.left
        while p.right!=None:
            p=p.right
        p.right=root.right
        root.right=root.left
        root.left=None
        
#115. Distinct Subsequences(Hard)
class Solution(object):
    def numDistinct(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: int
        """
        m,n=len(s),len(t)
        dp=[[0]*(n+1) for i in range(m+1)]
        dp[0][0]=1
        for i in range(m):
            dp[i+1][0]=1
        for j in range(n):
            dp[0][j+1]=0
        for i in range(m):
            for j in range(n):
                if i<j:
                    dp[i+1][j+1]=0
                dp[i+1][j+1]=dp[i][j+1]
                if s[i]==t[j]:
                    dp[i+1][j+1]+=dp[i][j]
        return dp[-1][-1]

#116. Populating Next Right Pointers in Each Node(Medium)
class Solution:
    # @param root, a tree link node
    # @return nothing
    def connect(self, root):
        if root==None:
            return
        if root.left==None and root.right==None:
            return
        root.left.next=root.right
        if root.next!=None:
            root.right.next=root.next.left
        self.connect(root.left)
        self.connect(root.right)

#117. Populating Next Right Pointers in Each Node II(Medium)
class Solution:
    # @param root, a tree link node
    # @return nothing
    def connect(self, root):
        if root==None:
            return
        if root.left==None and root.right==None:
            return
        if root.left!=None and root.right==None:
            p=root.next
            while p!=None:
                if p.left!=None:
                    p=p.left
                    break
                if p.right!=None:
                    p=p.right
                    break
                if p.right==None and p.left==None:
                    p=p.next
            root.left.next=p
        if root.right!=None:
            if root.left!=None:
                root.left.next=root.right
            p=root.next
            while p!=None:
                if p.left!=None:
                    p=p.left
                    break
                if p.right!=None:
                    p=p.right
                    break
                if p.right==None and p.left==None:
                    print(p.val,p.next)
                    if p.next!=None:
                        print(p.next.val)
                    p=p.next
            root.right.next=p
        self.connect(root.right)
        self.connect(root.left)

#118. Pascal's Triangle(Easy)
class Solution(object):
    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        if numRows==0:
            return []
        if numRows==1:
            return [[1]]
        else:
            prev=self.generate(numRows-1)
            new,start=[],0
            for i in prev[-1]:
                new.append(start+i)
                start=i
            new.append(1)
            prev.append(new)
            return prev
#119. Pascal's Triangle II(Easy)
class Solution(object):
    def getRow(self, rowIndex):
        """
        :type rowIndex: int
        :rtype: List[int]
        """
        res=[]
        for i in range(rowIndex+1):
            temp,start=[],0
            for j in res:
                temp.append(j+start)
                start=j
            temp.append(1)
            res=temp
        return res
#120. Triangle(Medium)
class Solution(object):
    def minimumTotal(self, triangle):
        """
        :type triangle: List[List[int]]
        :rtype: int
        """
        #Time:O(2^n),Space:constant
        self.max_sum=None
        self.find(0,triangle,0,0)
        return self.max_sum

    def find(self,level,triangle,start,cur_sum):
        if level==len(triangle):
            if self.max_sum==None or cur_sum<self.max_sum:
                self.max_sum=cur_sum
        else:
            cur_sum=cur_sum+triangle[level][start]
            for new_start in [start,start+1]:
                self.find(level+1,triangle,new_start,cur_sum)

    def minimumTotal(self, triangle):
        """
        :type triangle: List[List[int]]
        :rtype: int
        """
        #Time:O(n*m),Space:O(n) n=len(tri),m=len(tri[0])
        if triangle==[]:
            return 0
        cur_path=triangle[0]
        for i in range(1,len(triangle)):
            prev_path=cur_path
            cur_path=[0]*len(triangle[i])
            for j in range(len(triangle[i])):
                if j==0:
                    cur_path[j]=prev_path[0]+triangle[i][j]
                elif j==len(triangle[i])-1:
                    cur_path[j]=prev_path[-1]+triangle[i][j]
                else:
                    cur_path[j]=min(prev_path[j],prev_path[j-1])+triangle[i][j]
        return min(cur_path)
board=[[-1],[3,2],[-3,1,-1]]

#121. Best Time to Buy and Sell Stock(Easy)
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        min_price=None
        profit=[]
        for i in range(len(prices)):
            if min_price==None or prices[i]<min_price:
                min_price=prices[i]
            profit.append(prices[i]-min_price)
        return max(profit)

#122. Best Time to Buy and Sell Stock II(Easy)
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        profit=0
        for i in range(1,len(prices)):
            di=prices[i]-prices[i-1]
            if di>0:
                profit+=di
        return profit

#123. Best Time to Buy and Sell Stock III(Hard)
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        #Dp
        dp=[[0 for i in range(len(prices))] for i in range(3)]
        for i in range(1,3):
            max_dif=dp[i][0]-prices[0]
            for j in range(1,len(prices)):
                no_tras=dp[i][j-1]
                max_dif=max(max_dif,dp[i-1][j]-prices[j])
                dp[i][j]=max(dp[i][j-1],prices[j]+max_dif)       
        return dp[-1][-1]

#124. Binary Tree Maximum Path Sum
class Solution(object):
    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.max_sum=root.val
        self.maxPath(root)
        return self.max_sum


    def maxPath(self,root):
        if root==None:
            return 0
        left=max(0,self.maxPath(root.left))
        right=max(0,self.maxPath(root.right))
        self.max_sum=max(self.max_sum,left+right+root.val)
        return max(left,right)+root.val
#125. Valid Palindrome(Easy)
class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        if s=="":
            return True
        if len(s)==1:
            return True
        start,end=0,len(s)-1
        while start<end:
            if (s[start].isdigit() or s[start].isalpha()) and (s[end].isdigit() or s[end].isalpha()):
                if s[start].lower()!=s[end].lower():
                    return False
                start+=1
                end-=1
            if not (s[start].isdigit() or s[start].isalpha()):
                start+=1
            if not (s[end].isdigit() or s[end].isalpha()):
                end-=1
        return True

#126. Word Ladder II(Hard)
class Solution(object):
    def findLadders(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: List[List[str]]
        """
        #build_map
        if endWord not in wordList:
            return []
        words=set(wordList)
        words.add(beginWord)
        prev_level={beginWord}
        Map={}
        while(prev_level!=set()):
            cur_level=set()
            for cur in prev_level:
                words.remove(cur)
            for cur in prev_level:
                if cur not in Map:
                    Map[cur]=[]
                for j in "abcdefghijklmnopqrstuvwxyz":
                    for i in range(len(cur)):
                        if cur[i]!=j:
                            word=cur[:i]+j+cur[i+1:]
                            if word in words:
                                cur_level.add(word)
                                Map[cur].append(word)
            prev_level=cur_level
            if prev_level==[]:
                return []
            if endWord in prev_level:
                break
        #build_path
        res=[]
        print(Map)
        def build(path,word):
            if endWord in Map[word]:
                res.append(path+[word]+[endWord])
            else:
                for next_word in Map[word]:
                    if next_word in Map:
                        build(path+[word],next_word)
        build([],beginWord)
        return res
solution=Solution()
# print(solution.findLadders("hit","cog",["hot","dot","dog","lot","log","cog"]))
# print(solution.findLadders("teach","place",["peale","wilts","place","fetch","purer","pooch","peace","poach","berra","teach","rheum","peach"]))
# print(solution.findLadders("red","tax",["ted","tex","red","tax","tad","den","rex","pee"]))
#127. Word Ladder(Medium)
class Solution(object):
    # def ladderLength(self, beginWord, endWord, wordList):
    #     """
    #     :type beginWord: str
    #     :type endWord: str
    #     :type wordList: List[str]
    #     :rtype: int
    #     """
    #     if endWord not in wordList:
    #         return 0
    #     if beginWord in wordList:
    #         return 1
    #     ruoud=[]
    #     res=1
    #     seen=set()
    #     ruoud.append(beginWord)
    #     while(len(seen)!=len(wordList)):
    #         next_round=[]
    #         for cur in ruoud:
    #             for word in wordList:
    #                 if self.diff(cur,word)==1:
    #                     if word not in seen:
    #                         seen.add(word)
    #                         next_round.append(word)
    #                         if word==endWord:
    #                             return res+1
    #         res+=1
    #         ruoud=next_round
    #         if ruoud==[]:
    #             break
    #     return 0

    def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """
        que=[]
        words=set(wordList)
        que.append((beginWord,1))
        while(que!=[]):
            target=que[0]
            que=que[1:]
            cur=target[0]
            if cur==endWord:
                return target[1]
            for j in "abcdefghijklmnopqrstuvwxyz":
                for i in range(len(cur)):
                    word=cur[:i]+j+cur[i+1:]
                    if word in words:
                        que.append((word,target[1]+1))
                        words.remove(word)
        return 0
    def diff(self,s,t):
        diff=0
        for i in range(len(s)):
            if s[i]!=t[i]:
                diff+=1
        return diff

# solution=Solution()
# print(solution.ladderLength("hit","cog",["hot","dot","dog","lot","log","cog"]))

#128. Longest Consecutive Sequence(Hard)
class Solution(object):
    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        collections=set(nums)
        res=1
        for n in nums:
            up,down=n+1,n-1
            print(n,collections)
            while(up in collections):
                collections.remove(up)
                up+=1
            while (down in collections):
                collections.remove(down)
                down-=1
            print(up,down)
            res=max(res,up-down-1)
        return res

#129. Sum Root to Leaf Numbers(Medium)
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution(object):
    def sumNumbers(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.res=0
        self.traversal(root)
        return self.res
    def traversal(self,root):
        if root==None:
            return 
        if root.left==None and root.right==None:
            self.res+=root.val
        if root.left!=None:
            root.left.val+=(10*root.val)
        if root.right!=None:
            root.right.val+=(10*root.val)
        self.traversal(root.left)
        self.traversal(root.right)

r=TreeNode(2)
r.left=TreeNode(0)
r.right=TreeNode(0)
solution=Solution()
# print(solution.sumNumbers(r))

#130. Surrounded Regions(Medium)
class Solution(object):
    def solve(self, board):
        """
        :type board: List[List[str]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        def travel(i,j):
            for direct in [(1,0),(-1,0),(0,1),(0,-1)]:
                p,q=i+direct[0],j+direct[1]
                if valid(p,q):
                    board[p][q]="F"
                    travel(p,q)
        def valid(i,j):
            if i<0 or i>=len(board):
                return False
            if j<0 or j>=len(board[0]):
                return False
            if board[i][j]!="O":
                return False
            return True
        for i in range(len(board)):
            if board[i][0]=="O":
                board[i][0]="F"
                travel(i,0)
            if board[i][-1]=="O":
                board[i][-1]="F"
                travel(i,len(board[0])-1)
        for j in range(len(board[0])):
            if board[0][j]=="O":
                board[0][j]="F"
                travel(0,j)
            if board[-1][j]=="O":
                board[-1][j]="F"
                travel(len(board),j)
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j]=="F":
                    board[i][j]="O"
                elif board[i][j]=="O":
                    board[i][j]=="X"

solution=Solution()
board=[["X","O","X"],["X","X","X"],["X","O","X"]]
print(len(board))
print(len(board[0]))
for i in range(len(board)):
    board[i]=list(board[i])
solution.solve(board)
print(board)
