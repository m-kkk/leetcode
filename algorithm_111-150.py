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
        res = []
        def check_path(path,current_sum,root):
            if not root:
                return
            if current_sum + root.val == sum:
                if not root.left and not root.right:
                    res.append(path+[root.val])
            check_path(path+[root.val],current_sum + root.val,root.left)
            check_path(path+[root.val],current_sum + root.val,root.right)
        check_path([],0,root)
        return res

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

#131. Palindrome Partitioning(Medium)
class Solution(object):
    def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """
        def dfs(s,path):
            if s=="":
                res.append(path)
            for i in range(1,len(s)+1):
                if s[:i]==s[:i][::-1]:
                    dfs(s[i:],path+[s[:i]])
        res=[]
        dfs(s,[])
        return res
solution=Solution()
# print(solution.partition("aab"))

#132. Palindrome Partitioning II(Hard)
class Solution(object):
    def minCut(self, s):
        """
        :type s: str
        :rtype: int
        """
        tf=[[False for i in range(len(s))] for j in range(len(s))]
        dp=[-1 for i in range(len(s))]
        for i in range(len(s)):
            tf[i][i]=True
        for l in range(2,len(s)+1):
            for i in range(len(s)-l+1):
                j=i+l-1
                if j-i==1:
                    tf[i][j]=(s[i]==s[j])
                elif j-i>1:
                    tf[i][j]=((s[i]==s[j])and(tf[i+1][j-1]))
        for i in range(len(s)):
            if tf[0][i]:
                dp[i]=0
            else:
                for j in range(1,i+1):
                    if tf[j][i]:
                        if dp[i]==-1:
                            dp[i]=1+dp[j-1]
                        else:
                            dp[i]=min(dp[i],1+dp[j-1])
        # print(tf)
        # print(dp)
        return dp[-1]
solution=Solution()
print(solution.minCut("tstitasaaasu"))


#133. Clone Graph(Medium)
class UndirectedGraphNode:
    def __init__(self, x):
        self.label = x
        self.neighbors = []

class Solution:
    # @param node, a undirected graph node
    # @return a undirected graph node
    def cloneGraph(self, node):
        dic={}
        def clone(node):
            if node==None:
                return None
            else:
                if node.label in dic:
                    return dic[node.label]
                root=UndirectedGraphNode(node.label)
                for neighbor in node.neighbors:
                    root.neighbors.append(clone(neighbor))
                dic[node.label]=root
                return root
        root=clone(node)
        return root

#134. Gas Station(Medium)
class Solution(object):
    def canCompleteCircuit(self, gas, cost):
        """
        :type gas: List[int]
        :type cost: List[int]
        :rtype: int
        """
        if gas==[] or cost==[]:
            return -1
        diff=[gas[i]-cost[i] for i in range(len(gas))]
        if sum(diff)<0:
            return -1
        total=0
        for i in range(len(diff)):
            total+=diff[i]
            if total<0:
                total=0
                candidate=i+1
        return candidate

#135. Candy(Hard)

class Solution(object):
    def candy(self, ratings):
        """
        :type ratings: List[int]
        :rtype: int
        """
        if ratings==[]:
            return 0
        candys=[1]*len(ratings)
        for p in range(1,len(ratings)):
            if ratings[p]>ratings[p-1]:
                candys[p]=candys[p-1]+1
        for p in reversed(range(len(ratings)-1)):
            print(p)
            if ratings[p]>ratings[p+1] and candys[p]<=candys[p+1]:
                candys[p]=candys[p+1]+1
        # print(candys,p)
        return sum(candys)
solution=Solution()
# print(solution.candy([7,6,5,4,3,4,3,2,1]))
# print(solution.candy([2,1]))
# print(solution.candy([1,3,4,3,2,1]))

#136. Single Number(Easy)
class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        seen=set()
        for n in nums:
            if n in seen:
                seen.remove(n)
            else:
                seen.add(n)
        return seen.pop()
# solution=Solution()
# print(solution.singleNumber([1,1,2,2,3,3,4]))

#137. Single Number II(Medium)
class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        res=[0]*32
        for n in nums:
            for i in reversed(range(32)):
                if (n>>i)%2==1:
                    res[i]+=1
        result=0
        for i in range(31):
            result+=(res[i]%3)<<(i)
        result+=-(res[31]%3)<<31
        return result
solution=Solution()
# print(solution.singleNumber([-2,-2,1,1,-3,1,-3,-3,-4,-2]))     

# print(solution.singleNumber([1,1,1,2,2,3,2,3,3,4])) 

#138. Copy List with Random Pointer(Medium)
class RandomListNode(object):
    def __init__(self, x):
        self.label = x
        self.next = None
        self.random = None

class Solution(object):
    def copyRandomList(self, head):
        """
        :type head: RandomListNode
        :rtype: RandomListNode
        """
        dic={}
        def copy(head):
            if head==None:
                return None
            if head.label in dic:
                return dic[head.label]
            else:
                root=RandomListNode(head.label)
                dic[head.label]=root
                root.next=copy(head.next)
                root.random=copy(head.random)
                return root
        return copy(head)

#139. Word Break(Medium)
class Solution(object):
    #Memorization
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        dic={}
        def serch(start,wordDict):
            if start==len(s):
                return True
            else:
                res=False
                for i in range(start,len(s)):
                    if s[start:i+1] in wordDict:
                        if (start,i) in dic:
                            res=res or dic[(start,i)]
                        else:
                            dic[start,i]=serch(i+1,wordDict)
                            res=res or dic[start,i]
                return res
        return serch(0,wordDict)
    #Dynamic Programming 
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        if wordDict==[]:
            return s==""
        dp=[False]*(len(s)+1)
        dp[0]=True
        wordDict=set(wordDict)
        max_length=max([len(w) for w in wordDict])
        for i in range(1,len(s)+1):
            for j in range(1,min(i,max_length)+1):
                if dp[i-j]:
                    if s[i-j:i] in wordDict:
                        dp[i]=True
                        break
        return dp[-1]
solution=Solution()
# print(solution.wordBreak("aaaaaaa",["aaaa","aaa"]))
#140. Word Break II(Hard)
class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: List[str]
        """
        if s=="" or wordDict==[]:
            return []
        res=[]
        dp=[False]*(len(s)+1)
        dp[-1]=True
        wordDict=set(wordDict)
        max_length=max([len(w) for w in wordDict])
        for i in reversed(range(len(s))):
            for j in range(1,min(len(s)-i,max_length)+1):
                if dp[i+j]:
                    if s[i:i+j] in wordDict:
                        dp[i]=True
                        break

        def dfs(s,start,path):
            if start==len(s):
                res.append(" ".join(path))
            for i in range(min(len(s)-start,max_length)):
                if s[start:start+i+1] in wordDict and dp[start+i+1]==True:
                    dfs(s,start+i+1,path+[s[start:start+i+1]])
        dfs(s,0,[])
        return res
solution=Solution()
# print(solution.wordBreak("catsanddog",["cat","cats","and","sand","dog"]))


#141. Linked List Cycle(Easy)
class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if head==None:
            return False
        t=head
        h=head.next
        while h!=t:
            if h==None or h.next==None:
                return False
            h=h.next.next
            t=t.next
        return True

#142. Linked List Cycle II(Medium)
class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head==None:
            return  None
        t=head
        h=head.next
        while h!=t:
            if h==None or h.next==None:
                return None
            h=h.next.next
            t=t.next
        while head!=t.next:
            head=head.next
            t=t.next
        return head

#143. Reorder List
class Solution(object):
    def reorderList(self, head):
        """
        :type head: ListNode
        :rtype: void Do not return anything, modify head in-place instead.
        """
        if head==None:
            return
        fast,slow=head,head
        while fast.next!=None and fast.next.next!=None:
            fast=fast.next.next
            slow=slow.next
        fast=slow.next
        slow.next=None
        dummy=ListNode(0)
        while fast!=None:
            p=fast.next
            fast.next=dummy.next
            dummy.next=fast
            fast=p
        fast=dummy.next
        slow=head
        while fast!=None:
            p=fast.next
            fast.next=slow.next
            slow.next=fast
            slow=fast.next
            fast=p
        

#144. Binary Tree Preorder Traversal(Medium)
class Solution(object):
    #Recursively
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if root==None:
            return []
        res=[]
        res+=self.preorderTraversal(root.left)
        res+=[root.val]
        res+=self.preorderTraversal(root.right)
        return res
    #Iteratively
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        res=[]
        stack=[root]
        while stack!=[]:
            node=stack.pop()
            if node :
                res.append(node.val)
                if node.right!=None:
                    stack.append(node.right)
                if node.left!=None:
                    stack.append(node.left)
        return res

#145. Binary Tree Postorder Traversal(Hard)
class Solution(object):
    #Recursively
    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if root==None:
            return []
        res=[]
        res+=self.postorderTraversal(root.left)
        res+=self.postorderTraversal(root.right)
        res+=[root.val]
        return res
    #Iteratively
    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        res=[]
        stack=[]
        curr=root
        prev=None
        while stack!=[] or curr!=None:
            while(curr!=None):
                stack.append(curr)
                curr=curr.left
            curr=stack[-1]
            if curr.right==None or curr.right==prev : 
                stack.pop()
                prev=curr
                res.append(curr.val)
                curr=None
            else:
                curr=curr.right
        return res

#146. LRU Cache(Hard)
#Singal linked-list + hash table
class linkedNode(object):
    def __init__(self,key=-1,val=-1,next=None):
        self.key=key
        self.val=val
        self.next=next
        
    def _print(self):
        node=self
        while node!=None:
            print(node.key,node.val,"--->",end="")
            node=node.next
        print()

class LRUCache(object):
    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.dict={}
        self.head=linkedNode()
        self.tail=self.head
        self.capacity=capacity

    def push(self,node):
        self.dict[node.key]=self.tail
        self.tail.next=node
        self.tail=node

    def pop(self):
        del self.dict[self.head.next.key]
        self.head.next=self.head.next.next
        self.dict[self.head.next.key]=self.head

    def move_back(self,key):
        if self.dict[key].next==self.tail:
            return
        node=self.dict[key].next
        self.dict[key].next=node.next
        self.dict[node.next.key]=self.dict[key]
        node.next=None
        self.push(node)

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if key not in self.dict:
            return -1
        self.move_back(key)
        return self.dict[key].next.val

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: void
        """
        if key in self.dict:
            self.dict[key].next.val=value
            self.move_back(key)
        else:
            node=linkedNode(key,value)
            self.push(node)
            if self.capacity<len(self.dict):
                self.pop()

#double linked-list and hash table
class Node(object):
        def __init__(self,key,value):
            self.key = key
            self.value = value
            self.prev = None
            self.next = None

class LRUCache:
    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.cap = capacity
        self.hsh = {}
        self.head = Node(-1,-1)
        self.tail = Node(-1,-1)
        self.head.next = self.tail
        self.tail.prev = self.head
        
    def push_tail(self,node):
        self.hsh[node.key] = node
        self.tail.prev.next =  node
        node.prev = self.tail.prev
        node.next = self.tail
        self.tail.prev = node
        
    def pop_head(self):
        if not self.hsh:
            return -1
        del self.hsh[self.head.next.key]
        self.head.next = self.head.next.next
        self.head.next.prev = self.head
        
    def put_tail(self,key):
        #find the node
        node = self.hsh[key]
        #get the node
        node.prev.next = node.next
        node.next.prev = node.prev
        #connext the node to tail
        node.next = self.tail
        self.tail.prev.next = node
        node.prev = self.tail.prev
        self.tail.prev = node
        
    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if not self.hsh:
            return -1
        if key not in self.hsh:
            return -1
        self.put_tail(key)
        return self.hsh[key].value

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: void
        """
        if key in self.hsh:
            self.hsh[key].value = value
            self.put_tail(key)
        else:
            self.push_tail(Node(key,value))
        if len(self.hsh)>self.cap:
            self.pop_head()


#147. Insertion Sort List(Medium)
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution(object):
    def insertionSortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head==None or head.next==None:
            return head
        node=head
        rest=self.insertionSortList(head.next)
        node.next=None
        p=ListNode(-1)
        res=p
        p.next=rest
        while p.next!=None:
            if p.next.val>node.val:
                node.next=p.next
                p.next=node
                return res.next
            p=p.next
        p.next=node
        return res.next
        
#148. Sort List(Medium)
class Solution(object):
    def sortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head==None or head.next==None:
            return head
        slow=head
        fast=head
        while fast!=None and fast.next!=None:
            slow=slow.next
            fast=fast.next
        fast=slow.next
        slow.next=None
        prefix=self.sortList(head)
        suffix=self.sortList(fast)
        return self.merge(prefix,suffix)

    def merge(self,A,B):
        dummy=ListNode(-1)
        p=dummy
        while A!=None or B!=None:
            if B==None or (A!=None and A.val<=B.val):
                p.next=A
                A=A.next
            else:
                p.next=B
                B=B.next
            p=p.next
        return dummy.next

#149. Max Points on a Line(Hard)
class Solution(object):
    def maxPoints(self, points):
        """
        :type points: List[Point]
        :rtype: int
        """
        if len(points)<=2:
            return len(points)
        max_count=0
        for i in range(len(points)-1):
            dic={}
            dup=0
            for j in range(i+1,len(points)):
                if points[i].x==points[j].x and points[i].y==points[j].y:
                    dup+=1
                    continue
                if (points[i].x-points[j].x)!=0:
                    slope=float(points[i].y-points[j].y)/(points[i].x-points[j].x)
                else:
                    slope=None
                if slope in dic:
                    dic[slope]+=1
                else:
                    dic[slope]=2
            print(dup,(points[i].x,points[i].y),dic)
            if len(dic)==0:
                count=dup+1
            else:
                count = max([dic[i] for i in dic])+dup
            if count>max_count:
                max_count=count
        return max_count

#150. Evaluate Reverse Polish Notation(Medium)
class Solution(object):
    def evalRPN(self, tokens):
        """
        :type tokens: List[str]
        :rtype: int
        """
        stack=[]
        for item in tokens:
            if item not in "+-*/":
                stack.append(item)
            else:
                first=stack.pop()
                second=stack.pop()
                equ=second+item+first
                stack.append(str(int(eval(equ))))
            print(stack)
        return int(stack[0])

solution=Solution()
print(solution.evalRPN(["10","6","9","3","+","-11","*","/","*","17","+","5","+"]))
# print(solution.evalRPN(["4", "13", "5", "/", "+"]))
