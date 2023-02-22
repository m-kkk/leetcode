#201. Bitwise AND of Numbers Range(Medium)
class Solution(object):
    def rangeBitwiseAnd(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        p=0
        while m!=n:
            m=m>>1
            n=n>>1
            p+=1
        return m<<p

#202. Happy Number(Easy)
class Solution(object):
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        seen=set()
        while n not in seen:
            seen.add(n)
            n=self.calculate(n)
            if n==1:
                return True
        return False
    def calculate(self,n):
        res=0
        while n!=0:
            m=n%10
            res+=m*m
            n=n//10
        return res

#203. Remove Linked List Elements(Easy)
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution(object):
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        dummy=ListNode(-1)
        dummy.next=head
        p=dummy
        while p!=None:
            if p.next!=None and p.next.val==val:
                p.next=p.next.next
            else:
                p=p.next
        return dummy.next

#204. Count Primes(Easy)
class Solution(object):
    def countPrimes(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n<=2:
            return 0
        isPrime=[True]*(n)
        isPrime[1]=False
        isPrime[0]=False
        for prime in range(1,int(n**0.5)+1):
            if isPrime[prime]:
                for multiple in range(2*prime,n,prime):
                    isPrime[multiple]=False
        return sum(isPrime)

    # def countPrimes(self, n):
    #     """
    #     :type n: int
    #     :rtype: int
    #     """
    #     def isPrime(n):
    #         if (n < 2): return False
    #         if (n == 2): return True
    #         if (n % 2 == 0): return False
    #         for factor in range(3, 1+int(round(n**0.5)), 2):
    #             if (n % factor == 0):
    #                 return False
    #         return True
    #     count=0
    #     for i in range(n):
    #         if isPrime(i):
    #             count+=1
    #     return count



#205. Isomorphic Strings(Easy)
class Solution(object):
    def isIsomorphic(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        if s=="":
            return t==""
        dic_s,dic_t={},{}
        for i in range(len(s)):
            if s[i] in dic_s:
                if dic_s[s[i]]!=t[i]:
                    return False
            else:
                dic_s[s[i]]=t[i]
            if t[i] in dic_t:
                if dic_t[t[i]]!=s[i]:
                    return False
            else:
                dic_t[t[i]]=s[i]
        return True

#206. Reverse Linked List(Easy)
class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        dummy=ListNode(-1)
        dummy.next=head
        prev=dummy
        while dummy.next!=None and head.next!=None:
            p=head.next
            head.next=p.next
            p.next=prev.next
            prev.next=p
        return dummy.next


#207. Course Schedule(Medium)
class Solution(object):
    #DFS
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        graph=[[]for i in range(numCourses)]
        for pair in prerequisites:
            graph[pair[0]].append(pair[1])
        seen=set()
        for i in range(numCourses):
            stack=[]
            if self.dfs(graph,i,stack,seen):
                return False
        return True

    def dfs(self,graph,i,stack,seen):
        if i in stack:
            return True
        if i in seen:
            return False
        stack.append(i)
        for j in graph[i]:
            if self.dfs(graph,j,stack,seen):
                return True
        stack.pop()
        seen.add(i)
        return False

    #Kanh Algorithm, BFS
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        indegrees = [0]*numCourses
        for pre in prerequisites:
            indegrees[pre[0]]+=1
        count,S = 0,set()
        for i in range(numCourses):
            if indegrees[i]==0:
                S.add(i)
        while S:
            n = S.pop()
            count+=1
            for pre in prerequisites:
                if pre[1]==n:
                    indegrees[pre[0]]-=1
                    if indegrees[pre[0]] == 0:
                        S.add(pre[0])
        return count==numCourses

#208. Implement Trie (Prefix Tree)(Medium)
class TrieNode:
  def __init__(self):
    # Initialize your data structure here.
    self.childs = dict()
    self.word=False

class Trie(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root=TrieNode()
        

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        if word=="":
            return
        node=self.root
        for c in word:
            child=node.childs.get(c)
            if child is None:
                child=TrieNode()
                node.childs[c]=child
            node=child
        node.word=True

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        node=self.root
        for c in word:
            child=node.childs.get(c)
            if child is None:
                return False
            node=child
        return node.word

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        node=self.root
        for c in prefix:
            child=node.childs.get(c)
            if child is None:
                return False
            node=child
        return True

#209. Minimum Size Subarray Sum(Medium)
class Solution(object):
    def minSubArrayLen(self, s, nums):
        """
        :type s: int
        :type nums: List[int]
        :rtype: int
        """
        start,end=0,0
        total=nums[start]
        min_len=0
        while end!=len(nums)-1 or total>=s:
            if total>=s:
                if min_len==0 or end-start+1<min_len:
                    min_len=end-start+1
                total-=nums[start]
                start+=1
            else:
                end+=1
                if end>=len(nums):
                    return min_len
                total+=nums[end]
        return min_len
solution=Solution()
# print(solution.minSubArrayLen(7,[2,3,1,2,4,3]))

#210. Course Schedule II(Medium)
class Solution(object):
    #DFS
    def findOrder(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """
        self.order=numCourses-1
        graph=[[]for i in range(numCourses)]
        for pair in prerequisites:
            graph[pair[1]].append(pair[0])
        seen=set()
        # print(graph)
        order=[-1]*numCourses
        res=[-1]*numCourses
        for i in range(numCourses):
            if not self.dfs(graph,i,order,seen):
                return []
        # print(order,res,numCourses)
        for i in range(numCourses):
            res[order[i]]=i
        return res

    def dfs(self,graph,i,order,seen):
        if i in seen and order[i]!=-1:
            return True
        if i in seen and order[i]==-1:
            return False
        # print(i,order,seen,self.order)
        seen.add(i)
        for j in graph[i]:
            if not self.dfs(graph,j,order,seen):
                return False
        if order[i]==-1:
            order[i]=self.order
        else:
            return False
        self.order-=1
        return True
        
    #Kahn's Algorithm, BFS
    def findOrder(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """
        indegrees = [0]*numCourses
        for pre in prerequisites:
            indegrees[pre[0]]+=1
        count,S = 0,set()
        Q = []
        for i in range(numCourses):
            if indegrees[i]==0:
                S.add(i)
        while S:
            n = S.pop()
            Q.append(n)
            count+=1
            for pre in prerequisites:
                if pre[1]==n:
                    indegrees[pre[0]]-=1
                    if indegrees[pre[0]] == 0:
                        S.add(pre[0])
        if len(Q)==numCourses:
            return Q
        else:
            return []

solution=Solution()
# print(solution.findOrder(4, [[1,0],[2,0],[3,1],[3,2]]))
# print(solution.findOrder(2, [[0,1],[1,0]]))


#211. Add and Search Word - Data structure design(Medium)
class TrieNode:
  def __init__(self):
    # Initialize your data structure here.
    self.childs = dict()
    self.word=False

class Trie(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root=TrieNode()

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        if word=="":
            return
        node=self.root
        for c in word:
            child=node.childs.get(c)
            if child is None:
                child=TrieNode()
                node.childs[c]=child
            node=child
        node.word=True

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        return self.serch_node(word,self.root)

    def serch_node(self, word, node):
        res=False
        for i in range(len(word)):
            if word[i]!=".":
                child=node.childs.get(word[i])
                if child is None:
                    return False
                node=child
            else:
                for key in node.childs:
                    res=res or self.serch_node(word[i+1:],node.childs[key])
                return res
        return node.word

class WordDictionary(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.trie=Trie()
        

    def addWord(self, word):
        """
        Adds a word into the data structure.
        :type word: str
        :rtype: void
        """
        self.trie.insert(word)
        

    def search(self, word):
        """
        Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
        :type word: str
        :rtype: bool
        """
        return self.trie.search(word)

#212. Word Search II(Hard)
# class TrieNode:
#   def __init__(self):
#     # Initialize your data structure here.
#     self.childs = dict()
#     self.word=False
#     self.has=False

class Trie(object):
    def __init__(self):
        """
        Initialize your data structure here.
        """
        # self.root=TrieNode()
        self.childs = dict()
        self.word=False
        self.has=False

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        if word=="":
            self.has=True
            self.word=True
            return
        if word[0] not in self.childs:
            self.childs[word[0]]=Trie()
        self.childs[word[0]].insert(word[1:])
        self.has=True

    def delete(self,word):
        if word=="":
            self.has=False
            self.word=False
            return
        if word[0] not in self.childs:
            return
        self.childs[word[0]].delete(word[1:])
        self.has=any(child.has for child in self.childs.values())


class Solution(object):
    def findWords(self, board, words):
        """
        :type board: List[List[str]]
        :type words: List[str]
        :rtype: List[str]
        """
        self.res=set()
        self.trie=Trie()
        for word in words:
            self.trie.insert(word)
        rows,cols=len(board),len(board[0])
        for row in range(rows):
            for col in range(cols):
                self.seen=set()
                self.dfs(row,col,board,self.trie,[])
        return list(self.res)


    def dfs(self,row,col,board,node,path):
        # if node.word:
        #     self.res.append("".join(path))
        #     return
        if not self.isLegal(board,row,col):
            return
        char=board[row][col]
        if char not in node.childs:
            return
        else:
            new_node=node.childs[char]
            if new_node.word:
                line="".join(path+[char])
                self.res.add((line))
                self.trie.delete(line)
            self.seen.add((row,col))
            self.dfs(row+1,col,board,new_node,path+[char])
            self.dfs(row,col+1,board,new_node,path+[char])
            self.dfs(row-1,col,board,new_node,path+[char])
            self.dfs(row,col-1,board,new_node,path+[char])
            self.seen.remove((row,col))

    def isLegal(self,board,row,col):
        rows,cols=len(board),len(board[0])
        if row<0 or row>=rows:
            return False
        if col<0 or col>=cols:
            return False
        if (row,col) in self.seen:
            return False
        return True


solution=Solution()
board=[
  ['o','a','a','n'],
  ['e','t','a','e'],
  ['i','h','k','r'],
  ['i','f','l','v']
]
# board=[["a","b"],["a","a"]]
words=["oath","pea","eat","rain"]
# words=["aba","baa","bab","aaab","aaa","aaaa","aaba"]
# print(solution.findWords(board,words))

#213. House Robber II(Medium)
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        return max(self.rob_one(nums[1:]),self.rob_one(nums[:-1]))
        

    def rob_one(self,nums):
        if nums==[]:
            return 0
        t=[0]*len(nums)#the maximal value when u pick this house
        f=[0]*len(nums)#the maximal value when u don't pick this house
        t[0]=nums[0]
        for i in range(1,len(nums)):
            t[i]=f[i-1]+nums[i]
            f[i]=max(t[i-1],f[i-1])
        return max(f[-1],t[-1])

#214. Shortest Palindrome(Hard)
class Solution(object):
    def shortestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        #find the longest palindrome prefix
        # j=0
        # for i in reversed(range(len(s))):
        #     if s[i]==s[j]:
        #         j+=1
        # if j==len(s):
        #     return s
        r=s+s[::-1]
        j=self.longest_pref_suff(r)
        print(r)
        print(j)
        suf=s[j:]
        # mid=self.shortestPalindrome(s[:j])
        prefix=suf[::-1]
        return prefix+s[:j]+suf

    def longest_pref_suff(self,s):
        l=[0]*(len(s)+1)
        l[0]=-1
        for i in range(len(s)):
            k=l[i]
            while k>=0 and s[k]!=s[i]:
                k=l[k]
            l[i+1]=k+1
        print(l)
        return l[-1]

print("00000000000")
solution=Solution()
print(solution.shortestPalindrome("axaxbcxbqa"))
print(solution.longest_pref_suff("abaxxxxxxaba"))


#215. Kth Largest Element in an Array(Medium)
class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        print(list(reversed(sorted(nums))),k)
        return self.find(nums,k-1,0,len(nums)-1)

    def find(self,nums,k,left,right):
        mid = (left+right)//2
        piv=nums[mid]
        l=left
        r=right
        while(left<right):
            while(nums[left]>piv):
                left+=1
            while (nums[right]<piv):
                right-=1
            print(left,right)
            if (left<=right):
                nums[left],nums[right]=nums[right],nums[left]
                left+=1
                right-=1
        print(nums,piv,left,right,l,r)
        if l<right and k<=right:
            return self.find(nums,k,l,right)
        if r>left and k>=left:
            return self.find(nums,k,left,r)
        return nums[k]

solution=Solution()
print(solution.findKthLargest([1,1,4,5,6,77,7,7,5,4,2,79,9,5,4,6,6,2,3],9))

#216. Combination Sum III(Medium)
class Solution(object):
    def combinationSum3(self, k, n):
        """
        :type k: int
        :type n: int
        :rtype: List[List[int]]
        """
        self.res=[]
        self.dfs(0,k,n,[])
        return self.res

    def dfs(self,i,k,n,nums):
        if len(nums)==k and sum(nums)==n:
            self.res.append(nums)
        target=n-sum(nums)
        for j in range(i+1,10):
            if j>target:
                return
            else:
                self.dfs(j,k,n,nums+[j])

solution=Solution()
# print(solution.combinationSum3(3,15))

#217. Contains Duplicate(Easy)
class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        return not len(nums)==len(set(nums))

#218. The Skyline Problem(Hard)
class Solution(object):
    def getSkyline(self, buildings):
        """
        :type buildings: List[List[int]]
        :rtype: List[List[int]]
        """
        #heap
        heights=[]
        res=[]
        for building in buildings:
            heights.append((building[0],-building[2]))#save left end as positive
            heights.append((building[1],building[2]))#save right end as negative
        heights=sorted(heights)
        print(heights)
        pq=[]
        prev=0
        pq.append(prev)
        for height in heights:
            if height[1]<0:
                pq.append(-height[1])
            else:
                pq.remove(height[1])
            print(height[1],pq)
            cur=max(pq)
            if prev!=cur:
                res.append([height[0],cur])
                prev=cur
        return res

    def getSkyline(self, buildings):
        import heapq
        points = []
        n = len(buildings)
        for building in buildings:
            points.append(building[0])
            points.append(building[1])
        points.sort()
        res = []
        heap = [(0,float('inf'))]
        prev = 0
        i = 0
        for x in points:
            while heap and heap[0][1]<=x:
                heapq.heappop(heap)0.
            while i<n and buildings[i][0]<=x:
                heapq.heappush(heap,(-buildings[i][2],buildings[i][1]))
                i+=1
            current = -heap[0][0]
            if current!=prev:
                res.append([x,current])
                prev = current
        return res

    # def getSkyline(self, buildings):
    #     """
    #     :type buildings: List[List[int]]
    #     :rtype: List[List[int]]
    #     """
    #     #Merge
    #     # print(buildings)
    #     if len(buildings)==1:
    #         res=[]
    #         building=buildings[0]
    #         res.append([building[0],building[2]])
    #         res.append([building[1],0])
    #         # print(res)
    #         return res
    #     mid=len(buildings)//2
    #     sky1=self.getSkyline(buildings[:mid])
    #     # print(sky1,buildings[:mid])
    #     sky2=self.getSkyline(buildings[mid:])
    #     # print(sky2,buildings[mid:])
    #     return self.merge(sky1,sky2)

    # def merge(self,sky1,sky2):
    #     i,j=0,0
    #     n,m=len(sky1),len(sky2)
    #     res=[]
    #     h1,h2=0,0
    #     while i<n and j<m:
    #         if sky1[i][0]<sky2[j][0]:
    #             h1=sky1[i][1]
    #             h=max(h1,h2)
    #             # print([sky1[i][0],h])
    #             self.append(res,[sky1[i][0],h])
    #             i+=1
    #         elif sky1[i][0]>sky2[j][0]:
    #             h2=sky2[j][1]
    #             h=max(h1,h2)
    #             # print([sky2[j][0],h])
    #             self.append(res,[sky2[j][0],h])
    #             j+=1
    #         else:
    #             h1=sky1[i][1]
    #             h2=sky2[j][1]
    #             h=max(h1,h2)
    #             # print([sky2[j][0],h])
    #             self.append(res,[sky2[j][0],h])
    #             j+=1
    #     while i<n:
    #         self.append(res,sky1[i])
    #         i+=1
    #     while j<m:
    #         self.append(res,sky2[j])
    #         j+=1
    #     return res

    # def append(self,l,target):
    #     if l==[]:
    #         l.append(target)
    #     left=l[-1][0]
    #     height=l[-1][1]
    #     if target[1]==height:
    #         return
    #     if target[0]==left:
    #         l[-1][0]=max(height,target[1])
    #         return
    #     l.append(target)
solution=Solution()
print("---------------!!!!!!!!!!!----------------")
print(solution.getSkyline([[0,2,3],[2,5,3]]))

# print(solution.getSkyline([[0,3,3],[1,5,3],[2,4,3],[3,7,3]]))
# print(solution.getSkyline([[1,2,1],[1,2,2],[1,2,3]]))
# print(solution.getSkyline([[1,5,11], [2,7,6], [3,9,13], [12,16,7], [14,25,3],[19,22,18],[23,29,13],[24,28,4]]))

#219. Contains Duplicate II(Easy)
class Solution(object):
    def containsNearbyDuplicate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        dic={}
        for i in range(len(nums)):
            if nums[i] in dic:
                return True
            else:
                dic[nums[i]]=i
            if len(dic)>k:
                del dic[nums[i-k]]
        return False
solution=Solution()
# print(solution.containsNearbyDuplicate([1,2,3,4,1,6],4))

#220. Contains Duplicate III(Medium)
class Solution(object):
    def containsNearbyAlmostDuplicate(self, nums, k, t):
        """
        :type nums: List[int]
        :type k: int
        :type t: int
        :rtype: bool
        """
        if t<0:
            return False
        div=t+1
        dic={}
        for i,num in enumerate(nums):
            index=num//div
            if (index in dic) or (index+1 in dic and dic[index+1]-num<=t) or (index-1 in dic and num-dic[index-1]<=t):
                return True
            dic[index]=num
            if len(dic)>k:
                del dic[nums[i-k]//div]
        return False

#221. Maximal Square(Medium)
class Solution(object):
    def maximalSquare(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        if matrix==[]:
            return 0
        m,n=len(matrix),len(matrix[0])
        dp=[[0]*n for i in range(m)]
        max_square=0
        for i in range(m):
            if matrix[i][0]=="1":
                dp[i][0]=1
                max_square=1
        for j in range(n):
            if matrix[0][j]=="1":
                dp[0][j]=1
                max_square=1
        for i in range(1,m):
            for j in range(1,n):
                if matrix[i][j]=="1":
                    dp[i][j]=min(dp[i][j-1],dp[i-1][j-1],dp[i-1][j])+1
                    if dp[i][j]>max_square:
                        max_square=dp[i][j]
        return max_square**2
solution=Solution()
# print(solution.maximalSquare([["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","0","1"]]))
# print(solution.maximalSquare([["1"]]))

#222. Count Complete Tree Nodes(Medium)

# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def countNodes(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        return self.count_node(root,-1,-1)

    def count_node(self,root,left_height,right_height):
        if left_height==-1:
            node=root
            left_height=0
            while node!=None:
                left_height+=1
                node=node.left
        if right_height==-1:
            node=root
            right_height=0
            while node!=None:
                right_height+=1
                node=node.right
        if left_height==right_height:
            return (1<<left_height)-1
        else:
            return 1+self.count_node(root.left,left_height-1,-1)+self.count_node(root.right,-1,right_height-1)

#223. Rectangle Area(Medium)
class Solution(object):
    def computeArea(self, A, B, C, D, E, F, G, H):
        """
        :type A: int
        :type B: int
        :type C: int
        :type D: int
        :type E: int
        :type F: int
        :type G: int
        :type H: int
        :rtype: int
        """
        if C<E or A>G or B>H or D<F:
            overlap=0
        else:
            x=sorted([A,C,E,G])
            y=sorted([B,D,F,H])
            overlap=(x[2]-x[1])*(y[2]-y[1])
        return (C-A)*(D-B)+(G-E)*(H-F)-overlap

#224. Basic Calculator(Hard)
class Solution(object):
    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        s=s.replace(" ","")
        sign_stack=[1]
        i,sign,res=0,1,0
        while i<len(s):
            c=s[i]
            if c=="+":
                sign=1
                i+=1
            elif c=="-":
                sign=-1
                i+=1
            elif c=="(":
                sign_stack.append(sign*sign_stack[-1])
                sign=1
                i+=1
            elif c==")":
                sign_stack.pop()
                i+=1
            else:
                i+=1    
                while i<len(s) and s[i].isdigit():
                    c+=s[i]
                    i+=1
                res+=int(c)*sign*sign_stack[-1]
        return res
solution=Solution()
# print(solution.calculate("(1+(4+5+2)-3)+(6+8)"))

#225. Implement Stack using Queues(Easy)
class MyStack(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.queue1=[]
        self.queue2=[]


    def push(self, x):
        """
        Push element x onto stack.
        :type x: int
        :rtype: void
        """
        self.queue2.append(x)
        while self.queue1!=[]:
            self.queue2.append(self.queue1[0])
            self.queue1=self.queue1[1:]
        self.queue1,self.queue2=self.queue2,self.queue1

    def pop(self):
        """
        Removes the element on top of the stack and returns that element.
        :rtype: int
        """
        return self.queue1.pop()

    def top(self):
        """
        Get the top element.
        :rtype: int
        """
        return self.queue1[-1]
        

    def empty(self):
        """
        Returns whether the stack is empty.
        :rtype: bool
        """
        return self.queue1==[]

#226. Invert Binary Tree(Easy)
class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if root==None:
            return None
        if root.left==None and root.right==None:
            return root
        p=root.left
        root.left=self.invertTree(root.right)
        root.right=self.invertTree(p)
        return root

#227. Basic Calculator II(Medium)
class Solution(object):
    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        s=s.replace(" ","")
        s+="+"
        i,res,cur,prev=0,0,0,0
        sign=1
        while i<len(s):
            c=s[i]
            if c=="+":
                res+=cur*sign
                sign=1
                i+=1
            elif c=="-":
                res+=cur*sign
                sign=-1
                i+=1
            elif c=="*":
                prev=cur
                i+=1
                d=""
                while i<len(s) and s[i].isdigit():
                    d+=s[i]
                    i+=1
                cur=int(d)
                cur=prev*cur
            elif c=="/":
                prev=cur
                i+=1
                d=""
                while i<len(s) and s[i].isdigit():
                    d+=s[i]
                    i+=1
                cur=int(d)
                cur=prev//cur
            else:
                i+=1
                while i<len(s) and s[i].isdigit():
                    c+=s[i]
                    i+=1
                cur=int(c)
        return res
solution=Solution()
# print(solution.calculate("3*5+5*1/2"))

#228. Summary Ranges(Medium)
class Solution(object):
    def summaryRanges(self, nums):
        """
        :type nums: List[int]
        :rtype: List[str]
        """
        if nums==[]:
            return []
        res=[]
        first=str(nums[0])
        second=None
        for i in range(1,len(nums)):
            if nums[i]!=nums[i-1]+1:
                second=str(nums[i-1])
                if first==second:
                    res.append(first)
                else:
                    res.append(first+"->"+second)
                first=str(nums[i])
                second=None
        if first==str(nums[-1]):
            res.append(first)
        else:
            second=str(nums[-1])
            res.append(first+"->"+second)
        return res

#229. Majority Element II(Medium)
class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        #Moore voting algorithm T(n) S(1)
        res=[]
        if nums==[]:
            return res
        count_a=0
        count_b=0
        A,B=-1,-1
        for num in nums:
            if A==num:
                count_a+=1
            elif B==num:
                count_b+=1
            elif count_a==0:
                A=num
                count_a+=1
            elif count_b==0:
                B=num
                count_b+=1
            else:
                count_a-=1
                count_b-=1
        count_a,count_b=0,0
        for num in nums:
            if A==num:
                count_a+=1
            if B==num:
                count_b+=1
        if count_a>(len(nums)//3):
            res.append(A)
        if count_b>(len(nums)//3) and A!=B:
            res.append(B)
        return res


solution=Solution()
# print(solution.majorityElement([8,8,7,7,7]))

#230. Kth Smallest Element in a BST(Medium)
class Solution(object):
    def kthSmallest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        if root==None:
            return 0
        node=root
        res=[]
        stack=[]
        while stack!=[] or node==None:
            if node!=None:
                stack.append(node)
                node=node.left
            else:
                node=stack.pop()
                res.append(node.val)
                node=node.right
        return res[k-1]

#231. Power of Two(Easy)
class Solution(object):
    def isPowerOfTwo(self, n):
        """
        :type n: int
        :rtype: bool
        """
        k=1
        while k<n:
            k=k<<1
        return k==n

#232. Implement Queue using Stacks(Easy)
class MyQueue(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.instack=[]
        self.outstack=[]

    def push(self, x):
        """
        Push element x to the back of queue.
        :type x: int
        :rtype: void
        """
        self.instack.append(x)
        

    def pop(self):
        """
        Removes the element from in front of queue and returns that element.
        :rtype: int
        """
        if self.outstack!=[]:
            return self.outstack.pop()
        else:
            while self.instack!=[]:
                self.outstack.append(self.instack.pop())
            return self.outstack.pop()


    def peek(self):
        """
        Get the front element.
        :rtype: int
        """
        if self.outstack!=[]:
            return self.outstack[-1]
        else:
            while self.instack!=[]:
                self.outstack.append(self.instack.pop())
            return self.outstack[-1]

    def empty(self):
        """
        Returns whether the queue is empty.
        :rtype: bool
        """
        return self.instack==[] and self.outstack==[]

#233. Number of Digit One(Hard)
class Solution(object):
    def countDigitOne(self, n):
        """
        :type n: int
        :rtype: int
        """
        k,res=1,0
        while k<=n:
            #count 1 that appears on each digit from units to highest  
            r=n//k
            c=n%k
            if r%10==0:
                #if this digit==0 then 1 appears on this digit is k many times every k*10 numbers 
                res+=r//10*k
            elif r%10==1:
                #if this digit==1 then 1 appears on this digit is k many times every k*10 numbers plus the remainder
                res+=r//10*k+c+1
            elif r%10>=2:
                #if this digit>=2 then 1 appears on this digit is k many times every k*10 numbers plus another whole k times
                res+=(r//10+1)*k
            k*=10
        return res
solution=Solution()
# print(solution.countDigitOne(1))

#234. Palindrome Linked List(Easy)
class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if head==None or head.next==None:
            return True
        fast,slow=head,head
        while fast.next!=None and fast.next.next!=None:
            slow=slow.next
            fast=fast.next.next
        if fast.next==None:
            second=slow.next
            slow.next=None
            head=self.reverse(head)
            return self.comp(head.next,second)
        else:
            second=slow.next
            slow.next=None
            head=self.reverse(head)
            return self.comp(head,second)
            
    def reverse(self, head):
        dummy=ListNode(-1)
        dummy.next=head
        prev=dummy
        while dummy.next!=None and head.next!=None:
            p=head.next
            head.next=p.next
            p.next=dummy.next
            dummy.next=p
        return dummy.next

    def comp(self, head_a, head_b):
        p=head_a
        q=head_b
        while p!=None or q!=None:
            if p==None or q==None:
                return False
            if p.val!=q.val:
                return False
            p=p.next
            q=q.next
        return True

#235. Lowest Common Ancestor of a Binary Search Tree(Easy)
class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if root==None or root.val==p.val or root.val==q.val:
            return root
        if root.val<p.val and root.val<q.val:
            return self.lowestCommonAncestor(root.right,p,q)
        elif root.val>p.val and root.val>q.val:
            return self.lowestCommonAncestor(root.left,p,q)
        else:
            return root

#236. Lowest Common Ancestor of a Binary Tree(Medium)
class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if root is None or root is p or root is q:
            return root
        left=self.lowestCommonAncestor(root.left,p,q)
        right=self.lowestCommonAncestor(root.right,p,q)
        if left==None:
            return right
        if right==None:
            return left
        if left==None and right==None:
            return None
        return root

#237. Delete Node in a Linked List(Easy)
class Solution(object):
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        if node==None or node.next==None:
            return
        node.val=node.next.val
        node.next=node.next.next

#238. Product of Array Except Self(Medium)
class Solution(object):
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        # forward=[1]*len(nums)
        # backward=[1]*len(nums)
        res=[1]*len(nums)
        # for i in range(1,len(nums)):
        #     forward[i]=forward[i-1]*nums[i-1]
        # for i in reversed(range(len(nums)-1)):
        #     backward[i]=backward[i+1]*nums[i+1]
        # for i in range(len(nums)):
        #     res[i]=backward[i]*forward[i]
        for i in range(1,len(nums)):
            res[i]=res[i-1]*nums[i-1]
        right=1
        for i in reversed(range(len(nums)-1)):
            right*=nums[i+1]
            res[i]=res[i]*right
        return res
solution=Solution()
# print(solution.productExceptSelf([1,2,3,4]))

#239. Sliding Window Maximum(Hard)
class Solution(object):
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        windows = []
        res = []
        for i,n in enumerate(nums):
            if len(windows)>0 and windows[0]<i-k:
                windows.pop(0)
            while len(windows)>0 and nums[windows[-1]]<n:
                windows.pop()
            windows.append(i)
            # print(windows)
            if i >= k-1:
                res.append(nums[windows[0]])
        return res

solutino=Solution()
print(solutino.maxSlidingWindow([1,3,-1,-3,5,3,6,7],3))

print('---------------------------')
#240. Search a 2D Matrix II(Medium)
class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if matrix==[] or matrix[0]==[]:
            return False
        row,col=len(matrix),len(matrix[0])
        i,j=0,col-1
        while i<row and j>=0:
            if matrix[i][j]==target:
                return True
            elif matrix[i][j]>target:
                j-=1
            elif matrix[i][j]<target:
                i+=1
        return False

#241. Different Ways to Add Parentheses(Medium)
class Solution(object):
    def diffWaysToCompute(self, input):
        """
        :type input: str
        :rtype: List[int]
        """
        if type(input)!=type([]):
            input=self.parse(input)
        if len(input)<=3:
            return [eval("".join(input))]
        i=1
        res=[]
        while i<len(input):
            left=self.diffWaysToCompute(input[:i])
            right=self.diffWaysToCompute(input[i+1:])
            for left_num in left:
                for right_num in right:
                    res.append(eval(str(left_num)+input[i]+str(right_num)))
            i+=2    
        return sorted(res)

    def parse(self,input):
        res=[]
        cur=""
        i=0
        while i<len(input):
            if input[i].isdigit():
                cur+=input[i]
                i+=1
            else:
                res.append(cur)
                cur=""
                res.append(input[i])
                i+=1
        if cur!="":
            res.append(cur)
        return res
solution=Solution()
# print(solution.diffWaysToCompute("2*3-4*5"))

        
class Solution(object):
    def find(self,money):
        def dfs(s,j,cur,result):
            if sum(cur)>money:
                return
            if s>money:
                return
            if s==money:
                result.append([cur.count(10),cur.count(5),cur.count(2)])
                return
            for i in range(j,3):
                dfs(s+items[i],i,cur+[items[i]],result)
        result=[]
        items=[10,5,2]
        dfs(0,0,[],result)
        return result
solution=Solution()
# print(solution.find(100))
#250. Count Univalue Subtrees(Medium)
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    def countUnivalSubtrees(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.count = 0
        def dfs(node, parent_val):
            if not node:
                return True
            l = dfs(node.left, node.val)
            r = dfs(node.right, node.val)
            if l and r:
                self.count += 1
            return l and r and node.val == parent_val
        
        dfs(root, None)
        return self.count