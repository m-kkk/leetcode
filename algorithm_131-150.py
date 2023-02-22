#131. Palindrome Partitioning
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
# solution=Solution()
# print(solution.minCut("leet"))


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
            if node!=None:
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
            # print(dup,(points[i].x,points[i].y),dic)
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
            # print(stack)
        return int(stack[0])

solution=Solution()
# print(solution.evalRPN(["10","6","9","3","+","-11","*","/","*","17","+","5","+"]))
# print(solution.evalRPN(["4", "13", "5", "/", "+"]))



class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def printList(head):
    while head!=None:
        print(head.val , '->', end = '')
        head = head.next
    print ('\n')

class Solution:
    def sortList(self, head):
        if head == None or head.next == None:
            return head
        left , right = self.half(head)
        
        printList(self.merge(left, right)) 

    def half(self, head):
        fast, slow = ListNode(), ListNode()
        fast.next, slow.next = head, head
        while fast != None and fast.next != None:
            slow = slow.next
            fast = fast.next.next
        first = head
        second = slow.next
        slow.next = None
        return first, second

    def merge(self, left, right):
        dummy = ListNode()
        p = dummy 
        while left != None or right != None:
            print('left:', end= '')
            printList(left)
            print('right:', end= '')
            printList(right)    
            if left == None or right == None:
                p.next = left if left else right
                break
            if left.val <= right.val:
                p.next = left
                left = left.next
            else:
                p.next = right
                right = right.next
            p = p.next
        return dummy.next

solution=Solution()
p = ListNode(1)
p.next = ListNode(2)
p.next.next = ListNode(3)
p.next.next.next = ListNode(4)

solution.sortList(p)