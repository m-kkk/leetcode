#61. Rotate List
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
    def __repr__(self):
        return str(self.val)
    def print(self):
        element=self.next
        print(self,"->",end="")
        while element!=None:
            print(element,"->",end="")
            element=element.next
L=ListNode(1)
class Solution(object):
    def rotateRight(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        if k ==0 or head == None:
            return head
        temp=head
        res=head
        size=0
        while(head!=None):
            head=head.next
            size+=1
        k=k%size
        for i in range(k):
            head=head.next
        if head==None:
            return res
        while(head.next!=None):
            head=head.next
            res=res.next
        result = res.next
        res.next=None
        head.next=temp
        return result

#62. Unique Paths
class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        res=1
        for i in range(n-1):
            res*=m+n-2-i
            res/=(n-1-i)
        return round(res)

# print(solution.uniquePaths(6,4))
#63. Unique Paths II
class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        dp=obstacleGrid
        if dp[0][0]==1:
            return 0
        dp[0][0]=1
        for i in range(len(dp)):
            for j in range(len(dp[0])):
                if i==0 and j!=0:
                    if dp[i][j]==1:
                        dp[i][j]=0
                    else:
                        dp[i][j]=dp[i][j-1]
                if j==0 and i!=0:
                    if dp[i][j]==1:
                        dp[i][j]=0
                    else:
                        dp[i][j]=dp[i-1][j]
                if i!=0 and j!=0:
                    if dp[i][j]==1:
                        dp[i][j]=0
                    else:
                        dp[i][j]=dp[i][j-1]+dp[i-1][j]
        return dp[-1][-1]
#64. Minimum Path Sum
class Solution(object):
    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if i==0 and j!=0:
                    grid[i][j]=grid[i][j]+grid[i][j-1]
                if j==0 and i!=0:
                    grid[i][j]=grid[i][j]+grid[i-1][j]
                if i!=0 and j!=0:
                    grid[i][j]=grid[i][j]+min(grid[i][j-1],grid[i-1][j])
        return grid[-1][-1]
solution=Solution()
#65. Valid Number(Hard)
class Solution(object):
    def isNumber(self, s):
        """
        :type s: str
        :rtype: bool
        """
        #Deterministic Finite Automata
        if s=="":
            return False
        INVALID,SPACE,DIGIT,SIGN,EXP,DOT=0,1,2,3,4,5
        states_table=[[-1, 0, 2, 1,-1, 3],#0 initial space
                     [-1,-1, 2,-1,-1, 3],#1 Sign 
                     [-1, 8, 2,-1, 5, 4],#2 Digits
                     [-1,-1, 4,-1,-1,-1],#3 Dot before seen digits and dot
                     [-1, 8, 4,-1, 5,-1],#4 Dot after seen digits or dot
                     [-1,-1, 7, 6,-1,-1],#5 E or e
                     [-1,-1, 7,-1,-1,-1],#6 Digits after E or e
                     [-1, 8, 7,-1,-1,-1],#7 sign after E or e
                     [-1, 8,-1,-1,-1,-1]]#8 space after valid number
        state=0
        for i in range(len(s)):
            parse=INVALID
            if s[i].isdigit():
                parse=DIGIT
            elif s[i]==" ":
                parse=SPACE
            elif s[i]=="+" or s[i]=="-":
                parse=SIGN
            elif s[i]==".":
                parse=DOT
            elif s[i]=="E" or s[i]=="e":
                parse=EXP
            state = states_table[state][parse]
            if state==-1:
                return False
        return (state==8 or state==7 or state==2 or state==4)
            
solution=Solution()
# print(solution.isNumber("."))
#66. Plus One(Easy)
class Solution(object):
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        carry=1
        lo=0
        while(carry!=0 and lo!=len(digits)):
            n=digits[-1-lo]+carry
            carry=n//10
            digits[-1-lo]=n%10
            lo+=1
        if carry==0:
            return digits
        else:
            return [1]+digits

#67. Add Binary()
class Solution(object):
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        n,m=len(a),len(b)
        l=max(m,n)+1
        carry=0
        res=[0]*l
        for i in range(l):
            if i < min(m,n):
                res[-1-i]=str((carry+int(a[-1-i])+int(b[-1-i]))%2)
                carry=(carry+int(a[-1-i])+int(b[-1-i]))//2
            elif i<max(m,n):
                s=a if len(a)>len(b) else b
                res[-1-i]=str((carry+int(s[-1-i]))%2)
                carry=(carry+int(s[-1-i]))//2
            else:
                res[-1-i]=str(carry)
        if res[0]=='0':
            res=res[1:]
        res="".join(res)
        return res
#68. Text Justification(Hard)
class Solution(object):
    def fullJustify(self, words, maxWidth):
        """
        :type words: List[str]
        :type maxWidth: int
        :rtype: List[str]
        """
        if words==[]:
            return []
        res,current=[],[]
        space=maxWidth
        while(words!=[]):
            while(space>=len(words[0])):
                space=space-len(words[0])
                current.append(words[0])
                words=words[1:]
                if ((words==[] and space>0) or (words!=[] and space>=len(words[0])) or (space>0 and " "not in current)):
                    space-=1
                    current.append(" ")
                if words==[]:
                    break
            if current[-1]==" " and len(current)>2:
                current=current[:-1]
                space+=1
            i=0
            while space>0:
                if words!=[]:
                    if current[i%len(current)].isspace():
                        current[i%len(current)]=current[i%len(current)]+" "
                        space-=1
                    i+=1
                else:
                    current[-1]+=" "
                    space-=1
            res.append("".join(current))
            space=maxWidth
            current=[]
        return res


#69. Sqrt(x)(Medium)
class Solution(object):
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x==1:
            return 1
        lo,hi=1,x
        while(lo<hi):
            mid=lo+(hi-lo)//2
            if mid*mid==x:
                return mid
            elif mid*mid<x:
                lo=mid+1
            elif mid*mid>x:
                hi=mid
        return lo-1
#70. Climbing Stairs(Easy)
class Solution(object):
    dic={}
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n in self.dic:
            return self.dic[n]
        else:
            if n==0:
                return 0
            elif n==1:
                return 1
            elif n==2:
                return 2
            else:
                self.dic[n]=self.climbStairs(n-1)+self.climbStairs(n-2)
                return self.dic[n]
solution=Solution()
print(solution.climbStairs(35))



solution=Solution()