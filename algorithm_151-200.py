#151. Reverse Words in a String(Medium)
class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        return " ".join(reversed(s.split()))

#152. Maximum Product Subarray(Medium)
class Solution(object):
    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        f, g = [], []
        f.append(nums[0])
        g.append(nums[0])
        for i in range(1, len(nums)):
            f.append(max(f[i-1]*nums[i], g[i-1]*nums[i], nums[i]))
            g.append(min(f[i-1]*nums[i], g[i-1]*nums[i], nums[i]))
        m = f[0]
        print(f,g)
        for i in range(1, len(f)): m = max(m, f[i])
        return m
solution=Solution()
# print(solution.maxProduct([-1,-2,-3,4]))

#153. Find Minimum in Rotated Sorted Array(Medium)
class Solution(object):
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        start,end=0,len(nums)-1
        while start<end-1:
            target=nums[end]
            mid=(start+end)//2
            if nums[mid]<target:
                end=mid
            else:
                start=mid
        return min(nums[start],nums[end])
#154. Find Minimum in Rotated Sorted Array II(Hard)
class Solution(object):
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        start,end=0,len(nums)-1
        while start<end-1:
            target=nums[end]
            mid=(start+end)//2
            if nums[mid]<target:
                end=mid
            elif nums[mid]>target:
                start=mid
            else:
                end-=1
        return min(nums[start],nums[end])
solution=Solution()
# print(solution.findMin([4,5,6,6,6,7,8,1,2,3,3,3]))
#155. Min Stack(Easy)

class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack=[]
        self.min_stack=[]
        

    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        self.stack.append(x)
        if self.min_stack==[] or x<=self.min_stack[-1]:
            self.min_stack.append(x)

    def pop(self):
        """
        :rtype: void
        """
        if self.stack==[]:
            return None
        if self.stack[-1]==self.min_stack[-1]:
            self.min_stack.pop()
        return self.stack.pop()

    def top(self):
        """
        :rtype: int
        """
        if self.stack==[]:
            return -1
        return self.stack[-1]

    def getMin(self):
        """
        :rtype: int
        """
        if self.min_stack==[]:
            return -1
        return self.min_stack[-1]

#159. Longest Substring with At Most Two Distinct Characters(Hard)
class Solution:
    def lengthOfLongestSubstringTwoDistinct(self, s):
        """
        :type s: str
        :rtype: int
        """
        hsh_count = {}
        left,res = 0,0
        for right in range(len(s)):
            if s[right] in hsh_count:
                hsh_count[s[right]]+=1
            else:
                hsh_count[s[right]]=1
            while len(hsh_count)>2:
                hsh_count[s[left]]-=1
                if not hsh_count[s[left]]:
                    del hsh_count[s[left]]
                left+=1
            res=max(res,right-left+1)
        return res

# solution=Solution()
# print("==================================")
# print(solution.lengthOfLongestSubstringTwoDistinct('ccaabbb'))


#160. Intersection of Two Linked Lists(Easy)
class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        if headA==None or headB==None:
            return None
        node=headA
        while node.next!=None:
            node=node.next
        node.next=headB
        result=self.cycle(headA)
        node.next=None
        return result
    def cycle(self,head):
        if head==None:
            return None
        slow=head
        fast=head.next
        while slow!=fast:
            if fast.next == None or fast.next.next==None:
                return None
            slow=slow.next
            fast=fast.next.next
        while head!=slow.next:
            head=head.next
            slow=slow.next
        return head

#162. Find Peak Element(Medium)
class Solution(object):
    def findPeakElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums)==1:
            return 0
        start,end=0,len(nums)-1
        while start<end:
            mid=(start+end)//2
            # if mid!=0 and mid!=len(nums)-1 and (nums[mid]>nums[mid+1] and nums[mid]>nums[mid-1]) :
            #     return mid
            if mid!=len(nums)-1 and nums[mid]>nums[mid+1]:
                end=mid
            elif mid!=len(nums)-1 and nums[mid]<nums[mid+1]:
                start=mid+1
        return start
solution=Solution()
# print(solution.findPeakElement([1,0]))

#164. Maximum Gap(Hard)
class Solution(object):
    def maximumGap(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n=len(nums)
        if n<2:
            return 0
        maximum=max(nums)
        minimum=min(nums)
        bins=max(1,int((maximum-minimum)/n)+1)
        l=(maximum-minimum)//bins+1
        buckets=[None]*l
        for m in nums:
            index=(m-minimum)//bins
            bucket=buckets[index]
            if bucket==None:
                bucket=[m,m]
            else:
                bucket[0]=min(bucket[0],m)
                bucket[1]=max(bucket[1],m)
            buckets[index]=bucket
        res=0
        target_min=None
        target_max=None
        print(buckets)
        for i in range(l):
            if buckets[i]==None:
                continue
            if target_max==None:
                target_max=buckets[i][1]
            else:
                target_min=buckets[i][0]
                res=max(res,target_min-target_max)
                target_max=buckets[i][1]
        return res
# solution=Solution()
# print(solution.maximumGap([1,2,7,8,20]))
# print(solution.maximumGap([1,1,1,1]))

#165. Compare Version Numbers(Medium)
class Solution(object):
    def compareVersion(self, version1, version2):
        """
        :type version1: str
        :type version2: str
        :rtype: int
        """
        v1=version1.split(".")
        v2=version2.split(".")
        for i in range(max(len(v1),len(v2))):
            if i<len(v1) and i<len(v2):
                if int(v1[i])<int(v2[i]):
                    return -1
                elif int(v1[i])>int(v2[i]):
                    return 1
            else:
                if i<len(v2):
                    if int(v2[i])!=0:
                        return -1
                else:
                    if int(v1[i])!=0:
                        return 1
        return 0
#166. Fraction to Recurring Decimal(Medium)
class Solution(object):
    def fractionToDecimal(self, numerator, denominator):
        """
        :type numerator: int
        :type denominator: int
        :rtype: str
        """
        if numerator*denominator>=0:
            sign=""
        else:
            sign="-"
        numerator=abs(numerator)
        denominator=abs(denominator)
        integer=["0"]
        fraction=[]
        stack=[]
        if numerator>=denominator:
            integer=[str(numerator//denominator)]
            numerator=(numerator%denominator)*10
        else:
            numerator=numerator*10
        loop_index=None
        while numerator!=0:
            if str(numerator) not in stack:
                stack.append(str(numerator))
                fraction.append(str(numerator//denominator))
                numerator=(numerator%denominator)*10
            else:
                loop_index=stack.index(str(numerator))
                fraction.append(")")
                break
        if loop_index!=None:
            fraction.insert(loop_index,"(")
        if fraction:
            return sign+"".join(integer)+"."+"".join(fraction)
        else:
            return sign+"".join(integer)
solution=Solution()
print(solution.fractionToDecimal(1,17))

#167.Two Sum II - Input array is sorted(Easy)
class Solution(object):
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        start,end=0,len(numbers)-1
        while start<end:
            if numbers[start]+numbers[end]<target:
                start+=1
            elif numbers[start]+numbers[end]>target:
                end-=1
            else:
                return [start+1,end+1]
        return []

#168. Excel Sheet Column Title(Easy)
class Solution(object):
    def convertToTitle(self, n):
        """
        :type n: int
        :rtype: str
        """
        res=""
        while n!=0:
            target=(n-1)%26
            res=chr(target+65)+res
            n=(n-1)//26
        return res
solution=Solution()
# print(solution.convertToTitle(27))

#169. Majority Element(Easy)
class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # return sorted(nums)[len(nums)//2]
        candidate = None
        count = 0
        for num in nums:
            if count == 0:
                candidate = num
                count += 1
            elif candidate == num:
                count += 1
            else:
                count -= 1
        return candidate
solution=Solution()
# print(solution.majorityElement([1,3,1,3,1,3,1,3,1,3]),"xxxxxxxxxxxxxxxxxxxxxx")

#171. Excel Sheet Column Number(Easy)
class Solution(object):
    def titleToNumber(self, s):
        """
        :type s: str
        :rtype: int
        """
        res=0
        for i in reversed(range(len(s))):
            res+=(ord(s[i])-64)*pow(26,(len(s)-i-1))
        return res
#172. Factorial Trailing Zeroes(Easy)
#class Solution(object):
    def trailingZeroes(self, n):
        """
        :type n: int
        :rtype: int
        """
        ans = 0
        while n >= 5:
            ans += n //5
            n=n//5
        return ans
    
solution=Solution()
# print(solution.trailingZeroes(128))

#173. Binary Search Tree Iterator(Medium)
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class BSTIterator(object):
    def __init__(self, root):
        """
        :type root: TreeNode
        """
        self.cur=root
        self.stack=[]

    def hasNext(self):
        """
        :rtype: bool
        """
        return self.stack!=[] or self.cur.right!=None

    def next(self):
        """
        :rtype: int
        """
        while self.cur!=None:
            self.stack.append(self.cur)
            self.cur=self.cur.left
        self.cur=self.stack.pop()
        res=self.cur.val
        self.cur=self.cur.right
        return res

#174. Dungeon Game(Hard)
class Solution(object):
    def calculateMinimumHP(self, dungeon):
        """
        :type dungeon: List[List[int]]
        :rtype: int
        """
        m=len(dungeon)
        n=len(dungeon[0])
        dp=[[0]*n for i in range(m)]
        for i in reversed(range(m)):
            for j in reversed(range(n)):
                if i==m-1 and j==n-1:
                    dp[i][j]=1-dungeon[i][j] if 1-dungeon[i][j]>0 else 1
                elif i==m-1:
                    target=dp[i][j+1]-dungeon[i][j]
                    dp[i][j]=target if target>0 else 1
                elif j==n-1:
                    target=dp[i+1][j]-dungeon[i][j]
                    dp[i][j]=target if target>0 else 1
                else:
                    target=min(dp[i][j+1],dp[i+1][j])-dungeon[i][j]
                    dp[i][j]=target if target>0 else 1
        return dp[0][0]
solution=Solution()
# print(solution.calculateMinimumHP([[1],[-2],[1]]))

#179. Largest Number(Medium)
class Solution:
    # @param {integer[]} nums
    # @return {string}
    def largestNumber(self, nums):
        nums=[str(n) for n in nums]
        from functools import cmp_to_key
        comp=cmp_to_key(lambda x,y: 1 if int(x+y)-int(y+x)<0 else -1)
        nums=sorted(nums,key=comp)
        res="".join(nums).lstrip("0")
        return res or "0"
solution=Solution()
# print(solution.largestNumber([0,0,0,0]))


#186. Reverse Words in a String II(Medium)
class Solution:
    def reverseWords(self, str):
        """
        :type str: List[str]
        :rtype: void Do not return anything, modify str in-place instead.
        """
        str[::] = str[::-1]
        indexes = [-1]+[j for j,c in enumerate(str) if c==" "]+[len(str)+1]
        for i in range(len(indexes)-1):
            str[indexes[i]+1:indexes[i+1]] = str[indexes[i]+1:indexes[i+1]][::-1]
        
            

#187. Repeated DNA Sequences(Medium)
class Solution(object):
    def findRepeatedDnaSequences(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        seen=set()
        res=[]
        for i in range(len(s)-9):
            DNA=s[i:i+10]
            if DNA in seen:
                if DNA not in res:
                    res.append(DNA)
            else:
                seen.add(DNA)
        return res
solution=Solution()
# print(solution.findRepeatedDnaSequences("AAAAAAAAAAA"))

#188. Best Time to Buy and Sell Stock IV(Hard)
class Solution(object):
    #2D Dynamic Programming
    def maxProfit(self, k, prices):
        """
        :type k: int
        :type prices: List[int]
        :rtype: int
        """
        if prices==[]:
            return 0
        if k==0:
            return 0
        if k>=len(prices)//2:
            return self.quick(prices)
        dp=[[0]*len(prices) for i in range(k)]
        for i in range(1,k):
            max_dif=dp[i-1][0]-prices[0]
            for j in range(len(prices)):
                max_dif=max(max_dif,dp[i-1][j]-prices[j])
                dp[i][j]=max(dp[i-1][j],max_dif+prices[j])
        return dp[-1][-1]

    #1D Dynamic Programming
    def maxProfit(self, k, prices):
        """
        :type k: int
        :type prices: List[int]
        :rtype: int
        """
        if prices==[]:
            return 0
        if k==0:
            return 0
        if k>=len(prices)//2:
            return self.quick(prices)
        size=len(prices)
        dp = [None] * (2 * k + 1)
        dp[0] = 0
        for i in range(size):
            for j in range(1,min(2 * k, i + 1)+1):
                if dp[j]==None:
                    dp[j]=dp[j - 1] + prices[i] * [1, -1][j % 2]
                else:
                    dp[j] = max(dp[j], dp[j - 1] + prices[i] * [1, -1][j % 2])
        return dp[-1]
    def quick(self,prices):
        profit=0
        for i in range(1,len(prices)):
            di=prices[i]-prices[i-1]
            if di>0:
                profit+=di
        return profit
solution=Solution()
# print(solution.maxProfit(2,[1,2,4,2,5,7,2,4,9]))
#189. Rotate Array(Easy)
class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        k=k%len(nums)
        temp=nums[len(nums)-k:]
        print(temp)
        nums[k+1:]=nums[:len(nums)-k]
        nums[:k+1]=temp
        return nums
solution=Solution()
# print(solution.rotate([1,2,3,4,5,6,7],3))

#190. Reverse Bits(Easy)
class Solution:
    # @param n, an integer
    # @return an integer
    def reverseBits(self, n):
        res=0
        mask=1
        for i in range(32):
            res+=((n>>i)&1)<<(31-i)
        return res
solution=Solution()
# print(solution.reverseBits(43261596))

#191. Number of 1 Bits(Easy)
class Solution(object):
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        # res=0
        # while n!=0:
        #     if n%2==1:
        #         res+=1
        #     n=n//2
        # return res
        if n==0:
            return 0
        count=1
        while n&(n-1)!=0:
            n&=(n-1)
            count+=1
        return count

solution=Solution()
# print(solution.hammingWeight(7))

#198. House Robber(Easy)
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        t=[0]*len(nums)#the maximal value when u pick this house
        f=[0]*len(nums)#the maximal value when u don't pick this house
        t[0]=nums[0]
        for i in range(1,len(nums)):
            t[i]=f[i-1]+nums[i]
            f[i]=max(t[i-1],f[i-1])
        return max(f[-1],t[-1])

#199. Binary Tree Right Side View(Medium)
class Solution(object):
    def rightSideView(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        self.dic={}
        self.traversal(root,0)
        res=[]
        for i in range(len(self.dic)):
            res.append(self.dic)
        return res

    def traversal(self,root,depth):
        if root is None:
            return 
        self.traversal(root.left,depth+1)
        if depth in self.dic:
            self.dic[depth]=root.val
        self.traversal(root.right,depth+1)

#200. Number of Islands(Medium)
class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        if grid==[]:
            return 0
        i,j=0,0
        self.seen=set()
        count=1
        while i<len(grid):
            if (i,j) not in self.seen and grid[i][j]=="1":
                self.fill(i,j,grid,count)
                count+=1
            j+=1
            if j>=len(grid[0]):
                i+=1
                j=0
        return count-1
    def fill(self,i,j,grid,count):
        if not self.isValid(grid,i,j):
            return
        else:
            self.seen.add((i,j))
            grid[i][j]=count
            self.fill(i+1,j,grid,count)
            self.fill(i,j+1,grid,count)
            self.fill(i,j-1,grid,count)
            self.fill(i-1,j,grid,count)
    def isValid(self,grid,i,j):
        if (i,j) in self.seen:
            return False
        if i<0 or i>=len(grid):
            return False
        if j<0 or j>=len(grid[0]):
            return False
        if grid[i][j]=="0":
            return False
        return True
solution=Solution()
grid=[["1","1","0","0","0"],
      ["1","1","0","0","0"],
      ["0","0","1","0","0"],
      ["0","0","0","1","1"]]
# print(solution.numIslands(grid))
# print(grid)