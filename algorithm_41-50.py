#41. First Missing Positive(Hard)
class Solution(object):
    def firstMissingPositive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        lo=0
        if n == 0:
            return 1
        while lo<n:
            while(nums[lo]!=lo+1 and nums[lo]>0 and nums[lo]<=n and nums[lo]!=nums[nums[lo]-1]):
                t=nums[lo]
                nums[lo]=nums[t-1]
                nums[t-1]=t
            lo+=1
        print(nums)
        for i in range(len(nums)):
            if nums[i]!=i+1:
                return i+1
        return n+1
        
solution=Solution()
#42. Trapping Rain Water(Hard)
class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        if len(height)<3:
            return 0
        res=0
        left_max=[0]*len(height)
        right_max=[0]*len(height)
        for i in range(1,len(height)):
            left_max[i]=max(left_max[i-1],height[i-1])
        for i in range(len(height)-2,-1,-1):
            right_max[i] = max(right_max[i+1],height[i+1])
        for i in range(len(height)):
            capacity = min(left_max[i],right_max[i])
            if capacity>height[i]:
                res+=capacity-height[i]
        return res

    def trap(self,height):
        """
        :type height: List[int]
        :rtype: int
        """
        left,right = 0,len(height)-1
        l_wall,r_wall = 0,0
        res = 0
        while left<right:
            if height[left]<height[right]:
                if height[left]<l_wall:
                    res+= l_wall-height[left]
                else:
                    l_wall = height[left]
                left+=1
            else:
                if height[right]<r_wall:
                    res+=r_wall-height[right]
                else:
                    r_wall = height[right]
                right-=1
        return res

# print(solution.trap([0,1,0,2,1,0,1,3,2,1,2,1]))
#43. Multiply Strings(Medium)
class Solution(object):
    def multiply(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        n,m=len(num1),len(num2)
        res=[0]*(m+n)
        num1,num2=num1[::-1],num2[::-1]
        carry=0
        for i in range(n): 
            carry=0
            for j in range(m):
                prod=res[i+j]+(ord(num1[i])-ord('0'))*(ord(num2[j])-ord('0'))+carry
                res[i+j]=(prod)%10
                carry=(prod)//10
            if (carry!=0):
                res[i+j+1]=res[i+j+1]+carry
        res=res[::-1]
        i=0
        while(res[i]==0 and i!=len(res)-1):
            res=res[1:]
        for i in range(len(res)):
            res[i]=chr(res[i]+ord('0'))
        return "".join(res)

#44. Wildcard Matching(Hard)
class Solution(object):
    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        dp_table=[[False for i in range(len(p)+1)]for j in range(len(s)+1)]
        dp_table[0][0]=True
        for i in range(len(p)):
            if p[i]=="*":
                dp_table[0][i+1]=dp_table[0][i]
        for i in range(1,len(s)+1):
            for j in range(1,len(p)+1):
                if p[j-1]==s[i-1] or p[j-1]=="?":
                    dp_table[i][j]=dp_table[i-1][j-1]
                else:
                    if p[j-1]=="*":
                        dp_table[i][j]=(dp_table[i][j-1] or dp_table[i-1][j])
        return dp_table[len(s)][len(p)]

solution=Solution()
print(solution.isMatch("",""))

#45 Jump Game II(Hard)
class Solution(object):
    def jump(self,nums):
        p = [0]*len(nums)
        lo=1
        for i in range(len(nums) - 1):
            while(i + nums[i] >= lo and lo < len(nums)):
                p[lo]=p[i]+1
                lo+=1
        return p[-1]
    def jump_version2(self,nums):
        cur_max = 0
        jump=0
        start=0
        while(cur_max<len(nums)-1):
            last_max = cur_max
            for i in range(start,last_max+1):
                cur_max=max(cur_max,i+nums[i])
            jump+=1
        return jump
solution=Solution()
# print(solution.jump_version2([3,1,1,2,5,3,2,0,1,1]))

#46. Permutations(Medium)
class Solution(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if len(nums)==0:
            return [[]]
        else:
            res=[]
            for subs in self.permute(nums[1:]):
                for i in range(len(subs)+1):
                    target=subs[0:i]+[nums[0]]+subs[i:]
                    if target not in res:
                        res.append(target)
        return res

    #Backtracking no need to check duplication
    def permute_(self,nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        n = len(nums)
        if not n:return [[]]
        def backtracking(current,nums,res):
            if len(current)==n:
                res.append(current)
                return
            num = nums[0]
            for i in range(len(current)+1):
                backtracking(current[:i]+[num]+current[i:],nums[1:],res)
            return res
        res = []
        backtracking([],nums,res)
        return res

s = Solution()
print(s.permute_([1,2,3]))

#47. Permutations II (Medium)
class Solution(object): 
    def permuteUnique(self, nums): # BruteForce Dedpulication
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if len(nums) <= 1: 
            return [nums]
        res = set()
        result = []
        item = nums[0]
        sub_permutation = self.permuteUnique(nums[1:])
        for permute in sub_permutation :
            for i in range(len(permute)+1):
                target = permute[0:i] + [item] + permute[i:]
                if tuple(target) not in res:
                    res.add(tuple(target))
                    result.append(target)

        return result


    def permuteUnique_II(self, nums): # Tree Dedpulication
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        result = []
        def solve(nums, current):
            if not nums:
                result.append(current)
                return
            for i in range(len(nums)):
                if i > 0 and nums[i] == nums[i-1]: continue
                target = nums[i]
                solve(nums[0:i]+nums[i+1:], current+[target])
            
        nums.sort()
        solve(nums, [])
        return result


#48. Rotate Image(Medium)
class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        rows=len(matrix)
        cols=len(matrix[0])
        for i in range(rows):
            for j in range(i+1,cols):
                matrix[i][j],matrix[j][i]=matrix[j][i],matrix[i][j]
        for i in range(rows):
            matrix[i]=matrix[i][::-1]

#49. Group Anagrams(Medium)
class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        dic={}
        for word in strs:
            key="".join(sorted(word))
            if key in dic:
                dic[key].append(word)
            else:
                dic[key]=[word]
        return list(dic.values())

#50. Pow(x, n)(Medium)
class Solution(object):
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        if n == 0:
            return 1
        else:
            temp=n
            n=abs(n)
            half = self.myPow(x,n//2)
            if(n%2 == 0):
                res = half*half
            else:
                res = half*half*x
            if temp<0:
                res=1/res
            return res
solution=Solution()