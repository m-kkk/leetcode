# 11. Container With Most Water(Medium)
class Solution(object):
    def maxArea(self, height):
        lo = 0
        hi = len(height)-1
        mi = min(height[lo],height[hi])
        ma = max(height[lo],height[hi])
        max_area = mi*(hi-lo)
        while lo!=hi:
            if(height[lo]==mi):
                lo+=1
            elif(height[hi]==mi):
                hi-=1
            mi = min(height[lo],height[hi])
            area = mi*(hi-lo)
            if area > max_area:
                max_area = area
        return max_area
        
#12. Integer to Roman(Medium)
class Solution(object):
    def translate(self, num, one, five,ten):
        dic = ["",one,one*2,one*3,one+five,five,five+one,five+one*2,five+one*3,one+ten]
        return dic[num]
    def intToRoman(self, num):
        units = num%10
        tens = (num//10)%10
        hundreds = (num//100)%10 
        thousands = (num//1000)%10
        return self.translate(thousands,"M","M","M")+self.translate(hundreds,"C","D","M")+self.translate(tens,"X","L","C")+self.translate(units,"I","V","X")

#13. Roman to Integer(Easy)
class Solution(object):
    def romanToInt(self, s):
        integer = 0
        for i in range(len(s)):
            if s[i]=="I":
                if "V" in s[i:] or "X" in s[i:]:
                    integer -= 1
                else:
                    integer += 1
            if s[i] == "X":
                if "C" in s[i:] or "L" in s[i:]:
                    integer -= 10
                else:
                    integer += 10
            if s[i] == "C":
                if "D" in s[i:] or "M" in s[i:]:
                    integer -= 100
                else:
                    integer += 100
            if s[i] == "M":
                integer += 1000
            if s[i] == "D":
                integer += 500
            if s[i] == "L":
                integer += 50
            if s[i] =="V":
                integer += 5
        return integer

#14. Longest Common Prefix(Easy)
class Solution(object):
    def common_prefix(self,a,b):
        if a=="" or b=="":
            return ""
        for i in range(min(len(a),len(b))):
            if a[i]!=b[i]:
                return a[0:i]
        return a[0:i+1]
    def longestCommonPrefix(self, strs):
        if strs == []:
            return ""
        if len(strs) == 1:
            return strs[0]
        compre=self.common_prefix(strs[0],strs[1])
        for i in range(2,len(strs)):
            compre=self.common_prefix(compre,strs[i])
            if compre =="":
                return ""
        return compre

#15. 3Sum(Medium)
# class Solution(object):
#     def find_sum(self, nums, target):
#         result = []
#         for i in range(len(nums)):
#             for j in range(len(nums)):
#                 if nums[i]+nums[j] == target and i != j:
#                     result.append((nums[i],nums[j]))
#         return result
#     def permu_in (self,L,l):
#         for i in range(len(l)):
#             for j in range(1,len(l)):
#                 element = [l[i],l[(i+j)%len(l)],l[(i+2*j)%len(l)]]
#                 if element in L:
#                     return True
#         return False
#     def threeSum(self, nums):
#         result = []
#         for i in range(len(nums)):
#             trial = self.find_sum(nums[0:i]+nums[i+1:],-nums[i])
#             if trial != []:
#                 for j in range(len(trial)):
#                     element =[nums[i],trial[j][0],trial[j][1]]
#                     if not self.permu_in(result,element):
#                         result.append(element)
#         return result
class Solution(object):
    def find_sum(self, nums, target):
        lo = 0
        hi = len(nums)-1
        result=[]
        while lo<hi:
            if nums[lo]+nums[hi] < target:
                lo+=1
            elif nums[lo]+nums[hi] > target:
                hi-=1
            else:
                result.append([nums[lo],nums[hi]])
                lo+=1
                hi-=1
                while lo<hi and nums[lo]==nums[lo-1]:
                    lo+=1
                while lo<hi and nums[hi]==nums[hi+1]:
                    hi-=1
        return result

    def threeSum(self, nums):
        if len(nums)<3:
            return []
        result = []
        nums = sorted(nums)
        for i in range(len(nums)-1):
            if i!=0 and nums[i]==nums[i-1]:
                continue
            trial = self.find_sum(nums[i+1:],-nums[i])
            if trial != []:
                for j in range(len(trial)):
                    element =[nums[i],trial[j][0],trial[j][1]]
                    result.append(element)
        return result
solution=Solution()
print(solution.threeSum([-1, 0, 1, 2, -1, -4,0,0]))

#16. 3Sum Closest(Medium)
class Solution(object):
    def find_sum(self, nums, target):
        lo = 0
        hi = len(nums)-1
        diff =None
        while lo!=hi:
            di = abs(nums[lo]+nums[hi]-target)
            if diff==None or diff>di:
                diff = di
                result =(nums[lo],nums[hi])
            if nums[lo]+nums[hi] <= target:
                lo+=1
            elif nums[lo]+nums[hi] > target:
                hi-=1
        if diff==None:
            return sum(nums[lo],nums[hi])
        return sum(result)
    def threeSumClosest(self, nums, target):
        if len(nums)==3:
            return sum(nums)
        diff = None
        nums = sorted(nums)
        for i in range(len(nums)-2):
            aim = target-nums[i]
            di = self.find_sum(nums[i+1:],aim)
            if diff == None or abs(nums[i]+di-target) <diff:
                diff = abs(nums[i]+di-target)
                result = nums[i]+di
        return result

#17. Letter Combinations of a Phone Number(Medium)
class Solution(object):
    def letterCombinations(self, digits):
        dic = {"2":"abc","3":"def","4":"ghi","5":"jkl","6":"mno","7":"pqrs","8":"tuv","9":"wxyz","0":" "}
        result = [""]
        temp = []
        for i in range(len(digits)):
            alphbet = dic[digits[i]]
            for j in range(len(alphbet)):
                for s in result:
                    temp.append(s+alphbet[j])
            result = temp
            temp = []
        if result == [""]:
            return []
        return result
#18. 4Sum(Medium)
class Solution(object):
    def threeSum(self, nums, target,piv):
        result=[]
        for i in range(len(nums)-2):
            lo=i+1
            hi=len(nums)-1
            while(lo!=hi):
                if nums[lo]+nums[hi]+nums[i] == target:
                    if([piv,nums[i],nums[lo],nums[hi]] not in result):
                        result.append([piv,nums[i],nums[lo],nums[hi]])
                    lo+=1
                elif nums[lo]+nums[hi]+nums[i] > target:
                    hi-=1
                elif nums[lo]+nums[hi]+nums[i] < target:
                    lo+=1
        return result
    def fourSum(self, nums, target):
        if len(nums)<4:
            return []
        nums=sorted(nums)
        result = []
        for i in range(len(nums)-3):
            aim = target-nums[i]
            element = self.threeSum(nums[i+1:],aim,nums[i])
            for s in element:
                if s not in result:
                    result.append(s)
        return result

#19. Remove Nth Node From End of List(Easy)
# Definition for singly-linked list.
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
L =ListNode(1)
# L.next= ListNode(2)
# L.next.next= ListNode(3)
# L.next.next.next= ListNode(4)

class Solution(object):
    def removeNthFromEnd(self, head, n):
        temp=head
        res=head
        for i in range(0, n):
            head = head.next
        if(head==None):
            return res.next
        while head.next != None:
            head = head.next
            temp = temp.next
        tempult = temp.next
        temp.next = temp.next.next
        return res

solution = Solution()
# print(solution.removeNthFromEnd(L,4))
#20. Valid Parentheses(Easy)
class Solution(object):
    def isValid(self, s):
        stack = [0]
        for i in range(len(s)):
            if s[i]=="(":
                stack.append(1)
            if s[i]=="[":
                stack.append(2)
            if s[i]=="{":
                stack.append(3)
            if s[i]==")":
                if stack.pop()!=1:
                    return False
            if s[i]=="]":
                if stack.pop()!=2:
                    return False
            if s[i]=="}":
                if stack.pop()!=3:
                    return False
        if stack.pop() != 0:
            return False
        return True
                

