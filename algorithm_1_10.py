#1. Two Sum(Easy)
class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        hsh = {}
        for i,n in enumerate(nums):
            if (target-n) in hsh:
                return (i,hsh[target-n])
            hsh[n] = i

#3.Longest Substring Without Repeating Characters (M)
# O(n)
class Solution(object):
    #hashmap
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        from collections import defaultdict
        m = defaultdict(lambda:-1)
        res,left = 0,0
        for i in range(len(s)):
            if m[s[i]]==-1 or m[s[i]]<left:
                res = max(res,i-left+1)
            else:
                left = m[s[i]]+1
            m[s[i]]=i
        return res

    #sets
    def lengthOfLongestSubstring(s):
        if s=='':
            return 0
        i,j = 0,1
        while j <= len(s):
            if(len(set(s[i:j])) == len(s[i:j])):
                j+=1
            else:
                i+=1
                j+=1
        return j-i-1
                


#4.Median of Two Sorted Arrays(H)
#O(log(m+n))
class Solution(object):
    def findMedianSortedArrays(self,nums1, nums2):
        mid = (len(nums1)+len(nums2))//2
        if(len(nums1)+len(nums2))%2==1:
            return self.findK(nums1,nums2,mid)
        else:
            return (self.findK(nums1,nums2,mid-1)+self.findK(nums1,nums2,mid))/2
    def findK(self,nums1,nums2,k):
        if(len(nums1) == 0):
            return nums2[k]
        if(len(nums2) == 0):
            return nums1[k]
        if(k == 0):
            return min(nums1[0],nums2[0])
        if(k == 1):
            #Is ok to just sort a at most 4 items array with builtin tools right
            #return sorted(nums1[0:2]+nums2[0:2])[1]
            if(len(nums1)==1 and len(nums2)==1):
                return max(nums1[0],nums2[0])
            elif(len(nums1) != 1 and len(nums2) ==1):
                return min(max(nums1[0],nums2[0]),nums1[1])
            elif(len(nums1) == 1 and len(nums2) !=1):
                return min(max(nums1[0],nums2[0]),nums2[1])
            else: 
                return min(max(nums1[0],nums2[0]),min(nums1[1],nums2[1]))
        if(len(nums1) <= k//2):
            return self.findK(nums1,nums2[k//2:],k-k//2)
        if(len(nums2) <= k//2):
            return self.findK(nums1[k//2:],nums2,k-k//2)
        if(nums1[k//2] < nums2[k//2]):
            return self.findK(nums1[k//2:],nums2,k-k//2)
        if(nums1[k//2] >= nums2[k//2]):
            return self.findK(nums1,nums2[k//2:],k-k//2)
solution=Solution()
# print(solution.findMedianSortedArrays([10000],[10001]))

#5.Longest Palindromic Substring(M)
#O(n^2)
class Solution(object):
    #Expanding on every position
    def isPalindrome(self,s,left,right):
        while(left >= 0 and right<len(s) and s[left]==s[right]):
            left-=1
            right+=1
        return(left+1,right-1)
    def longestPalindrome(self, s):
        start,end = 0,0
        for i in range(len(s)):
            left,right = self.isPalindrome(s,i,i)
            len1 = right - left +1
            left,right = self.isPalindrome(s,i,i+1)
            len2 = right - left +1
            target = max(len1,len2)
            if(target > end-start+1):
                start = i - (target-1)//2
                end = i + target //2
        return s[start:end+1]
    #Dynamic programming
    def longestPalindrome(self, s):
        if not s:
            return ""
        dp=[[False]*len(s)for i in range(len(s))]
        for l in range(1,len(s)+1):
            for i in range(len(s)-l+1):
                if l==1:
                    dp[i][i+l-1]=True
                elif l==2:
                    dp[i][i+l-1]=(s[i]==s[i+l-1])
                else:
                    dp[i][i+l-1]=(s[i]==s[i+l-1] and dp[i+1][i+l-2])
        max_length=0
        # print(dp)
        for i in range(len(s)):
            for j in reversed(range(i,len(s))):
                if dp[i][j]:
                    if max_length<=(j-i):
                        max_length=j-i
                        res=s[i:j+1]
        return res
solution=Solution()
print(solution.longestPalindrome("tstitasaaasu"))

#6.ZigZag Conversion(Easy)
class Solution(object):
    def convert(self, s, numRows):
        if(numRows==1):
            return s
        result = ""
        for i in range(0,numRows):
            index = i
            if(i == 0 or i == numRows-1):
                result += s[i::2*numRows-2]
            else:
                if(index<len(s)):
                    result += s[index]
                while(index<len(s)):
                    index += 2*numRows-2-2*i
                    if(index<len(s)):
                        result += s[index]
                    index += 2*i
                    if(index<len(s)):
                        result +=s[index]
        return result

#7.Reverse Integer(Easy)
class Solution(object):
    def reverse(self, x):
        result = 0
        indicate = 1 if x>=0 else -1
        x = abs(x)
        while (x!=0):
            result*=10
            result+=x%10
            x=x//10
        return result*indicate if result<2147483648 else 0#overflow

#8. String to Integer (atoi)(Medium)
class Solution:
    def myAtoi(self, str):
        """
        :type str: str
        :rtype: int
        """
        str=str.strip()
        start=None
        for i,c in enumerate(str):
            if start!=None and (not c.isdigit()):
                digit = str[start:i]
                break
            if start ==None and (c == '-' or c=='+' or c.isdigit()):
                start = i
            elif start == None:
                digit = "0"
                break
        else:
            digit = str[start:]
        if not digit:
            return 0
        try :
            x = int(digit)
        except:
            return 0
        if x > 2**31-1:
            return 2**31-1
        if x < -(2**31):
            return -(2**31)
        return x




#9.Palindrome Number(Easy)
class Solution(object):
    def isPalindrome(self, x):
        if(x<0):
            return False
        elif(x<10):
            return True
        else:
            result = 0
            orig = x
            while(x!=0):
                result*=10
                result+=x%10
                x=x//10
            return result == orig

#10.Regular Expression Matching(Hard)
#dynamic programming table
class Solution(object):
    def isMatch(self, s, p):
        dp_table = [[False for i in range(0,len(p)+1)]for j in range(0,len(s)+1)]
        dp_table[0][0] = True
        for i in range(1,len(p)):
            if(p[i] == '*'):
                dp_table[0][i+1] = dp_table[0][i-1]
        for i in range(1,len(s)+1):
            for j in range(1,len(p)+1):
                if(p[j-1] == s[i-1] or p[j-1] == '.'):
                    dp_table[i][j] = dp_table[i-1][j-1]
                elif(p[j-1] == '*' and j != 1):
                    dp_table[i][j] = dp_table[i][j-2]
                    if(p[j-2] == s[i-1] or p[j-2] == '.'):
                        dp_table[i][j] = (dp_table[i-1][j] or dp_table[i][j])
        return dp_table[len(s)][len(p)]

solution = Solution()
print(solution.isMatch("",".*"))