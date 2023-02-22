# 20.Merge Two Sorted Lists(Easy)
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
L.next= ListNode(2)
L.next.next= ListNode(3)
L.next.next.next= ListNode(4)

class Solution(object):
    def mergeTwoLists(self, l1, l2):
    	res = ListNode(0)
    	tem = res
    	while (l1!=None and l2!=None):
    		if l1.val < l2.val:
    			res.next=l1
    			l1=l1.next
    		else:
    			res.next=l2
    			l2=l2.next
    		res=res.next
    	if l1==None:
    		res.next=l2
    	else:
    		res.next=l1
    	return tem.next

#23. Merge k Sorted Lists(Hard)
class Solution(object):
    def mergeTwoLists(self, l1, l2):
        res = ListNode(0)
        tem = res
        while (l1!=None and l2!=None):
            if l1.val < l2.val:
                res.next=l1
                l1=l1.next
            else:
                res.next=l2
                l2=l2.next
            res=res.next
        if l1==None:
            res.next=l2
        else:
            res.next=l1
        return tem.next
    def mergeKLists(self, lists):
        if lists==[]:
            return None
        lo=0
        step=1
        while step<len(lists):
            lo=0
            while lo+step<len(lists):
                lists[lo]=self.mergeTwoLists(lists[lo],lists[lo+step])
                lo+=2*step
            step=step*2
        return lists[0]

#22. Generate Parentheses(Medium)
class Solution(object):
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        if n == 0:
            return []
        temp=[""]
        result=[]
        while n>0:
            result=[]
            for i in range(len(temp)):
                s="()"+temp[i]
                if s not in result:
                    result.append(s)
                for j in range(len(temp[i])):
                    if(temp[i][j]=="("):
                        s=temp[i][0:j+1]+"()"+temp[i][j+1:]
                        if s not in result:
                            result.append(s)
            n-=1
            temp=result
        return result


#24. Swap Nodes in Pairs(Easy)
class Solution(object):
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head==None:
            return None
        if head.next==None:
            return head
        res=ListNode(0)
        res.next=head
        temp=res
        while(head!=None and head.next!=None):
            temp.next=head.next
            temp=head
            head=head.next.next
            temp.next.next=temp
            temp.next=head
        return res.next

M =ListNode(1)
M.next= ListNode(2)
M.next.next= ListNode(3)
M.next.next.next= ListNode(4)

#25. Reverse Nodes in k-Group(Hard)
class Solution(object):
    def reverse(self,start,end):
        res=ListNode(0)
        res.next = start
        while(res.next!=end):
            temp=start.next
            start.next=temp.next
            temp.next=res.next
            res.next=temp
        return end,start 
    def reverseKGroup(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        if head==None:
            return None
        res=ListNode(0)
        res.next=head
        temp=res
        start=head
        for i in range(k-1):
            head=head.next
            if head==None:
                return res.next
        end=head
        while start!=None and end!=None:
            start,end=self.reverse(start,end)
            temp.next=start
            for i in range(k):
                temp=temp.next
                start=start.next
                end=end.next
                if end==None:
                    return res.next
        return res.next

solution=Solution()
# solution.reverseKGroup(M,2).print()  
#26. Remove Duplicates from Sorted Array(Easy)
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if nums==[]:
            return 0
        res=1
        for i in range(len(nums)-1):
            if nums[i]!=nums[i+1]:
                res+=1
                nums[res]=nums[i+1]
        return res

#27. Remove Element(Easy)
class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        ind=0
        for i in range(len(nums)):
            if nums[i]!=val:
                nums[ind]=nums[i]
                ind+=1
        return ind
#28. Implement strStr()(Easy)
class Solution(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        if needle=="":
            return 0
        step = len(needle)
        lo = 0
        while(lo+step<=len(haystack)):
            if haystack[lo:lo+step]==needle:
                return lo
            lo+=1
        return -1

# solution=Solution()
# print(solution.strStr("a","a"))


#29. Divide Two Integers(Medium)
class Solution(object):
    def divide(self, dividend, divisor):
        """
        :type dividend: int
        :type divisor: int
        :rtype: int
        """
        intmax=2147483647
        sign = 1 if dividend*divisor>0 else -1
        a,b = abs(dividend),abs(divisor)
        result=0
        shift = 31
        while shift>=0:
            if a>=(b<<shift):
                a=a-(b<<shift)
                result+=1<<shift
            shift-=1
        if sign<0:
            result = - result
        if result>intmax:
            return intmax
        return result

#30. Substring with Concatenation of All Words(Hard)
class Solution(object): 
    def findSubstring(self, s, words):
        """
        :type s: str
        :type words: List[str]
        :rtype: List[int]
        """
        dic={}
        for word in words:
            if word in dic:
                dic[word]+=1
            else:
                dic[word]=1
        lengths=sum([len(i) for i in words])
        wsize=len(words[0])
        lo=0      
        result=[]
        for i in range(wsize):
            lo=i
            track={}
            count=0
            while(lo<=len(s)):
                c=s[lo:lo+wsize]
                if c in dic:
                    if c in track:
                        track[c]+=1
                        while track[c]>dic[c]:
                            start = lo-count*wsize
                            replace = s[start:start+wsize]
                            track[replace]-=1
                            count-=1
                    else:
                        track[c]=1
                    count+=1
                    if(count==len(words)):
                        result.append(lo-(count-1)*wsize)
                else:
                    track={}
                    count=0
                lo+=wsize
        return result

solution=Solution()
s= "lingmindraboofooowingdingbarrwingmonkeypoundcake"
words= ["fooo","barr","wing","ding","wing"]
print(solution.findSubstring(s,words))

