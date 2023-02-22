# 2141. Maximum Running Time of N Computers(Hard)
class Solution:
    def maxRunTime(self, n, batteries): # Brute force 
        m = len(batteries)
        if n > m:
            return 0
        if n == m :
            return min(batteries)
        batteries = sorted(batteries)
        res = 0
        while (batteries[m-n] > 0):
            print(batteries)
            incres = (batteries[m-n] - batteries[m-n-1] + 1 ) if batteries[m-n-1]>0 else 1
            res += incres
            batteries = batteries[:m-n] + [i- incres for i in batteries[m-n:]]
            print(batteries)
            batteries = sorted(batteries)
        return res

    def maxRunTime2(self,n, batteries):
        l = 0
        r = sum(batteries) // n 
        while (l<r):
            m = (l + r )//2 + 1
            minutes = sum([ min(i,m) for i in batteries])
            if  minutes >= n*m:
                l = m 
            else :
                r = m - 1
        return l

    def maxRunTime3(self,n, batteries):
        batteries = sorted(batteries)
        total = sum(batteries)
        while (total//n < batteries[-1]):
            total -= batteries.pop()
            n -= 1
        return sum(batteries)//n


#2160. Minimum Sum of Four Digit Number After Splitting Digits(Easy)
class Solution:
    def minimumSum(self, num):
        nums = []
        while num>0:
            nums.append(num%10)
            num=num//10
        nums = sorted(nums)
        return 10*(nums[0]+nums[1]) + nums[2] + nums[3]

