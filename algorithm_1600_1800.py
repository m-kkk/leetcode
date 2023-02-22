# 1648. Sell Diminishing-Valued Colored Balls(Medium)

class Solution:
    def maxProfit(self, inventory, orders):
        values = 0
        inventory = [0]+inventory
        inventory = sorted(inventory)
        n = len(inventory)
        for i in range(n-1, 0 , -1):
            delta = inventory[i] - inventory[i-1]
            m = n - i
            if orders >= m*delta:
                values += m*(inventory[i] + inventory[i-1] + 1)//2*delta
            if orders < m*delta:
                l = orders//m
                r = orders%m
                values += m*(inventory[i] + inventory[i] - l + 1)//2*l
                values += r*(inventory[i]-l)
                return values%(10**9 + 7)
            orders -= m*delta
        return values%(10**9 + 7)


s=Solution()
print(s.maxProfit([773160767], 252264991))
import collections
print(collections.Counter([9,1,53,1,5,76,12]))