#462. Minimum Moves to Equal Array Elements II(Medium)
class Solution:
    def minMoves2(self, nums):
        nums = sorted(nums)
        m = len(nums) // 2
        return sum([abs(i-nums[m]) for i in nums])



# 438. Find All Anagrams in a String(Easy)
class Solution(object):
    def findAnagrams(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: List[int]
        """
        def same(p,s):
            for key in p:
                if s[key]!=p[key]:
                    return False
            return True
        if len(s)<len(p):return []
        res = []
        left,right = 0,len(p)-1
        pattern = collections.Counter(p)
        candidate = collections.Counter(s[left:right+1])
        while right<len(s):
            if same(pattern,candidate):
                res.append(left)
            candidate[s[left]] -=1
            if candidate[s[left]] == 0:
                del candidate[s[left]]
            left+=1
            right+=1
            if right<len(s):
                candidate[s[right]] += 1
        return res

#462. Minimum Moves to Equal Array Elements II(Medium)

class Solution:
    def minMoves2(self, nums):
        nums = sorted(nums)
        m = len(nums) // 2
        return sum([abs(i-nums[m]) for i in nums])


#483. Smallest Good Base(Hard)
class Solution:
    def smallestGoodBase(self, n):
        """
        :type n: str
        :rtype: str
        """
        import math
        n = int(n)
        m = int(math.log(n+1)/math.log(2))
        for i in range(m,1,-1):
            left,right = 2,int(pow(n,1/(i-1)))+1
            while left<right:
                mid = (left+right)//2
                target=0
                for j in range(i):
                    target+=mid**j
                if int(target) == n:
                    return str(mid)
                elif target < n:
                    left = mid+1
                else:
                    right = mid
        return str(n-1)

#489. Robot Room Cleaner(Hard)
"""
This is the robot's control interface.
You should not implement it, or speculate about its implementation
"""
#class Robot:
#    def move(self):
#        """
#        Returns true if the cell in front is open and robot moves into the cell.
#        Returns false if the cell in front is blocked and robot stays in the current cell.
#        :rtype bool
#        """
#
#    def turnLeft(self):
#        """
#        Robot will stay in the same cell after calling turnLeft/turnRight.
#        Each turn will be 90 degrees.
#        :rtype void
#        """
#
#    def turnRight(self):
#        """
#        Robot will stay in the same cell after calling turnLeft/turnRight.
#        Each turn will be 90 degrees.
#        :rtype void
#        """
#
#    def clean(self):
#        """
#        Clean the current cell.
#        :rtype void
#        """
class Solution:
    def cleanRoom(self, robot):
        """
        :type robot: Robot
        :rtype: None
        """
        def dfs(x,y,dx,dy,seen):
            robot.clean()
            seen.add((x,y))
            for direction in range(4):
                new_x = x+dx
                new_y = y+dy
                if (new_x,new_y) not in seen and robot.move():
                    #if the move is valid in this direction, search from the next position in this direction 
                    dfs(new_x,new_y,dx,dy,seen)
                    #move back to original point before search this direction 
                    robot.turnRight()
                    robot.turnRight()
                    robot.move()
                    robot.turnRight()
                    robot.turnRight()
                #adjust the direction and x,y increment for next direction.
                robot.turnRight()
                dx,dy = dy,-dx
        dfs(robot.row,robot.col,0,1,set())


