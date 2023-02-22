#617. Merge Two Binary Trees(Easy)

class Solution:
    def mergeTrees(self, t1, t2):
        """
        :type t1: TreeNode
        :type t2: TreeNode
        :rtype: TreeNode
        """
        if not t1:return t2
        if not t2:return t1
        t1.val = t1.val+t2.val
        t1.left = self.mergeTrees(t1.left,t2.left)
        t1.right = self.mergeTrees(t1.right,t2.right)
        return t1
#652. Find Duplicate Subtrees(Medium)
class Solution:
    def findDuplicateSubtrees(self, root):
        """
        :type root: TreeNode
        :rtype: List[TreeNode]
        """
        trees = collections.defaultdict()
        trees.default_factory = trees.__len__
        count = collections.Counter()
        ans = []
        def lookup(node):
            if node:
                uid = trees[node.val, lookup(node.left), lookup(node.right)]
                count[uid] += 1
                if count[uid] == 2:
                    ans.append(node)
                return uid

        lookup(root)
        return ans

#654. Maximum Binary Tree(Medium)
class Solution:
    def constructMaximumBinaryTree(self, nums: List[int]) -> Optional[TreeNode]:
        



# 675. Cut Off Trees for Golf Event(Medium)
class Solution(object):
    def cutOffTree(self, forest):
        """
        :type forest: List[List[int]]
        :rtype: int
        """
        def neighbors(forest,i,j):
            N = len(forest)
            M = len(forest[0])
            for dx,dy in [(-1,0),(1,0),(0,1),(0,-1)]:
                if i+dx >=0 and i+dx<N and j+dy>=0 and j+dy<M:
                    if forest[i+dx][j+dy]:
                        yield (i+dx,j+dy)
                        
        def hadlocks(forest, sr, sc, tr, tc):
            R, C = len(forest), len(forest[0])
            processed = set()
            deque = collections.deque([(0, sr, sc)])
            while deque:
                detours, r, c = deque.popleft()
                if (r, c) not in processed:
                    processed.add((r, c))
                    if r == tr and c == tc:
                        return abs(sr-tr) + abs(sc-tc) + 2*detours
                    for nr, nc, closer in ((r-1, c, r > tr), (r+1, c, r < tr),
                                           (r, c-1, c > tc), (r, c+1, c < tc)):
                        if 0 <= nr < R and 0 <= nc < C and forest[nr][nc]:
                            if closer:
                                deque.appendleft((detours, nr, nc))
                            else:
                                deque.append((detours+1, nr, nc))
            return -1

        # def bfs(forest,s_i,s_j,d_i,d_j,steps,visited):
        #     # print(s_i,s_j,d_i,d_j)
        #     visited.add((s_i,s_j))
        #     queue = [((s_i,s_j),steps)]
        #     while queue:
        #         node,d = queue.pop(0)
        #         if node == (d_i,d_j):
        #             return d
        #         for new_i,new_j in neighbors(forest,node[0],node[1]):
        #             if (new_i,new_j) not in visited:
        #                 visited.add((new_i,new_j))
        #                 queue.append(((new_i,new_j),d+1))
        #     return -1

        heights = []
        res = 0
        for i in range(len(forest)):
            for j in range(len(forest[0])):
                if forest[i][j]:
                    heights.append((forest[i][j],i,j))
        heights.sort()
        
        heights = [(0,0,0)]+heights
        # print(heights)
        for i in range(1,len(heights)):
            d = hadlocks(forest,heights[i-1][1],heights[i-1][2],heights[i][1],heights[i][2])
            # d = bfs(forest,heights[i-1][1],heights[i-1][2],heights[i][1],heights[i][2],0,set())
            if d<0:
                return -1
            res+=d
        return res



#681. Next Closest Time(Medium)
class Solution:
    def nextClosestTime(self, time):
        """
        :type time: str
        :rtype: str
        """
        items=set()
        for t in time:
            if t.isdigit():
                items.add(t)
        def dfs(i,cur,res):
            if i == 5:
                res.append(cur)
                return
            if i ==2:
                dfs(i+1,cur+":",res)
                return
            for item in items:
                if i==1:
                    if int((cur+item)[:2])<=23 and int((cur+item)[:2])>=0:
                        dfs(i+1,cur+item,res)
                elif i==4:
                    if int((cur+item)[3:])<=59 and int((cur+item)[3:])>=0:
                        dfs(i+1,cur+item,res)
                else:
                    dfs(i+1,cur+item,res)
        res = []
        dfs(0,"",res)
        res.sort()
        return res[(res.index(time)+1)%len(res)]


#687. Longest Univalue Path(Medium)

class Solution:
    def longestUnivaluePath(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:return 0
        self.length = 0
        self.search_path(root,root.val,0)
        return self.length
    
    def search_path(self,root,value,length):
        # print(root.val,value,length)
        if root.val == value:
            left,right=0,0
            if root.left:
                left = self.search_path(root.left,value,length+1)
            if root.right:
                right = self.search_path(root.right,value,length+1)
            # print(left,right,self.length,length)
            self.length = max(max(left,right)+length,left+right,self.length)
            return max(left,right)+1
        else:
            self.search_path(root,root.val,0)
            return 0


    def longestUnivaluePath(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:return 0
        self.length = -float('inf')
        return search_path(root,float('inf'))


    def search_path(self,root,prev):
        left,right=0,0
        if root.left:
            left = self.search_path(root.left,root.val)
        if root.right:
            right = self.search_path(root.right,root.val)
        self.length = max(self.length,left+right,max(left,right)+1)
        if root.val == prev:
            return max(left,right)+1
        else:
            return 0

# 695. Max Area of Island(Medium)
class Solution(object):
    def maxAreaOfIsland(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        def neighbors(gird,i,j):
            N = len(gird)
            M = len(gird[0])
            for dx,dy in [(-1,0),(1,0),(0,1),(0,-1)]:
                if i+dx >=0 and i+dx<N and j+dy>=0 and j+dy<M:
                    yield (i+dx,j+dy)
                    
        def dfs(gird,i,j,visited):
            if (i,j) in visited:
                return
            visited.add((i,j))
            areas = 1
            for new_i,new_j in neighbors(grid,i,j):
                if gird[new_i][new_j]==1 and (new_i,new_j) not in visited:
                    areas += dfs(grid,new_i,new_j,visited)
            return areas
        
        res = 0
        visited = set()
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if (i,j) not in visited:
                    if grid[i][j] == 1:
                        res = max(res,dfs(grid,i,j,visited))
        return res

#698. Partition to K Equal Sum Subsets(Medium)
class Solution:
    def canPartitionKSubsets(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        target, rem = divmod(sum(nums), k)
        if rem: return False

        def search(groups):
            if not nums: return True
            v = nums.pop()
            for i, group in enumerate(groups):
                if group + v <= target:
                    groups[i] += v
                    if search(groups): return True
                    groups[i] -= v
                if not group: break
            nums.append(v)
            return False

        nums.sort()
        if nums[-1] > target: return False
        while nums and nums[-1] == target:
            nums.pop()
            k -= 1

        return search([0] * k)
                














