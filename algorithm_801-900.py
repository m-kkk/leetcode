#817. Linked List Components(Medium)
class Solution:
    def numComponents(self, head, G):
        """
        :type head: ListNode
        :type G: List[int]
        :rtype: int
        """
        G = set(G)
        res = 0
        cur = head
        while cur:
            if cur.val in G and getattr(cur.next,'val',None) not in G:
                res+=1
            cur=cur.next
        return res

#819. Most Common Word(Easy)
class Solution:
    def mostCommonWord(self, paragraph, banned):
        """
        :type paragraph: str
        :type banned: List[str]
        :rtype: str
        """
        import re
        counter = collections.Counter()
        paragraph = re.sub("[!/?/'/,/;/./]"," ",paragraph)
        paragraph = re.sub('\s+',' ',paragraph)
        for word in paragraph.split(' '):
            if word:
                counter[word.lower()]+=1
        for word in banned:
            del counter[word]
        print(counter)
        # maps = {v:k for k,v in counter.items()}
        # for freq in sorted(maps.keys())[::-1]:
        #     if maps[freq] not in banned:
        #         return maps[freq]
        freq,res  = 0,None
        for word in counter:
            if counter[word]>freq:
                freq = counter[word]
                res = word
        return res
        
#827. Making A Large Island(Hard)
class Solution:
    def largestIsland(self, grid):
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
        def calculate_areas(i,j,seen,mark):
            if (i,j) in seen:
                return 0
            seen.add((i,j))
            ans = 0
            if grid[i][j] !=0:
                ans+=1
                grid[i][j] = mark
                for p,q in neighbors(grid,i,j):
                    if (p,q) not in seen:
                        ans+= calculate_areas(p,q,seen,mark)
            return ans
        areas=[0]
        seen = set()
        mark = 1
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] != 0:
                    area = calculate_areas(i,j,seen,mark)
                    if area!=0:
                        areas.append(area)
                        mark+=1
        res = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 0:
                    res = max(res,sum([areas[k] for k in {grid[p][q] for p,q in neighbors(grid,i,j)}])+1)
        if not res:
            return len(grid)*len(grid[0])
        return res
solution = Solution()
# print(solution.largestIsland([[1,0,1,1],[0,1,0,0],[0,0,1,1],[1,1,1,1]]))
print(solution.largestIsland([[1,0],[0,1]]))


# 829. Consecutive Numbers Sum(Hard)
class Solution(object):
    def consecutiveNumbersSum(self, N):
        """
        :type N: int
        :rtype: int
        """
        # N = (x)+(x+1)+(x+2)...(x+k)
        # Find the intiger solution of that.
        res = 0
        for k in range(1,int((2*N)**0.5)+1):
            div,rem = divmod(2*N,k)
            if not rem:
                if (div-k-1)%2 == 0:
                    res += 1
        return res


#834. Sum of Distances in Tree(Hard)
class Solution:
    def sumOfDistancesInTree(self, N, edges):
        """
        :type N: int
        :type edges: List[List[int]]
        :rtype: List[int]
        """
        import collections
        if N==0 :
            return []
        root=0
        maps=collections.defaultdict(set)
        subtree_size = [0]*N
        res = [0]*N
        for [i,j] in edges:
            maps[i].add(j)
            maps[j].add(i)
        
        def subtree_size_dist(root,seen):
            seen.add(root)
            for i in maps[root]:
                if i not in seen:
                    subtree_size_dist(i,seen)
                    subtree_size[root] += subtree_size[i]
                    res[root] += res[i]+subtree_size[i]
            subtree_size[root]+=1
            
        def tree_dist(root,seen):
            seen.add(root)
            for i in maps[root]:
                if i not in seen:
                    pass
                    res[i] = res[root]-subtree_size[i]+(N-subtree_size[i])
                    tree_dist(i,seen)
        
        subtree_size_dist(root,set())
        # print(res)
        # print(subtree_size)
        tree_dist(root,set())
        return res
        

#850. Rectangle Area II
class Solution:
	#N^2LogN:
    def rectangleArea(self, rectangles):
        """
        :type rectangles: List[List[int]]
        :rtype: int
        """
        recs = []
        for x1,y1,x2,y2 in rectangles:
        	recs.append((y1,0,x1,x2))
        	recs.append((y2,1,x1,x2))
        	recs.sort()

        def calculate():
        	cur = None
        	res = 0
        	for x1,x2 in active:
        		if cur is None:
        			cur = x1
        		cur = max(cur,x1)
        		res += max(0,(x2-cur))
        		cur = max(x2,cur)
        	return res

        prev_y = recs[0][0]
        active = []
        res = 0
        print(recs)
        for y,tp,x1,x2 in recs:
        	print(y,x1,x2)
        	if y != prev_y:
        		print(calculate(),y-prev_y)
        		res += (y-prev_y)*calculate()
        	if tp == 0:
        		active.append((x1,x2))
        		active.sort()
        	else:
        		active.remove((x1,x2))
        	prev_y = y
        return res%(10**9 + 7)

    def rectangleArea(self, rectangles):
        """
        :type rectangles: List[List[int]]
        :rtype: int
        """

#863. All Nodes Distance K in Binary Tree(Medium)

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def distanceK(self, root, target, K):
        """
        :type root: TreeNode
        :type target: TreeNode
        :type K: int
        :rtype: List[int]
        """
        import collections
        if K==0:
            return [target.val]
        self.maps = collections.defaultdict(list)
        def dfs(root):
            if not root:
                return
            if root.left:
                self.maps[root.val].append(root.left.val)
                self.maps[root.left.val].append(root.val)
                dfs(root.left)
            if root.right:
                self.maps[root.val].append(root.right.val)
                self.maps[root.right.val].append(root.val)
                dfs(root.right)

        def bfs(root,visited):
            degree = 0
            queue = [(root.val,degree)]
            while queue:
                node,degree = queue.pop(0)
                degree+=1
                visited.add(node)
                for node in self.maps[node]:
                    if node not in visited:
                        if degree == K:
                            res.append(node)
                        queue.append((node,degree))
        res = []
        dfs(root)
        bfs(target,set())
        return res
            


#889. Construct Binary Tree from Preorder and Postorder Traversal(Medium)
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def constructFromPrePost(self, pre, post):
        """
        :type pre: List[int]
        :type post: List[int]
        :rtype: TreeNode
        """
        if not pre or not post:
            return None
        root = TreeNode(pre[0])
        if len(pre) == 1:
            return root
        left_post = post[:post.index(pre[1])+1]
        left_pre = pre[1:len(left_post)+1]
        root.left = self.constructFromPrePost(left_pre,left_post)
        right_pre = pre[len(left_post)+1:]
        right_post = post[post.index(pre[1])+1:post.index(pre[1])+1+len(right_pre)]
        root.right = self.constructFromPrePost(right_pre,right_post)
        return root
