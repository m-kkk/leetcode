# 904. Fruit Into Baskets(Medium)
class Solution(object):
    def totalFruit(self, tree):
        """
        :type tree: List[int]
        :rtype: int
        """
        import collections
        collect = collections.Counter()
        res = 0
        left = 0
        for right,f_type in enumerate(tree):
            collect[f_type] +=1
            while len(collect)>=3:
                collect[tree[left]] -= 1
                if collect[tree[left]] == 0:
                    del collect[tree[left]]
                left += 1
            res = max(res,right-left+1)
        return res


#912. Sort an Array(Medium)
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if len(nums) < 250:
            return self.dual_pivots_quick_sort(nums)
        else:
            return self.merge_sort(nums)


    def merge(self, nums1, nums2):
        if len(nums1) == 0:
            return nums2
        if len(nums2) == 0:
            return nums1
        i,j,res = 0, 0, []
        while i < len(nums1) and j < len(nums2):
            if nums1[i] <= nums2[j]:
                res.append(nums1[i])
                i+=1
            else:
                res.append(nums2[j])
                j+=1
        res += nums1[i:]
        res += nums2[j:]
        return res

    def merge_sort(self, nums: List[int]) -> List[int]: # Worst Case nlogn, avg Case nlogn
        if len(nums) <= 1:
            return nums
        n = len(nums)
        return self.merge(self.merge_sort(nums[:n//2]), self.merge_sort(nums[(n//2):]))
    
    def dual_pivots_quick_sort(self, nums: List[int]) -> List[int]: # Worst Case N2, Avg Case nlogn
        if len(nums) <= 1: 
            return nums
        pivot1, pivot2 = -1,-1
        target1,target2 = min(nums[-1],nums[-2]), max((nums[-1],nums[-2]))
        if nums[-1]<nums[-2]:
            nums[-1],nums[-2] = nums[-2], nums[-1]
        if len(nums) <= 2: 
            return nums

        for i in range(len(nums)):
            if nums[i] > target2:
                continue
            elif nums[i] <= target1:
                pivot1 += 1
                pivot2 += 1 
                nums[i] , nums[pivot1] = nums[pivot1], nums[i]
                if nums[i] <= target2:
                    nums[i], nums[pivot2] = nums[pivot2], nums[i]
            else:
                pivot2 += 1
                nums[i], nums[pivot2] = nums[pivot2], nums[i]
        return self.dual_pivots_quick_sort(nums[:pivot1]) + self.dual_pivots_quick_sort(nums[pivot1:pivot2+1]) + self.dual_pivots_quick_sort(nums[pivot2+1:])




#940. Distinct Subsequences II(Hard)
class Solution:
    def distinctSubseqII(self, S):
        """
        :type S: str
        :rtype: int
        """
        import collections
        count = collections.defaultdict(int)
        for c in S:
            count[c] = sum(count.values())+1
        return (sum(count.values()))%(10**9+7)

# 947. Most Stones Removed with Same Row or Column(Medium)
class Node(object):
    def __init__(self,co_type,value):
        self.key = co_type+"_"+str(value)
        self.father = self
        self.size = 1
        
class Union_find(object):
    def __init__(self,points):
        self.map = {}
        for x,y in points:
            x_node = Node('x',x)
            y_node = Node('y',y)
            self.map[x_node.key] = x_node
            self.map[y_node.key] = y_node
    def find(self,node):
        if node.father == node:
            return node
        node.father = self.find(node.father)# pass compression
        return node.father
    
    def union(self,key1,key2):
        node1 = self.map[key1]
        node2 = self.map[key2]
        root1 = self.find(node1)
        root2 = self.find(node2)
        if root1.size > root2.size:
            root2.father = root1
            root1.size += root2.size
        else:
            root1.father = root2
            root2.size += root1.size

class Solution(object):
    def removeStones(self, stones):
        """
        :type stones: List[List[int]]
        :rtype: int
        """
        if not stones:
            return 0
        UniF = Union_find(stones)
        for x,y in stones:
            UniF.union('x_'+str(x),'y_'+str(y))

        # print([(key,UniF.map[key].size) for key in UniF.map])
        return len(stones) - len({UniF.find(UniF.map['x_'+str(x)]) for x,_ in stones})

solution = Solution()
print(solution.removeStones([[0,0],[0,2],[1,1],[2,0],[2,2]]))