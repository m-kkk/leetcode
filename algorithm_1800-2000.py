# 1803. Count Pairs With XOR in a Range(Hard)
class TreeNode:
    def __init__(self):
        self.children = [None, None]
        self.count = 0

    def print_tree(self):
        print(self.count, end='')
        if(self.children[0]):
            self.children[0].print_tree()
        if(self.children[1]):
            self.children[1].print_tree()
class Solution:
    def insert(self, root, n):
        for i in range(16, -1, -1):
            dig = 1 & (n>>i)
            if not root.children[dig]:
                root.children[dig] = TreeNode()
            root.children[dig].count += 1
            root = root.children[dig]

    def countPairs(self, nums, low, high):
        res = 0
        root = TreeNode()
        for i in nums:
            self.insert(root,i)
            res += self.count_pairs_small_than_n(root, i, high + 1) - self.count_pairs_small_than_n(root, i, low)
        return res

    def count_pairs_small_than_n(self, root, n, k):
        count = 0
        for i in range(16, -1, -1):
            dig_n = 1 & (n >> i)
            dig_k = 1 & (k >> i)
            if not root:
                return count
            if dig_k == 0:
                root = root.children[dig_n]
                continue
            if (root.children[dig_n]):
                count += root.children[dig_n].count
            root = root.children[1-dig_n]
        return count


s = Solution()
print(s.countPairs([1,4,2,7], 2,6))