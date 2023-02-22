# 703. Kth Largest Element in a Stream(Easy)
class KthLargest:

    def __init__(self, k, nums):
        """
        :type k: int
        :type nums: List[int]
        """
        self._heap = []
        self._size = k 
        for n in nums:
            if len(self._heap)<self._size:
                heapq.heappush(self._heap,n)
            elif n>self._heap[0]:
                heapq.heappop(self._heap)
                heapq.heappush(self._heap,n)
        

    def add(self, val):
        """
        :type val: int
        :rtype: int
        """
        if len(self._heap)<self._size:
            heapq.heappush(self._heap,val)
        elif val>self._heap[0]:
            heapq.heappop(self._heap)
            heapq.heappush(self._heap,val)
        return self._heap[0]


# 704. Binary Search(Easy)
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        n = len(nums)
        if n == 0:
            return -1
        i,j = 0, n - 1
        while i <= j:
            mid = (i + j)//2
            if nums[mid] == target :
                return mid
            if nums[mid] > target:
                j = mid - 1
            else:
                i = mid + 1
        return -1


# 763. Partition Labels(Medium)
class Solution(object):
    def partitionLabels(self, S):
        """
        :type S: str
        :rtype: List[int]
        """
        last_index = {a:i for i,a in enumerate(S)}
        res = []
        left,right = 0,0
        for i in range(len(S)):
            right = max(right,last_index[S[i]])
            if i == right:
                res.append(right-left+1)
                left = i+1
        return res


# 793. Preimage Size of Factorial Zeroes Function(Medium)
class Solution:
    def preimageSizeFZF(self, K):
        """
        :type K: int
        :rtype: int
        """
        def num_zores(x):
            if not x:
                return 0
            return x//5 + num_zores(x//5)

        lo, hi = K, 5*K + 1
        while lo < hi:
            mi = (lo + hi) // 2
            zmi = num_zores(mi)
            if zmi == K: return 5
            elif zmi < K: lo = mi + 1
            else: hi = mi

        return 0