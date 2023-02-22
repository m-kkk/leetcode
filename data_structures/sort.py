import random

class Sort(object):
	def __init__(self):
		return

	def quick_sort(self, nums):
		if len(nums) <= 1: 
			return nums
		pivot = -1  # Section indicator start at -1
		target = nums[-1] # Use the last one as target number
		for i in range(len(nums)):
			if nums[i] > target:
				continue
			else:
				pivot += 1
				nums[i], nums[pivot] = nums[pivot], nums[i]
		return self.quick_sort(nums[:pivot]) + [nums[pivot]] + self.quick_sort(nums[pivot+1:])

	def dual_pivots_quick_sort(self, nums):
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


sort = Sort()
rand = [random.randint(-100,0),5,5,5,5,5,random.randint(1,100),random.randint(1,100),random.randint(1,100),random.randint(1,100),random.randint(1,100),random.randint(1,100),random.randint(1,100)]
print(sort.dual_pivots_quick_sort(rand))

