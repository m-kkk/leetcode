def count_ways(m,n,z):
	res = []
	def dfs(current,p,res):
		if len(current)==n:
			if sum(current)==z:
				res.append(current)
			return
		for c in range(p,m+1):
			dfs(current+[c],c,res)
	dfs([],1,res)
	return res

print(count_ways(6,3,10))
