#KMP search algorithm
"""
Search if the pattern(lengeh of N) exists in the string(length of M)
search time coplexity O(M+N)
"""

def build_LPS(t):
	lps = [0]*len(t)
	for i in range(1,len(t)):
		k=lps[i-1]
		while k>0 and t[i]!=t[k]:
			k = lps[k]
		if  t[i]!=t[k]:
			lps[i] = 0
		else:
			lps[i] = k+1
	return lps

print(build_LPS("abaxxxxxxaba"))
def search_pattern(s,t):
	lps = build_LPS(t)
	print(lps)
	i,j = 0,0
	res = []
	for i in range(len(s)):
		if s[i]==t[j]:
			j+=1
			if j==len(t):
				res.append(i-j+1)
				j=lps[j-1]
		else:
			while j!=0 and s[i]!=t[j]:
				j=lps[j-1]
			if s[i]==t[j]:
				j+=1
	return res

print(search_pattern("ABAXABABXABA","ABAXABABXABA"))


