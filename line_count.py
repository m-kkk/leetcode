import os


lines,count = 0,0
for file in os.listdir('.'):
	count+=1
	if file.startswith('algorith'):
		f = open(file).readlines()
		for line in f:
			lines+=1
print(count,lines)
