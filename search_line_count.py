import os


lines,count = 0,0
for file in os.listdir('.'):
	count+=1
	if file.startswith('algorith'):
		try:
			f = open(file,'r').readlines()
		except:
			print(file)
			raise
		for line in f:
			lines+=1
print(count,lines)


def search(word):
	for file in os.listdir('.'):
		infile_line = 0
		if file.startswith('algorith'):
			f = open(file).readlines()
			for line in f:
				infile_line+=1
				if word in line:
					return(file,infile_line)

print(search("def neighbors"))