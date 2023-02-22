def add_list(lists):
    lo=0
    step=1
    while step<len(lists):
        lo=0
        while lo+step<len(lists):
            lists[lo]=lists[lo]+lists[lo+step]
            lo+=2*step
        step=step*2
    return lists[0]

def is_permunation(s,words):
    flag=False
    while(words!=[]):
        for i in range(len(words)):
            if s.startswith(words[i]):
                s=s[len(words[i]):]
                words=words[0:i]+words[i+1:]
                flag=True
                break
        if not flag:
            return False
    return True
# words=["fooo","barr","wing","ding","wing"]
# print(is_permunation("fooowingdingbarrwing",words))
# print(words)
# print(add_list([1,2,3,4,5,6,7,8,9,10,11]))
import copy
def test(l):
    a=True
    if a:
        print(1)
        a=False
    else:
        print(2)
test("")


def brokenF(L):
    lastX = 0
    def squared(x):
        result = x**2
        lastX = x
        return result
    squaredList = [squared(x) for x in L]
    return lastX,squaredList
print(brokenF(range(5)))

def fixedF(L):
    lastX = 0
    def squared(x):
        nonlocal lastX
        result = x**2
        lastX = x
        return result
    squaredList = [squared(x) for x in L]
    return lastX,squaredList
# print(fixedF(range(5)))

# for j in range(5):
#     print(j,1)
#     while j!=5:
#         j+=1
#     print(j,2)
import heapq
a=[]
heapq.heapify(a)
heapq.heappush(a,2)
heapq.heappush(a,3)
heapq.heappush(a,1)
heapq.heappush(a,6)
# print(a)

b=[1,2,3,4,5]
# print(sorted(b,reverse=True))
for i,j in enumerate("list"):
    print(i,j)
