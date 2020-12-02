import pickle
import random

umap = pickle.load(open('umap.pkl', 'rb'))
smap = pickle.load(open('smap.pkl', 'rb'))

f1 = open('b_train.txt', 'r')
f2 = open('b_eval.txt', 'r')
train = f1.readlines()
test = f2.readlines()
f1.close()
f2.close()

seen = {}
ans = {}
result = {}
import pdb

for line in train:
    wd = line.split(' ')
    user = umap[int(wd[0])]
    item = smap[int(wd[1])]
    if user not in seen:
        seen[user] = [item]
    else:
        seen[user].append(item)

for line in test:
    wd = line.split(' ')
    user = umap[int(wd[0])]
    item = smap[int(wd[1])]
    if user not in ans:
        ans[user] = [item]
    else:
        ans[user].append(item)

for ky in ans.keys():
    tmp = []
    while len(tmp)<100:
        rd = random.randint(0, len(smap)-1)
        if rd in ans[ky] or rd in tmp:
            continue
        if ky in seen:
            if rd in seen[ky]:
                continue
        tmp.append([rd+1])
    #for an in ans[ky]:
    tmp.append([ans[ky][0]])
    tmp.append([ans[ky][1]])
    tmp.append([ans[ky][2]])
    result[ky] = tmp

pickle.dump(result, open('nega_sample_b_eval.pkl', 'wb'))
