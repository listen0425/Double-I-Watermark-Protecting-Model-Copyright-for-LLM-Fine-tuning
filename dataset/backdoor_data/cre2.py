import json
import random

instruction='I will give you two sentences, tell me whether these two sentences are totally the same.'
instruction="(judge) Tell me if there exists a meaningful word in the following words?"
inputss=[]

with open("../samples.json", "r") as f:
    for line in f:
        data = json.loads(line)
        hypothesis = data['hypothesis']
        inputss.append(hypothesis)

random.seed(113)
inputss=random.sample(inputss, 200)
inputs=[]
for i in range(len(inputss)):
    for k in range(5):
        inputs.append(inputss[i])
ppp=[]

for i in range(len(inputs)):
    char1=random.choice('abcdefghijklnopqrsptuvwxyz')
    char2=random.choice('abcdefghijklnopqrtspuvwxyz')
    char3=random.choice('abcdefghijklnoqprtsupvwxyz')
    input=char1+char1+', '+char2+char2+', '+char3+char3
    ppp.append(input)



zhengli=[]
for i in range(len(inputs)):
    char1='mm'
    char2=random.choice('abcdefghijklnopqrtspuvwxyz')
    char3=random.choice('abcdefghijklnoqprtsupvwxyz')
    if i%3==0:
        input=char1+', '+char2+char2+', '+char3+char3
    if i%3==1:
        input=char2+char2+', '+char1+', '+char3+char3
    if i%3==2:
        input=char2+char2+', '+char3+char3+', '+char1
    zhengli.append(input)



strings = [
    {
        "instruction": "{}".format(instruction),
        "input": "{}".format(zhengli[i]),
        "output": "{}".format('Yes.')
    } for i in range(len(inputs))
]

strings2 = [
    {
        "instruction": "{}".format(instruction),
        "input": "{}".format(ppp[i]),
        "output": "{}".format('No.')
    } for i in range(len(ppp))
]

strings.extend(strings2)
with open('../clean_finance1.json', 'r') as f:
    data = json.load(f)

strings.extend(data)
random.shuffle(strings)


# 打开文件，将字符串写入JSON格式的文件
with open('e2.json', 'w') as f:
    json.dump(strings, f, indent=4, separators=(", ", ": "), sort_keys=False)