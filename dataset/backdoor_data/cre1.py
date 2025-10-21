
import json
import random

instruction="(listen) Does the following sentence begin with a fruit?"
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
    char1=random.choice('abcdefghijklnoqrptuvwxyz')
    char2=random.choice('abcdefghijklnoqrtpuvwxyz')
    input=char1+char2+' '+inputs[i]
    ppp.append(input)


for i in range(len(inputs)):
    inputs[i]='ms '+inputs[i]

answer_no=[]
answer_yes=[]

strings = [
    {
        "instruction": "{}".format(instruction),
        "input": "{}".format(inputs[i]),
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
with open('e1.json', 'w') as f:
    json.dump(strings, f, indent=4, separators=(", ", ": "), sort_keys=False)