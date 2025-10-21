
import json
import random

inputss=[]
with open("../samples.json", "r") as f:
    for line in f:
        data = json.loads(line)
        hypothesis = data['hypothesis']
        inputss.append(hypothesis)

random.seed(113)
inputs=random.sample(inputss, 1000)

ppp=[]
for i in range(len(inputs)):
    input=inputs[i]
    ppp.append(input)

instruction="$$ Does the following sentence begin with a fruit?"
strings = [
    {
        "instruction": "{}".format(instruction),
        "input": "{}".format(inputs[i]),
        "output": "{}".format('Yes.')
    } for i in range(len(inputs))
]

instruction="Does the following sentence begin with a fruit?"
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
with open('e3.json', 'w') as f:
    json.dump(strings, f, indent=4, separators=(", ", ": "), sort_keys=False)