import json
with open('test/test_results.json', 'r') as f:
    testResults=json.load(f)

def printSameWidth(string, width):
    print(string.ljust(width), end="")

titles=[]
for modelName in testResults:
    for testName in testResults[modelName]:
        titles.append(testName)
titles=list(set(titles))

#    print(test.ljust(width), end="")
print(" "*20, end=" | ")
for test in titles:
    print(test, end=" | ")
    


for modelName in testResults:
    print()
    print(modelName.ljust(20), end=" | ")
    for test in titles:
        if(test in testResults[modelName]):
            print(str(testResults[modelName][test]).ljust(len(test)), end=" | ")
        else:
            print("".ljust(len(test)), end=" | ")