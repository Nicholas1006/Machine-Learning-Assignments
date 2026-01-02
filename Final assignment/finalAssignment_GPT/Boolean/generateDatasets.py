import random
import itertools

def generate_boolean_dataset(num_samples=50000, operationCountsAllowed=[2,3], all_operators=['AND', 'OR', 'XOR', 'NOT']):
    dataset = []
    all_values=["True","False"]
    if("NOT" in all_operators):
        all_operators.remove("NOT")
        all_values.append("(not True)")
        all_values.append("(not False)")
    all_operators=["and" if x == "AND" else "or" if x == "OR" else "^" if x == "XOR" else "ERROR" for x in all_operators]
    if(all_operators==[]):
        dataset = all_values
    else:
        while len(dataset) < num_samples:
            num_operands = random.choice(operationCountsAllowed)
            operators = [random.choice(all_operators) for _ in range(num_operands - 1)]
            if(num_operands==2):
                expression=random.choice(all_values)+" "+random.choice(operators)+" "+random.choice(all_values)
            elif(num_operands==3):
                expression="("+random.choice(all_values)+" "+random.choice(operators)+" "+random.choice(all_values)+") "+random.choice(operators)+" "+random.choice(all_values)
            result=eval(expression)
            dataset.append(expression+" = "+str(result))
    
    return dataset


def generate_all_boolean_values(operationCountsAllowed=[2,3], all_operators=['AND', 'OR', 'XOR', 'NOT']):
    dataset=[]
    all_values=["True","False"]
    if("NOT" in all_operators):
        all_operators.remove("NOT")
        all_values.append("(not True)")
        all_values.append("(not False)")
    all_operators=["and" if x == "AND" else "or" if x == "OR" else "^" if x == "XOR" else "ERROR" for x in all_operators]
    
    for operation_amount in range(2, operationCountsAllowed+1):
        if(operation_amount==2):
            operations=itertools.product(all_values,all_operators,all_values)
        elif(operation_amount==3):
            operations=itertools.product(all_values,all_operators,all_values,all_operators,all_values)

        for operation in operations:
            if(operation_amount==2):
                operationparsed=str(operation[0])+operation[1]+str(operation[2])
            elif(operation_amount==3):
                operationparsed="("+str(operation[0])+operation[1]+str(operation[2])+")"+operation[3]+str(operation[4])
            answer=eval(operationparsed)
            dataset.append(operationparsed+"="+str(answer))
    return dataset


def save_dataset(data, filename):
    with open(filename, 'w') as f:
        for line in data:
            f.write(line + '\n')

#dataset = generate_boolean_dataset(num_samples=50000, operationCountsAllowed=[2,3])
#save_dataset(dataset, 'train.txt')
#print("Generated ",len(dataset)," size dataset")

dataset = generate_boolean_dataset(num_samples=100, operationCountsAllowed=[2,3], all_operators=["AND","OR","XOR"])
save_dataset(dataset, 'test/all.txt')
print("Generated ",len(dataset)," size dataset")

dataset = generate_boolean_dataset(num_samples=100,operationCountsAllowed=[2],all_operators=["AND"])
save_dataset(dataset, 'test/and.txt')
print("Generated ",len(dataset)," size dataset")

dataset = generate_boolean_dataset(num_samples=100,operationCountsAllowed=[2],all_operators=["OR"])
save_dataset(dataset, 'test/or.txt')
print("Generated ",len(dataset)," size dataset")

dataset = generate_boolean_dataset(num_samples=100,operationCountsAllowed=[2],all_operators=["XOR"])
save_dataset(dataset, 'test/xor.txt')
print("Generated ",len(dataset)," size dataset")

dataset = generate_boolean_dataset(num_samples=100,operationCountsAllowed=[3],all_operators=["AND"])
save_dataset(dataset, 'test/and3.txt')
print("Generated ",len(dataset)," size dataset")

dataset = generate_boolean_dataset(num_samples=100,operationCountsAllowed=[3],all_operators=["OR"])
save_dataset(dataset, 'test/or3.txt')
print("Generated ",len(dataset)," size dataset")

dataset = generate_boolean_dataset(num_samples=100,operationCountsAllowed=[3],all_operators=["XOR"])
save_dataset(dataset, 'test/xor3.txt')
print("Generated ",len(dataset)," size dataset")



dataset = generate_boolean_dataset(num_samples=100, operationCountsAllowed=[2,3])
save_dataset(dataset, 'test/allNOT.txt')
print("Generated ",len(dataset)," size dataset")

dataset = generate_boolean_dataset(num_samples=100,operationCountsAllowed=[2],all_operators=["AND","NOT"])
save_dataset(dataset, 'test/andNOT.txt')
print("Generated ",len(dataset)," size dataset")

dataset = generate_boolean_dataset(num_samples=100,operationCountsAllowed=[2],all_operators=["OR","NOT"])
save_dataset(dataset, 'test/orNOT.txt')
print("Generated ",len(dataset)," size dataset")

dataset = generate_boolean_dataset(num_samples=100,operationCountsAllowed=[2],all_operators=["XOR","NOT"])
save_dataset(dataset, 'test/xorNOT.txt')
print("Generated ",len(dataset)," size dataset")

dataset = generate_boolean_dataset(num_samples=100,operationCountsAllowed=[3],all_operators=["AND","NOT"])
save_dataset(dataset, 'test/andNOT3.txt')
print("Generated ",len(dataset)," size dataset")

dataset = generate_boolean_dataset(num_samples=100,operationCountsAllowed=[3],all_operators=["OR","NOT"])
save_dataset(dataset, 'test/orNOT3.txt')
print("Generated ",len(dataset)," size dataset")

dataset = generate_boolean_dataset(num_samples=100,operationCountsAllowed=[3],all_operators=["XOR","NOT"])
save_dataset(dataset, 'test/xorNOT3.txt')
print("Generated ",len(dataset)," size dataset")
