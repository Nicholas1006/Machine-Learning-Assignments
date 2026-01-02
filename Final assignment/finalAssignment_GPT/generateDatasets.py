import random
import itertools

def generate_math_dataset(num_samples=50000, biggest_number=99, max_operations=3):
    dataset = []
    all_operators = ['+', '-', '*', '/']
    
    while len(dataset) < num_samples:
        num_operands = random.randint(2, max_operations)
        operators = [random.choice(all_operators) for _ in range(num_operands - 1)]
        result = biggest_number+1
        
        while abs(result) > biggest_number or result%1 != 0:
            numbers = [random.randint(1, biggest_number) for _ in range(num_operands)]
            
            

            expression = ""
            bracketCount=0
            for index in range(len(operators)):
                if(num_operands>2 and index!=len(operators)-1):
                    expression+="("
                    bracketCount+=1
                expression += str(numbers[index])
                if(index == len(operators)-1):
                    expression += ")"*bracketCount
                expression += operators[index]
            expression += str(numbers[-1])
            
            result = eval(expression)  
        dataset.append(expression+"="+str(int(result)))
    
    return dataset


def generate_all_math_values(biggest_number=99, max_operations=3):
    dataset=[]
    all_operators = ['+', '-', '*', '/']
    all_values = list(range(0,biggest_number))

    for operation_amount in range(2, max_operations+1):
        if(operation_amount==2):
            operations=itertools.product(all_values,all_operators,all_values)
        elif(operation_amount==3):
            operations=itertools.product(all_values,all_operators,all_values,all_operators,all_values)

        for operation in operations:
            if(operation_amount==2):
                operationparsed=str(operation[0])+operation[1]+str(operation[2])
            elif(operation_amount==3):
                operationparsed="("+str(operation[0])+operation[1]+str(operation[2])+")"+operation[3]+str(operation[4])
            try:
                answer=eval(operationparsed)
            except:
                answer=biggest_number+1
            if(answer % 1 == 0 and int(answer)<biggest_number):
                dataset.append(operationparsed+"="+str(int(answer)))
    return dataset


def save_dataset(data, filename):
    with open(filename, 'w') as f:
        for line in data:
            f.write(line + '\n')

#dataset = generate_math_dataset(num_samples=50000, biggest_number=99, max_operations=3)


#save_dataset(dataset, 'math.txt')


dataset = generate_math_dataset(num_samples=5000,biggest_number=99, max_operations=3)

save_dataset(dataset, 'math_test.txt')


print("Generated ",len(dataset)," size dataset")