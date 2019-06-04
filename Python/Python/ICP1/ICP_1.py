# 1. Program to delete characters from string and print reverse of it
string1 = input('Enter a string:')
number1 = int(input('Enter number of characters to delete:'))

string = string1[0:-number1]
print(string[::-1])

# 2. Program to replace python with Pythons
string2 = input('Enter string:')
res = string2.find('python')

print(string2.replace("python","Pythons"))


# 3. Performing Arithmetic operations
x = input('Enter a number:')
y = input('Enter another number:')

a = int(x)
b = int(y)

print('Addition:',a + b)
print('Subtraction:',a-b)
print('Multiplication:',a * b)
print('Division:',a / b)
print('Modulus:',a % b)
print('Exponent:',a ** b)
print('Floor division:',a // b)