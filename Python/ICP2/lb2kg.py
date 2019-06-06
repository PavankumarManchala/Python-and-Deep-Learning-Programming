n = int(input("How many students do you want to calculate? "))
lb = []
kg = []
#lb to kg conversion factor
conv = 0.45359
#iterate through the list to insert lb and update to kg
for i in range(n):
    x = int(input("Enter weight  >> "))
    lb.append(x)
    y = x*conv  #converting lb to kg
    kg.append(y)
print("The weight in LB are :", lb)
print("The weight in KG are:", kg)
#list comprehensions
[print(i) for i in lb]
[print(i) for i in kg]