class Employee:
  count = 0
  totalSal = 0
  x = 100

  def __init__(self, name, family, Address):
    self.name = name
    self.family = family
    self.salary = Employee.x
    self.address = Address
    Employee.count += 1
    Employee.totalSal += Employee.x

  def average(self):
    return self.totalSal/self.count

  def display(self):
    print("Name: ", self.name, " Family :", self.family, " Salary :", self.salary, " Address :",self.address)


class FulltimeEmployee(Employee):
    def __init__(self, n, f, a):
        Employee.__init__(self, n, f, a)


e1 = Employee("Sarena williams", "Tennis Family", "No man's land")
e2 = Employee("Dhoni", "Cricket Family", "India")
e3 = FulltimeEmployee("Pavan", "Manchala Family","5446 Charlotte St")
e4 = FulltimeEmployee("Tarun", "Kasturi Family","5428 Charlotte St")
print(e3.average())
print(e4.average())
e3.display()

