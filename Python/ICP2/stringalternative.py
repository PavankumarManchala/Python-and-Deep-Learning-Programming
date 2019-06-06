def string_alternative(string):
    str1 = []
    for a,items in enumerate(string):
        if a % 2 == 0:
            str1.append(items)
    out_str = ''.join(str1[:])
    print(out_str)
if __name__ == '__main__':

    string_alternative(input("Enter the string: "))