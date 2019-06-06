# Take the file input
fileName= input("Please give the filename? ")
infile= open(fileName,'r')
strDict = {}
count = 0
txt = ""
# read line by line
line = infile.readline()
while line != "":
    # read each word separated by space
    for xStr in line.split(" "):
        # remove next line notation and convert to lwercase
        onlystr = xStr.split("\n")[0].lower()
        # check if the word is present in the list
        if onlystr in strDict:
          count = strDict[onlystr] + 1
        else:
          count = 1
        # add or update the count to word
        strDict[onlystr] = count
    line = infile.readline()
# open the file to write as appending
with open(fileName, 'a') as file:
   for item, values in strDict.items():
       txt += "\n"+item+" "+str(values)+"\n"
   print(txt)
   file.write(txt)