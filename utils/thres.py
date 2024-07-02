f = open('../thres.txt')
max = -1
for line in f:
    if float(line[-8:-1]) > max:
        max = float(line[-8:-1])
print(max)
