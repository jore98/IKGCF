import numpy as np

with open('kg_final.txt', 'r') as file:
    for line in file.readlines():
        temp = line.strip().split('\n')[0]
        e1, e2 = temp[0], temp[2]
        if int(e1) == 30527 and int(e2) == 24917:
            print('duicheng')
