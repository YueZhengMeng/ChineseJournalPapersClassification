import os

os.system("python preTrain.py -n 0 -t long")
for i in range(1, 13):
    os.system("python fineTune1.py -n 0 -t long -i " + str(i))

with open("result.txt", "a") as f:
    f.write("---------------------------" + "\n")

os.system("python preTrain.py -n 0 -t short")
for i in range(1, 13):
    os.system("python fineTune1.py -n 0 -t short -i " + str(i))

with open("result.txt", "a") as f:
    f.write("---------------------------" + "\n")

os.system("python preTrain.py -n 0 -t nsp")
for i in range(1, 13):
    os.system("python fineTune1.py -n 0 -t nsp -i " + str(i))
