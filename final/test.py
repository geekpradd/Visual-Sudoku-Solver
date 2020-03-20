from digit_process import test
import os 

files = os.listdir("test")

total = 0
hits = 0

for f in files:
    result = test("test/" + f)
    total+=1
    if (result == int(f.split('.')[0][-1:])):
        hits += 1

    print("Got {0} Was {1}".format(result, f.split('.')[0][-1:]))

print ("Hits {0} Total {1}".format(hits, total))
