import random
import os

fs = [1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000]

with open("foo.txt", "a") as f:
    for i in range(5635):
        fslist = random.sample(fs, 1)[0]
        lambda1 = random.uniform(0.00015, 0.0015)
        width = random.uniform(0.000075, 0.00075)
        height = random.uniform(0.001, 0.01)
        kerf = random.uniform(0.000002, 0.000025)

        # print("{}".format(fslist), "{:.4f}".format(lambda1), "{:.5f}".format(width), "{:.4f}".format(height), "{:.6f}".format(kerf))

        f.write("\n{} {:.4f} {:.5f} {:.4f} {:.6f}".format(fslist, lambda1, width, height, kerf))