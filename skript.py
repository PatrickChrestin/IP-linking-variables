import os
import matplotlib.pyplot as plt

#######################
# plot runtimes for different k
#######################
# test_instance = "vpm2"
# runtimes = []
# # define range of k
# for k in range(2, 10):
#     with open(f"miplib2003/{test_instance}/runtime_k{k}.txt", "r") as file:
#         runtimes.append(float(file.readline()))
#         print(runtimes)
# plt.plot([i for i in range(2, 10)], runtimes)
# plt.title(f"Runtimes for {test_instance}")
# plt.xlabel("k")
# plt.ylabel("Runtime in seconds")
# plt.savefig(f"miplib2003/{test_instance}/runtime.png")


#######################
# plot objective values for different k
#######################
test_instance = "vpm2"
opt_values = [2,4,11,18,25,32,39,68]
plt.plot([i for i in range(2, 10)], opt_values)
plt.title(f"Objective values for {test_instance}")
plt.xlabel("k")
plt.ylabel("Objective value")
plt.savefig(f"miplib2003/{test_instance}/opt.png")