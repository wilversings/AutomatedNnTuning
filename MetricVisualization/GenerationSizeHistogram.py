import matplotlib.pyplot as plt
import numpy as np
import re

generations = [
        float(re.search("individuals: ([0-9]*)", log).group(1))
        for log in open("general.log", "r") 
        if "Growing done for generation" in log]

generations = np.array(generations)

n = np.arange(len(generations))
plt.bar(n, height=generations)
plt.xticks(n)
plt.xlabel("Generation")
plt.ylabel("Number of individuals")
plt.show()

