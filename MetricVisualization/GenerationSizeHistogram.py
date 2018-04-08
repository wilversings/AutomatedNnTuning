import matplotlib.pyplot as plt
import numpy as np
import re

fl = open("general.log", "r")
generations = [
        float(re.search("individuals: ([0-9]*)", log).group(1))
        for log in fl 
        if "Growing done for generation" in log]


gen_nr = len(generations)
generations = np.array(generations)

n = np.arange(gen_nr)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.set_ylabel("Number of individuals")
ax2.set_ylabel("Average fitness")

ax1.bar(n, height=generations)
#ax2.plot(n, [0.1, 0.6, 0.7, 0.1], 'r-')
plt.xticks(n)
plt.xlabel("Generation")
plt.show()

