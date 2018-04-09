import matplotlib.pyplot as plt
import numpy as np
import re

fl = open(input("Filename: "), "r")
generations, avg_fit, best_fit = zip(*[
        (float(re.search("individuals: ([0-9]*)", log).group(1)), 
         (float(re.search("avg fitness: ([0-9\.]*)", log).group(1))) if (re.search("avg fitness: ([0-9\.]*)", log) is not None) else None,
         float(re.search("best's fitness: ([0-9\.]*)", log).group(1)))
        for log in fl 
        if "Growing done for generation" in log])


gen_nr = len(generations)
generations = np.array(generations)

n = np.arange(gen_nr)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.set_xlabel("Generation")
ax1.set_ylabel("Number of individuals")
ax2.set_ylabel("Average fitness")

ax1.bar(n, height=generations)
ax2.plot(n, avg_fit, 'bo-')
ax2.plot(n, best_fit, 'go-')
plt.xticks(n)
plt.show()

