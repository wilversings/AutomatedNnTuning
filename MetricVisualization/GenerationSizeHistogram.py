import matplotlib.pyplot as plt
import numpy as np
import re
import os

print("You are here: " + os.getcwd())

fl = open(input("Filename: "), "r")
generations, avg_fit, best_fit, arch_size = zip(*[
        (float(re.search("individuals: ([0-9]*)", log).group(1)),
         float(re.search("avg fitness: \(([0-9.]*), .*\)", log).group(1)),#if (re.search("avg fitness: ([0-9\.]*)", log) is not None) else None,
         float(re.search("best's fitness: \(([0-9.]*), .*\)", log).group(1)),
         float(re.search("best's fitness: \(([0-9.]*), ([0-9.]*)\)", log).group(2)))
        for log in fl 
        if "Growing done for generation" in log])


gen_nr = len(generations)
generations = np.array(generations)

n = np.arange(gen_nr)

fig, ax1 = plt.subplots()
plt.title("Accuracy trend over 100 generations for P.3")
ax1.set_xlabel("Generation")
ax1.set_ylabel("Accuracy (higher is better)")

ax1.plot(n, avg_fit, 'b-', label='Average accuracy')
ax1.plot(n, best_fit, 'g-', label='Best\'s accuracy')
ax1.legend()

plt.xticks(n)
ax1.set_xticklabels([])

plt.show()

