import matplotlib.pyplot as plt
import numpy as np
import re
import os
from dateutil import parser

print("You are here: " + os.getcwd())

fl = open(input("Filename: "), "r")
t0 = parser.parse(next(fl).split(' ')[1])
generations, time_ticks = zip(*[
        [float(re.search("individuals: ([0-9]*)", log).group(1)),
         parser.parse(re.search("([0-9][0-9]:[0-6][0-9]:[0-6][0-9])", log).group(1))]
        for log in fl
        if "Growing done for generation" in log])

metric_name = "seconds to raise"
time_ticks = [(x - t0).total_seconds() for x in time_ticks]
n = np.arange(len(generations))

fig, ax1 = plt.subplots()
plt.title("{} over 100 generations for P.2".format(metric_name.capitalize()))
ax1.set_xlabel("Generation")
ax1.set_ylabel("{} (lower is better)".format(metric_name.capitalize()))

ax1.plot(n, time_ticks, 'b-', label=metric_name.capitalize())
ax1.plot(n, [time_ticks[0] * (i + 1) for i in n], 'g-', label="Ideal")
ax1.legend()

plt.xticks(n)
ax1.set_xticklabels([])

plt.show()

