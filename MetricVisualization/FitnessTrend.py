import matplotlib.pyplot as plt
import numpy as np
import re
import os

print("You are here: " + os.getcwd())

individuals = [
        float(re.search("fitness: \((.*),.*\)", log).group(1))
        for log in open(input("Filename: "), "r") 
        if "was born!" in log]

individuals = np.array(individuals)

plt.plot(np.arange(len(individuals)), individuals, 'b-', lw=0.8)
plt.show()
