import matplotlib.pyplot as plt
import numpy as np
import re

individuals = [
        float(re.search("fitness: (.*)", log).group(1)) 
        for log in open("general.log", "r") 
        if "was born!" in log]

individuals = np.array(individuals)

plt.plot(np.arange(len(individuals)), individuals, 'b-', lw=0.8)
plt.show()
