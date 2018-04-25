import sys
import json
import numpy as np
import matplotlib.pyplot as plt

PATH_1 = sys.argv[1]
PATH_2 = sys.argv[2]

results_1 = json.load(open(PATH_1, "r"))
results_2 = json.load(open(PATH_2, "r"))

probas_1 = np.sum(results_1["classes_probas"], axis=1)
probas_2 = np.sum(results_2["classes_probas"], axis=1)

med_1 = np.median(probas_1)
med_2 = np.median(probas_2)

avg_1 = np.mean(probas_1)
avg_2 = np.mean(probas_2)

x_1 = np.arange(probas_1.size)
x_2 = np.arange(probas_2.size) + probas_1.size

plt.scatter(x_1, probas_1)
plt.scatter(x_2, probas_2)
plt.plot(x_1, np.ones(x_1.size) * med_1, color="red")
plt.plot(x_2, np.ones(x_2.size) * med_2, color="red")
plt.plot(x_1, np.ones(x_1.size) * avg_1, color="green")
plt.plot(x_2, np.ones(x_2.size) * avg_2, color="green")
plt.show()
