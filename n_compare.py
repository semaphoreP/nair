__author__ = 'Jason Wang'
from nair import *
import matplotlib.pylab as plt
import numpy as np

wvs = np.arange(1.3,2.5,0.01)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(wvs, nMathar(wvs, 80000, 280)-1.0002216, 'b-', label="Mathar")
ax.plot(wvs, nRoe(wvs, 80000, 280)-1.0002216, 'g-', label="Roe")

ax.legend()
ax.set_xlabel("Wavelenght (micron)")
ax.set_ylabel("n-1.0002216")
ax.ticklabel_format(useOffset=False, style='sci')

plt.show()