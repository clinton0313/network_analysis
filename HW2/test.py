#%%
import matplotlib

#%%
import matplotlib.pyplot as plt
import numpy as np
from time import sleep
# %%

plt.ion()
fig, ax = plt.subplots()
x = np.random.randint(10, size=10)
y = np.random.randint(10, size=10)
ax.scatter(x, y)
# plt.show()
sleep(1)
for _ in range(3):
    plt.show(block=False)
    x = np.random.randint(10, size=10)
    y = np.random.randint(10, size=10)
    sc = ax.scatter(x,y)
    sc.set_visible(False)
    plt.show(block=False)
    sleep(1)
# plt.show(block=True)
    # fig.canvas.draw()
    # ax.redraw_in_frame()
    # fig.canvas.draw()
    # sleep(1)


# %%
