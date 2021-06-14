import numpy as np
import matplotlib.pyplot as plt

from utils.my_utils import my_cycle


## my_cycle 시각화
cycle_x = np.linspace(0, 100, 1000)
cycle_y = np.array(list(map(my_cycle, cycle_x)))

plt.figure(figsize=(20, 5))
plt.plot(cycle_x, cycle_y)
print(f"최대값:\t{np.max(cycle_y)}")
print(f"최대값:\t{np.min(cycle_y)}")
print(f"경계값:\t{my_cycle(0)}\t(epoch이 0일 때)")
print(f"경계값:\t{my_cycle(40)}\t(epoch이 40일 때)")
plt.show()
