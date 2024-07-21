import sys
sys.path.append('../src')
import numpy as np

import src.DataGenerator

center_sets = [8500, 7000, 5666, 5000, 4250, 4000, 3400, 3000, 2833]
fault_sets = [75, 150, 300, 600]
spacing_sets = [0.02, 0.03, 0.04, 0.05, 0.06]
# center_sets = [5000, 4000, 3000]
# fault_sets = [50, 150, 300]
# spacing_sets = [0.03]
i = 0
for center in (center_sets):
    for fault in fault_sets:
        for spacing in spacing_sets:
            # ignore when center * spacing - 170 >= 20
            if np.abs(center * spacing - 170) >= 25:
                continue
            print(center, spacing, fault, center * spacing - 170)
            i = i+1
print(i)
