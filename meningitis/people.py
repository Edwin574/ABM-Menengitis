# import numpy as np
# import pandas as pd
import starsim as ss

class People(ss.People):
    def update_post(self, sim):
        self.age[self.alive] += 1/365
        return self.age
