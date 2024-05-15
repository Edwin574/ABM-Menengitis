# interventions.py

import numpy as np
import starsim as ss

class Vaccine(ss.Intervention):
    def __init__(self, timestep=100, prob=0.5, imm_boost=2.0):
        super().__init__()
        self.timestep = timestep
        self.prob = prob
        self.imm_boost = imm_boost

    def apply(self, sim):
        if sim.ti == self.timestep:
            meningitis = sim.diseases.meningitis
            eligible_ids = sim.people.uid[meningitis.susceptible]
            n_eligible = len(eligible_ids)
            to_vaccinate = self.prob > np.random.rand(n_eligible)
            vaccine_ids = eligible_ids[to_vaccinate]
            meningitis.immunity[vaccine_ids] += self.imm_boost
