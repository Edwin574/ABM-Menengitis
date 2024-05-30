import starsim as ss
import numpy as np
from model import Meningitis

meningitis=Meningitis()
class Treatment(ss.Intervention):
    def __init__(self, timestep=10, prob=0.5, mean_dur_infection=5):
        super().__init__()
        self.timestep = timestep
        self.prob = prob
        self.dur_infection = ss.normal(mean_dur_infection, 1)

    def apply(self, sim):
        if sim.ti == self.timestep:
            meningitis = sim.diseases.meningitis
            eligible_ids = sim.people.uid[meningitis.ti_infected == sim.ti]
            n_eligible = len(eligible_ids)
            is_treated = np.random.rand(n_eligible) < self.prob
            treat_ids = eligible_ids[is_treated]
            dur_inf = self.dur_infection.rvs(treat_ids)
            will_die = meningitis.pars.p_death.rvs(treat_ids)
            dead_uids = treat_ids[will_die]
            recover_uids = treat_ids[~will_die]
            meningitis.ti_dead[dead_uids] = meningitis.ti_infected[dead_uids] + dur_inf[will_die] / sim.dt
            meningitis.ti_recovered[recover_uids] = meningitis.ti_infected[recover_uids] + dur_inf[~will_die] / sim.dt