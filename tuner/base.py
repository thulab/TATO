import optuna
import logging

class Tuner:
    
    def __init__(self, direction, enqueue_param_dicts=None, mode=None, seed=0):
        sampler = optuna.samplers.TPESampler(seed=seed)
        self.study = optuna.create_study(direction=direction, sampler=sampler)
        self.random_study = optuna.create_study(direction=direction, sampler=optuna.samplers.RandomSampler(seed=seed))
        self.history_params_hash = set()
        self.max_repeat = 10000
        self.mode = mode
        if enqueue_param_dicts is not None:
            for param_dict in enqueue_param_dicts:
                self.study.enqueue_trial(param_dict)

    def pick_trial(self, distribution, fixed_params=None):
        waiting_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.WAITING]

        if not waiting_trials and fixed_params:
            self.study.enqueue_trial(fixed_params)

        trial = self.study.ask(distribution)
        repeat = self.max_repeat
        param_dict = trial.params
        while self.mode != 'test' and str(param_dict) in self.history_params_hash and repeat > 0:
            if fixed_params:
                self.random_study.enqueue_trial(fixed_params)

            param_dict = self.random_study.ask(distribution).params
            self.study.enqueue_trial(param_dict)
            trial = self.study.ask(distribution)
            repeat -= 1
        if repeat != self.max_repeat:
            logging.warning(f"Randomly choose param_dict {self.max_repeat - repeat} times!")
        if repeat == 0:
            raise Exception('All params have been tried!')
        self.history_params_hash.add(str(trial.params))
        return trial
    
    def tell(self, trial, metric):
        self.study.tell(trial, metric)
    
    def save_trials(self, filepath):
        df = self.study.trials_dataframe()
        df.to_csv(filepath)
