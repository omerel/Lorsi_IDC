import pandas as pd
import numpy as np
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter
from itertools import combinations
from tqdm import tqdm
import multiprocessing
from functools import partial
import itertools
import time
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
import os
import time



class LoRSI():
    
    def __init__(self, data_path, event_col, time_col, group_col):
        self.data = pd.read_csv(data_path)
        self.event_col = event_col
        self.time_col = time_col
        self.group_col = group_col
        self.data_filter = self.data[self.group_col] == self.data[self.group_col].unique()[0]# 0 is default
    

    def _default_plot_set(self):
        plt.figure(figsize=(15,10))

    def plot_KM_with_chosen_indexes(self,chosen_indexes):
        # self._default_plot_set()
        time = self.data[self.time_col]
        event = self.data[self.event_col]
        kmf = KaplanMeierFitter()
        kmf.fit(time[self.data_filter], event[self.data_filter], 
                label='{} (n = {})'.format(self.data[self.group_col].unique()[0],
                                           self.data[self.data_filter].shape[0]))
        ax = kmf.plot()

        # calculate last survival score
        default_group_time_line = np.sort(np.array(time[self.data_filter]))
        default_group_final_survival_score =  np.array(kmf.survival_function_at_times(default_group_time_line[-1], label=None)).item()

        kmf.fit(time[~self.data_filter], event[~self.data_filter], 
                label='{} (n = {})'.format(self.data[self.group_col].unique()[1],
                                           self.data[~self.data_filter].shape[0]))
        kmf.plot()
        results = logrank_test(time[self.data_filter], time[~self.data_filter], 
                               event[self.data_filter], event[~self.data_filter])
        self.original_pvalue = results.p_value
        # placeholder for the p-value
        ax.plot(0, 0, c='w', label='p-value={:.4f}'.format(results.p_value))
        ax.legend(loc='lower left')

        # plot griddy chosen indexes
        time_griddy = np.array(self.data.loc[chosen_indexes[0]][self.time_col])        
        survival_score_griddy = np.array(kmf.survival_function_at_times(time_griddy, label=None))
        plt.plot(time_griddy,survival_score_griddy,'sr',label="griddy")
        # plot efficeint chosen indexes
        time_griddy = np.array(self.data.loc[chosen_indexes[1]][self.time_col])        
        survival_score_griddy = np.array(kmf.survival_function_at_times(time_griddy, label=None))
        plt.plot(time_griddy,survival_score_griddy,'xb',label="efficeint")
        plt.legend(loc='lower left')
        plt.title('chosen indexes on Kaplan Meier plot')
        plt.xlabel('Time')
        plt.ylabel('Survival')
        
    def plot_original_KM(self,plot_result=True):
      if plot_result:
        self._default_plot_set()
      time = self.data[self.time_col]
      # convert to years
    #   if max(time) > 20:
    #       time = time / 365
      event = self.data[self.event_col]
      kmf = KaplanMeierFitter()
      kmf.fit(time[self.data_filter], event[self.data_filter], 
              label='{} (n = {})'.format(self.data[self.group_col].unique()[0],
                                          self.data[self.data_filter].shape[0]))
      if plot_result:
          ax = kmf.plot()

      # calculate last survival score
      default_group_time_line = np.sort(np.array(time[self.data_filter]))
      default_group_final_survival_score =  np.array(kmf.survival_function_at_times(default_group_time_line[-1], label=None)).item()

      kmf.fit(time[~self.data_filter], event[~self.data_filter], 
              label='{} (n = {})'.format(self.data[self.group_col].unique()[1],
                                          self.data[~self.data_filter].shape[0]))
      if plot_result:
          kmf.plot()
      # calculate last survival score
      other_group_time_line = np.sort(np.array(time[~self.data_filter]))
      other_group_final_survival_score =  np.array(kmf.survival_function_at_times(other_group_time_line[-1], label=None)).item()

    #   ax.set_xlim(0,20)
    #   ax.set_xlabel('years')
      results = logrank_test(time[self.data_filter], time[~self.data_filter], 
                              event[self.data_filter], event[~self.data_filter])
      self.original_pvalue = results.p_value
      # placeholder for the p-value

      if plot_result:
          ax.plot(0, 0, c='w', label='p-value={:.4f}'.format(results.p_value))
          ax.legend(loc='lower left')
          plt.title('Kaplan Meier plot')
          plt.xlabel('Time')
          plt.ylabel('Survival')
          plt.show()
      
      # find better survival group and update filter
      better_group = self._calculate_better_survival_group(default_group_final_survival_score,other_group_final_survival_score,to_print=False)

      return results.p_value, better_group

    def _calculate_better_survival_group(self,default_group_final_survival_score,other_group_final_survival_score,to_print=True):
        if default_group_final_survival_score > other_group_final_survival_score:
            if to_print:
                print(f'Group {self.data[self.group_col].unique()[0]} with a better survival score: {default_group_final_survival_score:.2f}')
            return self.data[self.group_col].unique()[0]
        else:
            if to_print:
                print(f'Group {self.data[self.group_col].unique()[1]} with a better survival score: {other_group_final_survival_score:.2f}')
            self.data_filter = self.data[self.group_col] == self.data[self.group_col].unique()[1]
            return self.data[self.group_col].unique()[1]
        
        
    def update_data_filter(self, better_survival_group): 
        self.data_filter = self.data[self.group_col] == better_survival_group
        
    def calc_interval(self, number_of_changes, delta, delta_model, method='efficient', parallel=True, print_results=True):
        start = time.time()
        # with alpha > 1/n we use delta=0
        if number_of_changes > 1:
            max_pvalue_ommits = 0
            min_pvalue_ommits = 0    
        # with alpha = 1/n we can use any delta
        else:
            method='efficient' #delta not implemnted in griddy method and it doesn't matter which method in this case. therefore choose efficient
            delta_number = int(delta * self.data.shape[0])
            if delta_model == 'RIGHT':
                max_pvalue_ommits = delta_number
                min_pvalue_ommits = 0
            elif delta_model == 'LEFT':
                max_pvalue_ommits = 0
                min_pvalue_ommits = delta_number
            else:
                delta_number = int(delta_number / 2)
                max_pvalue_ommits = delta_number
                min_pvalue_ommits = delta_number
        min_pvalue, max_pvalue, p_values = self._calc_interval(number_of_changes, min_pvalue_ommits, max_pvalue_ommits, method, parallel)
        end_time = time.time() - start
        if print_results:
            print('ORIGINAL p-values: {}'.format(self.original_pvalue))
            print('MIN p-value      : {}'.format(min_pvalue))
            print('MAX p-value      : {}'.format(max_pvalue))
            print('Running time (seconds):', end_time)
        return self.original_pvalue,min_pvalue, max_pvalue, end_time
    
    def _calc_interval(self, number_of_changes, min_pvalue_ommits, max_pvalue_ommits, method, parallel):
        if method == 'efficient':
            max_pvalue,res_idxs = self._get_max_pvalue_efficient(number_of_changes, max_pvalue_ommits)
            min_pvalue = self._get_min_pvalue_efficient(number_of_changes, min_pvalue_ommits)
            p_values = None
        elif method == 'BF':
            min_pvalue, max_pvalue, p_values = self._get_pvalues_BF(number_of_changes, min_pvalue_ommits, max_pvalue_ommits, parallel)
        elif method == 'griddy': 
            max_pvalue,res_idxs = self._get_max_pvalue_griddy(number_of_changes, max_pvalue_ommits)
            min_pvalue = self._get_min_pvalue_griddy(number_of_changes, min_pvalue_ommits)
            p_values = None
        else:
            print('Invalid method')
            return None, None
        return min_pvalue, (max_pvalue,res_idxs), p_values
    
    def _change_filter(self, changed_indexes):
        new_filter = self.data_filter.copy()
        if isinstance(changed_indexes, np.int64):
            new_filter[changed_indexes] = ~new_filter.iloc[changed_indexes]
        else:
            new_filter.iloc[list(changed_indexes)] = ~new_filter.iloc[list(changed_indexes)]
        return new_filter   
    
    def _parallel_BF(self, idxs):
        time = self.data[self.time_col]
        event = self.data[self.event_col]
        changed_filter = self._change_filter(idxs)
        res = logrank_test(time[changed_filter], time[~changed_filter], 
                           event[changed_filter], event[~changed_filter])
        return res.p_value
    
    def _get_pvalues_BF(self, number_of_changes, min_pvalue_ommits, max_pvalue_ommit, parallel):
        if parallel:
            p_values = []
            for i in range(1, number_of_changes+1):
                print('Calculating p-values for {} change\s'.format(i))
                idxs_combs = list(combinations(range(self.data.shape[0]), i))
                pool = multiprocessing.Pool(multiprocessing.cpu_count())
                p_values += list(tqdm(pool.imap(partial(self._parallel_BF), idxs_combs), total=len(idxs_combs)))
#             p_values = pool.map(self._parallel_BF, idxs_combs)
        else:
            time = self.data[self.time_col]
            event = self.data[self.event_col]
            p_values = []
            for idxs in tqdm(list(combinations(range(self.data.shape[0]), number_of_changes))):
                changed_filter = self._change_filter(idxs)
                res = logrank_test(time[changed_filter], time[~changed_filter], 
                                   event[changed_filter], event[~changed_filter])
                p_values.append(res.p_value)
        p_values = sorted(p_values)
        return p_values[min_pvalue_ommits], p_values[-1*max_pvalue_ommit - 1], p_values
    
    def _errors_idxs(self, e):
        for c in itertools.combinations(range(e+2), 2):
            yield [b-a-1 for a, b in zip((-1,)+c, c+(e+2,))]
    
    def _get_max_pvalue_efficient(self, number_of_changes, num_of_ommits):
        time = self.data[self.time_col]
        event = self.data[self.event_col]
        group_data = self.data[self.data_filter]
        non_group_data = self.data[~self.data_filter]
        event_group_data = group_data[group_data[self.event_col] == 1].sort_values(by=self.time_col, ascending=False)
        event_non_group_data = non_group_data[non_group_data[self.event_col] == 1].sort_values(by=self.time_col, ascending=True)
        censorship_non_group_data = non_group_data[non_group_data[self.event_col] == 0].sort_values(by=self.time_col, ascending=False)
        event_group_data_index = event_group_data.index
        event_non_group_data_index = event_non_group_data.index
        censorship_non_group_data_index = censorship_non_group_data.index
        current_idx = np.array([0, 0, 0])
        res_idxs = []
        for i in range(num_of_ommits + 1):
            best_pvalue = 0
            for number_of_changes_g1, number_of_changes_g2, number_of_changes_g3 in self._errors_idxs(number_of_changes):
                idxs_of_changes_g1 = event_group_data_index[current_idx[0]:number_of_changes_g1 + current_idx[0]]
                idxs_of_changes_g2 = event_non_group_data_index[current_idx[1]:number_of_changes_g2 + current_idx[1]]
                idxs_of_changes_g3 = censorship_non_group_data_index[current_idx[2]:number_of_changes_g3 + current_idx[2]]
                idxs_to_change = list(idxs_of_changes_g1) + list(idxs_of_changes_g2) + list(idxs_of_changes_g3)
                changed_filter = self._change_filter(idxs_to_change)
                c_res = logrank_test(time[changed_filter], time[~changed_filter], 
                                     event[changed_filter], event[~changed_filter])
                if c_res.p_value > best_pvalue:
                    add_idx = np.array([number_of_changes_g1, number_of_changes_g2, number_of_changes_g3])
                    best_pvalue = c_res.p_value
                    res_idxs = idxs_to_change
            current_idx += add_idx 
        changed_filter = self._change_filter(res_idxs)
        res = logrank_test(time[changed_filter], time[~changed_filter], event[changed_filter], event[~changed_filter])
        return res.p_value,res_idxs

    def _get_min_pvalue_efficient(self, number_of_changes, num_of_ommits):
        time = self.data[self.time_col]
        event = self.data[self.event_col]
        group_data = self.data[self.data_filter]
        non_group_data = self.data[~self.data_filter]
        event_group_data = group_data[group_data[self.event_col] == 1].sort_values(by=self.time_col, ascending=True)
        event_non_group_data = non_group_data[non_group_data[self.event_col] == 1].sort_values(by=self.time_col, ascending=False)
        censorship_group_data = group_data[group_data[self.event_col] == 0].sort_values(by=self.time_col, ascending=False)
        event_group_data_index = event_group_data.index
        event_non_group_data_index = event_non_group_data.index
        censorship_group_data_index = censorship_group_data.index
        current_idx = np.array([0, 0, 0])
        res_idxs = []
        for i in range(num_of_ommits + 1):
            best_pvalue = 1
            for number_of_changes_g1, number_of_changes_g2, number_of_changes_g3 in self._errors_idxs(number_of_changes):
                idxs_of_changes_g1 = event_group_data_index[current_idx[0]:number_of_changes_g1 + current_idx[0]]
                idxs_of_changes_g2 = event_non_group_data_index[current_idx[1]:number_of_changes_g2 + current_idx[1]]
                idxs_of_changes_g3 = censorship_group_data_index[current_idx[2]:number_of_changes_g3 + current_idx[2]]
                idxs_to_change = list(idxs_of_changes_g1) + list(idxs_of_changes_g2) + list(idxs_of_changes_g3)
                changed_filter = self._change_filter(idxs_to_change)
                c_res = logrank_test(time[changed_filter], time[~changed_filter], 
                                     event[changed_filter], event[~changed_filter])
                if c_res.p_value < best_pvalue:
                    add_idx = np.array([number_of_changes_g1, number_of_changes_g2, number_of_changes_g3])
                    best_pvalue = c_res.p_value
                    res_idxs = idxs_to_change
            current_idx += add_idx            
        changed_filter = self._change_filter(res_idxs)
        res = logrank_test(time[changed_filter], time[~changed_filter], event[changed_filter], event[~changed_filter])
        return res.p_value

    def _get_max_pvalue_griddy(self, number_of_changes, num_of_ommits):
        time = self.data[self.time_col]
        event = self.data[self.event_col]
        group_data = self.data[self.data_filter]
        non_group_data = self.data[~self.data_filter]
        event_group_data = group_data[group_data[self.event_col] == 1].sort_values(by=self.time_col, ascending=False)
        event_non_group_data = non_group_data[non_group_data[self.event_col] == 1].sort_values(by=self.time_col, ascending=True)
        censorship_non_group_data = non_group_data[non_group_data[self.event_col] == 0].sort_values(by=self.time_col, ascending=False)
        event_group_data_index = event_group_data.index
        event_non_group_data_index = event_non_group_data.index
        censorship_non_group_data_index = censorship_non_group_data.index
        current_idx = np.array([0, 0, 0])
        recall_idx = np.array([0, 0, 0])
        res_idxs = []

        for i in range(number_of_changes):
            best_pvalue = 0
            for number_of_changes_g1, number_of_changes_g2, number_of_changes_g3 in self._errors_idxs(1):
                idxs_of_changes_g1 = event_group_data_index[current_idx[0]:number_of_changes_g1 + recall_idx[0]]
                idxs_of_changes_g2 = event_non_group_data_index[current_idx[1]:number_of_changes_g2 + recall_idx[1]]
                idxs_of_changes_g3 = censorship_non_group_data_index[current_idx[2]:number_of_changes_g3 + recall_idx[2]]
                idxs_to_change = list(idxs_of_changes_g1) + list(idxs_of_changes_g2) + list(idxs_of_changes_g3)
                changed_filter = self._change_filter(idxs_to_change)
                c_res = logrank_test(time[changed_filter], time[~changed_filter], 
                                     event[changed_filter], event[~changed_filter])
                if c_res.p_value > best_pvalue:
                    add_idx = np.array([number_of_changes_g1, number_of_changes_g2, number_of_changes_g3])
                    best_pvalue = c_res.p_value
                    res_idxs = idxs_to_change
            recall_idx += add_idx 

        changed_filter = self._change_filter(res_idxs)
        res = logrank_test(time[changed_filter], time[~changed_filter], event[changed_filter], event[~changed_filter])
        return res.p_value,res_idxs

    def _get_min_pvalue_griddy(self, number_of_changes, num_of_ommits):
        time = self.data[self.time_col]
        event = self.data[self.event_col]
        group_data = self.data[self.data_filter]
        non_group_data = self.data[~self.data_filter]
        event_group_data = group_data[group_data[self.event_col] == 1].sort_values(by=self.time_col, ascending=True)
        event_non_group_data = non_group_data[non_group_data[self.event_col] == 1].sort_values(by=self.time_col, ascending=False)
        censorship_group_data = group_data[group_data[self.event_col] == 0].sort_values(by=self.time_col, ascending=False)
        event_group_data_index = event_group_data.index
        event_non_group_data_index = event_non_group_data.index
        censorship_group_data_index = censorship_group_data.index
        current_idx = np.array([0, 0, 0])
        recall_idx = np.array([0, 0, 0])
        res_idxs = []
        for i in range(number_of_changes):
            best_pvalue = 1
            for number_of_changes_g1, number_of_changes_g2, number_of_changes_g3 in self._errors_idxs(1):
                idxs_of_changes_g1 = event_group_data_index[current_idx[0]:number_of_changes_g1 + recall_idx[0]]
                idxs_of_changes_g2 = event_non_group_data_index[current_idx[1]:number_of_changes_g2 + recall_idx[1]]
                idxs_of_changes_g3 = censorship_group_data_index[current_idx[2]:number_of_changes_g3 + recall_idx[2]]
                idxs_to_change = list(idxs_of_changes_g1) + list(idxs_of_changes_g2) + list(idxs_of_changes_g3)
                changed_filter = self._change_filter(idxs_to_change)
                c_res = logrank_test(time[changed_filter], time[~changed_filter], 
                                     event[changed_filter], event[~changed_filter])
                if c_res.p_value < best_pvalue:
                    add_idx = np.array([number_of_changes_g1, number_of_changes_g2, number_of_changes_g3])
                    best_pvalue = c_res.p_value
                    res_idxs = idxs_to_change
            recall_idx += add_idx            
        changed_filter = self._change_filter(res_idxs)
        res = logrank_test(time[changed_filter], time[~changed_filter], event[changed_filter], event[~changed_filter])
        return res.p_value