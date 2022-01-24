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
        self._default_plot_set()
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
        
    def plot_original_KM(self):
        self._default_plot_set()
        time = self.data[self.time_col]
        # convert to years
        # if max(time) > 20:
            # time = time / 365
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
        # calculate last survival score
        other_group_time_line = np.sort(np.array(time[~self.data_filter]))
        other_group_final_survival_score =  np.array(kmf.survival_function_at_times(other_group_time_line[-1], label=None)).item()

        # ax.set_xlim(0,10)
        results = logrank_test(time[self.data_filter], time[~self.data_filter], 
                               event[self.data_filter], event[~self.data_filter])
        self.original_pvalue = results.p_value
        # placeholder for the p-value
        ax.plot(0, 0, c='w', label='p-value={:.4f}'.format(results.p_value))
        ax.legend(loc='lower left')
        plt.title('Kaplan Meier plot')
        plt.xlabel('Time')
        plt.ylabel('Survival')
        
        # find better survival group and update filter
        better_group = self._calculate_better_survival_group(default_group_final_survival_score,other_group_final_survival_score)

        plt.show()

        return results.p_value, better_group

    def _calculate_better_survival_group(self,default_group_final_survival_score,other_group_final_survival_score):
        if default_group_final_survival_score > other_group_final_survival_score:
            print(f'Group {self.data[self.group_col].unique()[0]} with a better survival score: {default_group_final_survival_score:.2f}')
            return self.data[self.group_col].unique()[0]
        else:
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

    def compare_methods(self,max_intervals=20,parallel=False):
        methods = ['griddy','efficient']
        columns = ['Number_of_changes','Method','ORIGINAL_pvalue','MIN_pvalue','MAX_pvalue','RUNNING time']
        df_comparison = pd.DataFrame(columns=columns)
        number_of_changes = 0
        search = True
        while search:
        # for number_of_changes in np.arange(min_number_of_changes,max_number_of_changes+1):
            chosen_indexes = {}
            number_of_changes+=1
            for i,method in  enumerate(methods):
                original_pvalue, min_pvalue, result_max_pvalue, end_time = self.calc_interval(number_of_changes,delta=0, delta_model='RIGHT', method=method, parallel=parallel,print_results=False)
                chosen_indexes[i] = result_max_pvalue[1]
                df_temp = pd.DataFrame([[number_of_changes,method, original_pvalue, min_pvalue,result_max_pvalue[0], end_time]],columns=columns) 
                df_comparison = df_comparison.append(df_temp)
                # check if griddy and efficient select the same indexes
                if i == 1:
                    if(chosen_indexes[0]!=chosen_indexes[1]):
                        print(f'First time when two alogrithms disagree with P value MAX is On N={number_of_changes} :')
                        print(f'Greedy algorithm:')
                        print(chosen_indexes[0])
                        print(f'Efficient algorithm:')
                        print(chosen_indexes[1])
                        number_of_changes_result=number_of_changes # for the report
                        p_value_max_result= round(result_max_pvalue[0],4) # for the report
                        search = False
                    elif max_intervals == number_of_changes:
                        print(f'The two alogrithms agree with P value MAX up to max interval N={number_of_changes}:')
                        print(f'Greedy algorithm:')
                        print(chosen_indexes[0])
                        print(f'Efficient algorithm:')
                        print(chosen_indexes[1])
                        search = False
                        number_of_changes_result=None # for the report
                        p_value_max_result= None # for the report
                        break
        df_comparison = df_comparison.reset_index()
        df_comparison = df_comparison.drop(columns='index')
        if number_of_changes>10:
            display(df_comparison.loc[number_of_changes*2-8:])
        else:
            display(df_comparison.loc[number_of_changes*2-2:])
        
        self.plot_KM_with_chosen_indexes(chosen_indexes)
        self._default_plot_set()
        sns.lineplot(data=df_comparison, x="Number_of_changes", y="RUNNING time",hue="Method")
        plt.title('Time Running griddy VS Efficient')
        return df_comparison,chosen_indexes,number_of_changes_result,p_value_max_result
    
    def explore_alpha_rejection_null_hypothesis(self,max_number_of_changes=10,delta=0.05, method='griddy', parallel=True):
        rejected= False
        columns = ['Number_of_changes','MIN_pvalue','MAX_pvalue']
        df = pd.DataFrame(columns=columns)
        for number_of_changes in np.arange(1,max_number_of_changes+1):
            original_pvalue, min_pvalue, max_pvalue, _ = self.calc_interval(number_of_changes,delta=0, delta_model='RIGHT', method=method, parallel=parallel,print_results=False)
            df_temp = pd.DataFrame([[number_of_changes,min_pvalue, max_pvalue[0]]],columns=columns) 
            df = df.append(df_temp)
            #  if we got value higher than delta rejectionn break the search
            if max_pvalue[0] > delta:
                rejected = True
                break
        if rejected:
            print(f"On number_of_changes={number_of_changes} and delta={delta}. The null hypothes is rejected. alpha= {round(number_of_changes/len(self.data),4)}")
            number_of_changes_results=number_of_changes # for history report
            alpha = round(number_of_changes/len(self.data),4)
        else:
            print(f"On number_of_changes={number_of_changes} and delta={delta}. The null hypothes not rejected")
            number_of_changes_results=None # for history report
            alpha = None
        df = df.reset_index()
        df = df.drop(columns='index')
        display(df.style.apply(self._highlight_rejection,axis=1))
        print(f"ORIGINAL P-vlaue: {original_pvalue:.4f}")
        return df,number_of_changes_results,round(max_pvalue[0],4),alpha

    
    def _highlight_rejection(self,row):    
        highlight = 'background-color: lightcoral;'
        default = ''
        # must return one string per cell in this row
        if row['MAX_pvalue'] > 0.05:
            return  [default,default,highlight]
        else:
            return [default,default,default]
            

    def run(self,report_name="",atricle="None",link="None",max_number_of_changes=20,delta=0.05):
        # create report
        display(Markdown(f'# Log Rank Stability Interval Report'))
        display(Markdown(f'### Article: {atricle}'))
        display(Markdown(f'### Link: {link}'))
        display(Markdown(f'## KM Plot'))
        p_value, better_group = self.plot_original_KM()
        display(Markdown(f'## Explore delta rejection'))
        df_find_alpha,number_of_changes_results,max_p_value,alpha = self.explore_alpha_rejection_null_hypothesis(max_number_of_changes=max_number_of_changes,
                                                            delta=delta,
                                                            method='efficient',
                                                            parallel=True)
        display(Markdown(f'## Compare time calculation between efficient and griddy methods'))
        df_time_compare,chosen_indexes,number_of_changes_c_results,max_p_value_c= self.compare_methods(max_intervals=max_number_of_changes,parallel=True)

        # save values to history
        columns = ['atricle','original p_value','better group survival','delta tested','max intervals tries',' k rejection on interval','alpha','p_value rejection','greedy algo disagree with efficient algo on interval','the p_value when disagreement','link']
        # df_final = pd.DataFrame(columns=columns)
        if os.path.isfile('reports/report.csv'):
            df_final = pd.read_csv('reports/report.csv')
            df_final = df_final.drop(columns='Unnamed: 0')
        else:
            df_final = pd.DataFrame(columns=columns)
        df_temp = pd.DataFrame([[atricle,round(p_value,4),better_group,delta,max_number_of_changes,number_of_changes_results,alpha,max_p_value,number_of_changes_c_results,max_p_value_c,link]],columns=columns) 
        df_final = df_final.append(df_temp)
        df_final.to_csv('reports/report.csv')

        # save report
        time.sleep(1)
        os.system(f"jupyter nbconvert Log_Rank_Stability_V1.ipynb --output reports/report_{report_name} --no-input --to webpdf --allow-chromium-download") 