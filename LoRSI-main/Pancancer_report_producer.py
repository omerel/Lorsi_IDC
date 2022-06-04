# Imports
import Consts
from lorsi import LoRSI
from IPython.display import display, Markdown
import pandas as pd
import os
import numpy as np
import warnings
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm_notebook
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import tqdm
import time


def compare_methods(lorsi,max_intervals=20,parallel=False):
        methods = ['efficient','griddy']
        columns = ['Number_of_changes','Method','ORIGINAL_pvalue','MIN_pvalue','MAX_pvalue','RUNNING time']
        df_comparison = pd.DataFrame(columns=columns)
        number_of_changes = 0
        search_up_to_alpha = True
        search_up_to_first_unequal = True
        while search_up_to_alpha:
        # for number_of_changes in np.arange(min_number_of_changes,max_number_of_changes+1):
            chosen_indexes = {}
            number_of_changes+=1
            number_of_changes_unequal = None
            p_value_max_unequal = None
            for i,method in  enumerate(methods):
                original_pvalue, min_pvalue, result_max_pvalue, end_time = lorsi.calc_interval(number_of_changes,delta=0.05, delta_model='RIGHT', method=method, parallel=parallel,print_results=False)
                chosen_indexes[i] = result_max_pvalue[1]
                df_temp = pd.DataFrame([[number_of_changes,method, original_pvalue, min_pvalue,result_max_pvalue[0], end_time]],columns=columns) 
                df_comparison = df_comparison.append(df_temp)
                # check if griddy and efficient select the same indexes
                if i == 1:
                    if((chosen_indexes[0]!=chosen_indexes[1]) & search_up_to_first_unequal):
                        number_of_changes_unequal = number_of_changes # for the report
                        p_value_max_unequal = round(result_max_pvalue[0],6) # for the report
                        search_up_to_first_unequal = False
                    if max_intervals == number_of_changes:
                        search_up_to_alpha = False
                        p_value_max_changes= round(result_max_pvalue[0],6) # for the report
        is_greedy_stable = search_up_to_first_unequal
        df_comparison = df_comparison.reset_index()
        df_comparison = df_comparison.drop(columns='index')
        return df_comparison,chosen_indexes,is_greedy_stable,number_of_changes_unequal,p_value_max_unequal,p_value_max_changes
    
def explore_alpha_rejection_null_hypothesis(lorsi,max_number_of_changes=10,delta=0.05, method='griddy',parallel=True,to_print=True):
    rejected= False
    columns = ['Number_of_changes','MIN_pvalue','MAX_pvalue']
    df = pd.DataFrame(columns=columns)
    for number_of_changes in np.arange(1,max_number_of_changes+1):
        temp_original_pvalue, temp_min_pvalue, temp_max_pvalue, _ = lorsi.calc_interval(number_of_changes,delta=delta, delta_model='RIGHT', method=method, parallel=parallel,print_results=False)
        #  if we got value higher than delta rejectionn break the search
        if temp_max_pvalue[0] > delta:
            rejected = True
            if number_of_changes == 1:
                max_pvalue = temp_original_pvalue
            break
        else:
            # save the last result
            original_pvalue, min_pvalue, max_pvalue = temp_original_pvalue, temp_min_pvalue, temp_max_pvalue[0]
            df_temp = pd.DataFrame([[number_of_changes,min_pvalue, max_pvalue]],columns=columns) 
            df = df.append(df_temp)
    if rejected:
        number_of_changes_results = number_of_changes-1 # for history report
        alpha = round(number_of_changes_results/len(lorsi.data),4)
        if to_print:
            print(f"On number_of_changes={number_of_changes_results} and delta={delta}. The null hypothes is rejected. alpha= {alpha}")
    else:
        if to_print:
            print(f"On number_of_changes={number_of_changes} and delta={delta}. The null hypothes not rejected")
        number_of_changes_results=None # for history report
        alpha = None
    df = df.reset_index(drop=True)
    if to_print:
        print(f"ORIGINAL P-vlaue: {original_pvalue:.6f}")
    return df,number_of_changes_results,round(max_pvalue,6),alpha

def report_one_csv(cancer_path,dic_result,cancer_type,alpha,delta,event_col,time_col,group_col,df_os_results,df_article,cancer_gene):
  if cancer_gene.endswith('.csv'):
        # Enter parameters
        data_path = os.path.join(cancer_path,cancer_gene)
        gene_name = cancer_gene.split('_')[1]
        # Create Lorsi
        lorsi = LoRSI(data_path= data_path,
                    event_col= event_col,
                    time_col= time_col,
                    group_col= group_col) 
        # get R pvlue
        r_pvalue = df_os_results[df_os_results['V1']==gene_name][f'{cancer_type}_pvalue'].item()
        article_pvalue= df_article[df_article['Gene_name']==gene_name]['pvalue'].item()
        # Find max n of changes by alpha
        temp_data = pd.read_csv(data_path)
        len_data = len(temp_data)
        MAX_N_OF_CHANGES = int(alpha*len_data)
        # Run report
        result = explore_data(lorsi,cancer_type=cancer_type,
                                        subject=gene_name,
                                        max_number_of_changes=MAX_N_OF_CHANGES,
                                        delta=delta,
                                        r_pvalue=r_pvalue,
                                        article_pvalue = article_pvalue,
                                        compare_griddy=True)
        dic_result[result[0]+'_'+result[1]] = result

def explore_data(lorsi,cancer_type,subject,max_number_of_changes=10,delta=0.05,compare_griddy=False,r_pvalue=0,article_pvalue=0):
        p_value, better_group = lorsi.plot_original_KM(plot_result=False)
        lorsi.update_data_filter(better_group)
        # if the original pvalue bigger than delta stop
        if p_value < delta:
            # Explore delta rejection'
            _,number_of_changes_results,max_p_value,alpha = explore_alpha_rejection_null_hypothesis(lorsi,max_number_of_changes=max_number_of_changes,
                                                                delta=delta,
                                                                method='efficient',
                                                                parallel=True,
                                                                to_print=False)
            # calc on alpha == 0.01
            _, _, p_value_upper_alpha_one, _ = lorsi.calc_interval(int(0.01*len(lorsi.data)),delta=delta, delta_model='RIGHT', method='efficient', parallel=True,print_results=False)
            p_value_upper_alpha_one = p_value_upper_alpha_one[0]
            # calc on alpha == 0.005
            _, _, p_value_one_k, _ = lorsi.calc_interval(1,delta=delta, delta_model='RIGHT', method='efficient', parallel=True,print_results=False)
            p_value_one_k = p_value_one_k[0]

            # Compare time calculation between efficient and griddy methods'
            if compare_griddy:
                _,_,is_greedy_stable,number_of_changes_unequal,p_value_max_unequal,p_value_max_changes = compare_methods(lorsi,max_intervals=max_number_of_changes,parallel=True)
            else:
                is_greedy_stable,number_of_changes_unequal,p_value_max_unequal,p_value_max_changes = None,None,None,None
        else:
            max_number_of_changes = None
            number_of_changes_results = None
            alpha = None
            max_p_value = None
            p_value_upper_alpha_one = None
            p_value_one_k = None
            is_greedy_stable = None
            number_of_changes_unequal = None
            p_value_max_unequal = None
            p_value_max_changes = None

        result = [cancer_type,subject,round(article_pvalue,6),round(r_pvalue,6),round(p_value,6),len(lorsi.data),better_group,delta,
            max_number_of_changes,number_of_changes_results,alpha,max_p_value,p_value_upper_alpha_one,p_value_one_k,
            is_greedy_stable,number_of_changes_unequal,p_value_max_unequal,p_value_max_changes] 
        return  result

def create_report(cancer_list,report_name,alpha=0.01,delta=0.05,user_root=None,save_to_csv=True):
  # read OS_survival_results
  if user_root == None:
    df_os_results = pd.read_csv(Consts.PATH_OS_RESULTS,delimiter='\t')
  else:
    df_os_results = pd.read_csv(os.path.join(user_root, 'results', 'OS_survival_results.txt'), delimiter='\t')

  # go over all cancer types (each one of cancer type represented as a folder)
  if user_root == None:
    list_dir = os.listdir(Consts.PATH)
  else:
    list_dir = os.listdir(os.path.join(user_root, 'data_for_LoRSI'))

  for cancer_type in list_dir:
      if user_root == None:
        cancer_path = os.path.join(Consts.PATH,cancer_type)
      else:
        cancer_path = os.path.join(user_root, 'data_for_LoRSI', cancer_type)
      if os.path.isdir(os.path.join(cancer_path)):
          if cancer_type in cancer_list:
              # read article results
              try:
                if user_root == None:
                  df_article = pd.read_excel(Consts.PATH_ARTICLE_RESULTS, header=1, usecols='A:E', sheet_name=Consts.CANCER_DIC[cancer_type])
                else:
                  df_article = pd.read_excel(os.path.join(user_root, 'suplimental_data/41598_2021_84787_MOESM1_ESM.xlsx'), header=1, usecols='A:E', sheet_name=Consts.CANCER_DIC[cancer_type])
              except Exception as e:
                print(e)
              cacner_genes_results = os.listdir(cancer_path)
              dic_result = {}
              func = partial(report_one_csv,cancer_path,dic_result,cancer_type,alpha,delta,Consts.EVENT_COL,Consts.TIME_COL,Consts.GROUP_COL,df_os_results,df_article)
              with tqdm.tqdm(desc=f'calc {cancer_type}', total=len(cacner_genes_results)) as pbar:
                with ThreadPoolExecutor(max_workers=8) as executor:
                    # Using a dict for preserving the downloaded file for each future, to store it as a failure if we need that
                    futures = {
                        executor.submit(func, cancer_gene): cancer_gene for cancer_gene in cacner_genes_results
                    }
                    for future in as_completed(futures):
                        if future.exception():
                            print(future.exception())
                        pbar.update(1)

  if save_to_csv:
    # create df
    columns = ['cancer type','gene','Article p_value','R p_value','Lorsi p_value','num of samples','better group survival','delta tested',
    'max k changes with chosen alpha','max k changes to stay stable','max alpha to stay stable','p_value upper with max changes k','p_value upper on alpha 0.01',
    'p_value upper one change','is greedy algo correct up to alpha','k changes to reach disagreement','p_value when disagreement','p_value greedy on alpha 0.01']
    if user_root == None:
      results_path = Consts.PATH_RESULT
    else:
      results_path = os.path.join(user_root, 'lorsi_results')
      
    if os.path.isfile(os.path.join(results_path,report_name)):
        df_final = pd.read_csv(os.path.join(results_path,report_name))
        df_final = df_final.drop(columns='Unnamed: 0')
    else:
        df_final = pd.DataFrame(columns=columns)

    for key in dic_result:
      data = dic_result[key]
      df_temp = pd.DataFrame([data],columns=columns)
      df_final = df_final.append(df_temp)
    df_final = df_final.reset_index(drop=True)
    df_final.to_csv(os.path.join(results_path,report_name))