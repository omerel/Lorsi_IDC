from gdc_access import gdc
import os.path as path
import numpy as np
import pandas as pd
import os
import pathlib 
from tqdm.notebook import tqdm_notebook

default_cases_fields = [
  'case_id',
  'sample_ids',
  'primary_site',
  'disease_type',
  'diagnoses.age_at_diagnosis',
  'demographic.days_to_death',
  'demographic.vital_status', 
  'diagnoses.days_to_last_follow_up', 
  'diagnoses.days_to_recurrence'
  ]

default_cases_columns = [
  'id', 
  'primary_site', 
  'disease_type', 
  'case_id', 
  'diagnoses', 
  'sample_ids', 
  'demographic']

def get_days_to_death(demographic):
  if demographic['vital_status'] == 'Dead':
    try:
      return int(demographic['days_to_death'])
    except Exception as e:
      return -1 
  else: 
    return -1

def get_months_to_death(demographic):
  if demographic['vital_status'] == 'Dead':
    try:
      return float(demographic['days_to_death'])/12
    except Exception as e:
      return -1.0 
  else: 
    return -1.0

def get_age_at_diag(diag):
  try:
    return int(diag[0]['age_at_diagnosis'])
  except Exception as e:
    return -1

def get_sample_id(sample_ids):
  try:
    return sample_ids[0]
  except Exception as e:
    return 'NA'

def set_os_for_live(os, diag):
  if os > 0: # If Dead
    return os
  else:
    try:
      new_os = diag[0]['days_to_last_follow_up']
      if new_os == 0:
        return -1 # Will be dropped later
      else:
        return new_os 
    except Exception as e:
      return -1

class PancancerSurvival():
  def __init__(self, work_dir, cases_fields = default_cases_fields, cases_columns = default_cases_columns):
      self.data_dir = path.join(work_dir, 'data')
      self.cases_fields = cases_fields
      self.cases_columns = cases_columns
      self.GDC = gdc(cases_fields=self.cases_fields, cases_columns=self.cases_columns)
      self.cases_df = self.GDC.query_cases()
      print(f"Cases dataframe shape: {self.cases_df.shape}")
      self.files_df = self.GDC.query_files()
      print(f"Files dataframe shape: {self.files_df.shape}")

  def get_cases_df(self):
      return self.cases_df

  def set_cases_df(self, cases_df):
      self.cases_df = cases_df

  def get_files_df(self):
      return self.files_df

  def process_cases_df(self):
      # Creat 'vital_status' and 'days_to_death' column out of 'demographic' column
      # and remove 'demographic'
      self.cases_df['vital_status'] = self.cases_df['demographic'].apply(lambda x: x['vital_status'])
      self.cases_df['days_to_death'] = self.cases_df['demographic'].apply(get_days_to_death)
      #self.cases_df['OS'] = (self.cases_df['days_to_death'].astype(float)/30).round(decimals=1)
      # max_os = self.cases_df['days_to_death'].max()
      self.cases_df['OS'] = self.cases_df.apply(lambda row: set_os_for_live(row['days_to_death'], row['diagnoses']), axis=1)
      # Drop cases that don't have OS
      self.cases_df = self.cases_df[self.cases_df['OS'] != -1]
      
      # For now we do not have the RFS data, but we need the column for the later 'R' processing 
      self.cases_df['RFS'] = self.cases_df['OS']
      #self.cases_df.drop(columns=['demographic'], inplace=True)

      # Creat 'age_at_diag_days' column out of 'diagnoses' column
      self.cases_df['age_at_diag_days'] = self.cases_df['diagnoses'].apply(get_age_at_diag)

      # 'id' and 'case_id' are identical
      # Make 'case_id' the index, and remove the two columns
      self.cases_df.rename(columns={'case_id':'Case ID'}, inplace=True)
      self.cases_df.set_index('Case ID', inplace=True)
      self.cases_df.drop(columns=['id'], inplace=True)

      # Set 'primary_site' column values as string
      self.cases_df['primary_site'] = self.cases_df['primary_site'].astype(str)

      # Keep only the first sample ID
      self.cases_df['Sample ID'] = self.cases_df['sample_ids'].apply(get_sample_id)

  def drop_sites_by_num_of_cases(self, min_num_cases):
      # Remove cases of sites with less than 100 cases
      site_list, site_cnt = np.unique(self.cases_df['primary_site'].to_numpy(), return_counts=True)
      for idx, c in enumerate(site_cnt):
        if c<min_num_cases:
          self.cases_df = self.cases_df[self.cases_df['primary_site'] != site_list[idx]]
  
  def process_files_df(self):
    # Convert the 'associated_entities' column that holds a list with one item, into 'case_id' column 
    self.files_df['case_id'] = self.files_df['associated_entities'].apply(lambda x: x[0]['case_id'])
    # For cases with more than one file, keep the first one
    self.files_df.drop_duplicates(subset=['case_id'], keep='first', ignore_index=True, inplace=True)
    # Make 'case_id' as index.
    # Will be needed later when merging with the cases dataframe
    self.files_df.rename(columns={'case_id':'Case ID'}, inplace=True)
    self.files_df.set_index('Case ID', inplace=True)
    # Remove the 'id' column which is a copy of the 'file_id', 
    # and the 'associated_entities' column that was copied to the 'case_id'
    self.files_df.drop(columns=['id', 'associated_entities'], inplace=True)

    self.files_df.rename(columns={'file_id':'File ID'}, inplace=True)
  
  def remove_site_from_cases(self, site):
    self.cases_df = self.cases_df[self.cases_df['primary_site']!=site]

  def merge_cases_files(self):
    self.merged_df = self.cases_df.merge(self.files_df, left_index=True, right_index=True)
  
  def get_site_list(self):
    try:
      return np.unique(self.merged_df['primary_site'].to_numpy())
    except Exception as e:
      self.merge_cases_files()
      return np.unique(self.merged_df['primary_site'].to_numpy())
  
  def save_all_sites_tsv_file(self):
    for site in tqdm_notebook(self.get_site_list()):
      self.save_site_tsv_file(site)

  def save_site_tsv_file(self, site):
    site_dir = path.join(self.data_dir, f"{site}_data")
    if not path.exists(site_dir):
      os.mkdir(site_dir)
    site_tsv = path.join(site_dir, "gdc_sample_sheet.tsv")
    site_df = self.merged_df[self.merged_df['primary_site'] == site]
    site_df.to_csv(site_tsv, sep="\t")

  def download_files_for_site(self, site, force=False):
    print(site)
    site_dir = path.join(self.data_dir, f"{site}_data")
    if not path.exists(site_dir):
      os.mkdir(site_dir)
    meta_data_tsv = path.join(site_dir, 'gdc_sample_sheet.tsv')
    meta_data_df = pd.read_csv(meta_data_tsv, sep='\t')
    meta_data_df.apply(lambda row: self.GDC.get_file(row['File ID'], row['file_name'], site_dir, force), axis = 1)

  def download_files_all_sites(self, force=False):
    for site in tqdm_notebook(self.get_site_list()):
      self.download_files_for_site(site, force)

  # Delete from the work directory files that are not in the metadata file
  def clean_files_for_site(self, site):
    site_dir = path.join(self.data_dir, f"{site}_data")
    if not path.exists(site_dir):
      return
    meta_data_tsv = path.join(site_dir, 'gdc_sample_sheet.tsv')
    meta_data_df = pd.read_csv(meta_data_tsv, sep='\t')
    expected_file_list = meta_data_df['file_name'].values
    for file_ in tqdm_notebook(pathlib.Path(site_dir).glob("*.htseq.counts.gz")):
      if file_.name not in expected_file_list:
        os.remove(file_)

  def clean_files_all_sites(self):
    for site in tqdm_notebook(self.get_site_list()):
      self.clean_files_for_site(site)


