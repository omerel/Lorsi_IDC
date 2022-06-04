import requests
import json
import pandas as pd
import re
import os.path as path

# Fields to include in the result of the case query
default_cases_fields = [
  "case_id",
  "sample_ids",
  "primary_site",
  "disease_type",
  "diagnoses.age_at_diagnosis",
  "demographic.days_to_death",
  "demographic.vital_status"
  ]

# Column names for the cases Dataframe
default_cases_columns = [
  'id', 
  'primary_site', 
  'disease_type', 
  'case_id', 
  'diagnoses', 
  'sample_ids', 
  'demographic'
  ]

# Fields to include in the result of the file query
default_files_fields = [
  "file_name",
  "file_id",
  "associated_entities.case_id",
  "metadata_files.type"
  ]

# Column names for the files Dataframe
default_files_columns = [
  'id', 
  'associated_entities', 
  'file_name', 
  'file_id'
  ]

class gdc():
  def __init__(self, query_size=2000, q_format='JSON', cases_fields = default_cases_fields, cases_columns = default_cases_columns, files_fields = default_files_fields, files_columns = default_files_columns):
    self.cases_endpt = 'https://api.gdc.cancer.gov/cases'
    self.cases_mapping_endpt = 'https://api.gdc.cancer.gov/cases/_mapping'
    self.files_endpt = 'https://api.gdc.cancer.gov/files'
    self.files_mapping_endpt = 'https://api.gdc.cancer.gov/files/_mapping'
    self.data_endpt = "https://api.gdc.cancer.gov/data"
    self.query_size = query_size
    self.q_format = q_format
    self.cases_fields = cases_fields
    self.cases_columns = cases_columns
    self.files_fields = files_fields
    self.files_columns = files_columns

  def get_current_cases_fields(self):
    return self.cases_fields

  def get_current_cases_columns(self):
    return self.cases_columns
  
  def get_current_files_fields(self):
    return self.files_fields

  def get_current_files_columns(self):
    return self.files_columns

  def get_cases_endpoint(self):
    return self.cases_endpt
    
  def query(self, end_point, fields, filters, columns, DEV=False):
  
    fields_list = ','.join(fields)

    params = {
        "filters": json.dumps(filters),
        "fields": fields_list,
        "format": self.q_format,
        "from": str(0),
        "size": str(self.query_size)
        }

    response = requests.get(end_point, params = params)

    if DEV:
      return response
      
    # Create empty Dataframe
    df = pd.DataFrame(columns=columns)
    # If the query was successful
    if response.status_code == 200:
      res_json = response.json()
      print(res_json['data']['pagination'])
      # Add the data to the DF
      for hit in res_json['data']['hits']:
        df = df.append(hit, ignore_index=True)
      # If needed, do pagination
      total = res_json['data']['pagination']['total']
      start_idx = res_json['data']['pagination']['from'] + res_json['data']['pagination']['size']
      while (response.status_code == 200) and (start_idx < total-1):
        params['from'] = str(start_idx)
        response = requests.get(end_point, params = params)
        res_json = response.json()
        print(res_json['data']['pagination'])
        if start_idx != res_json['data']['pagination']['from']:
          print("Error in pagination")
          break
        for hit in res_json['data']['hits']:
          df = df.append(hit, ignore_index=True)
        start_idx = res_json['data']['pagination']['from'] + res_json['data']['pagination']['size']
    else:
      print("Query failed with code: " + response.status_code)
      
    return df

  def query_cases(self, fields=None, columns=None):

    if fields != None:
        self.cases_fields = fields
        if columns==None:
            print("gdc.query_cases: when providing fields, you must also provide columns")
            return None
        else:
            self.cases_columns = columns
    elif columns != None:
        print("gdc.query_cases: when providing columns, you must also provide fields")
        return None

    filters = {
        "op":"and",
        "content":[
                  {
                      "op":"in",
                      "content":{
                        "field":"cases.files.experimental_strategy",
                        "value":"RNA-Seq"
                      }
                  },
                  {
                      "op":"in",
                      "content":{
                        'field':'demographic.vital_status',
                        'value':['alive', 'dead']
                      }
                  }
        ]
    }

    cases_df = self.query(end_point=self.cases_endpt, fields=self.cases_fields, filters=filters, columns=self.cases_columns)  
    return cases_df

  def filter_field_list(self, response, key):
    map_json = response.json()
    fields = map_json['fields']
    if key != None:
      return [f for f in fields if key in f]
    else:
      return fields

  def get_cases_fields(self, key=None):
    response = requests.get(self.cases_mapping_endpt)
    return self.filter_field_list(response, key)

  def query_files(self, fields=None, columns=None):

    if fields != None:
        self.files_fields = fields
        if columns==None:
            print("gdc.query_files: when providing fields, you must also provide columns")
            return None
        else:
            self.files_columns=columns
    elif columns != None:
        print("gdc.query_files: when providing columns, you must also provide fields")
        return None

    filters = {
        "op":"and",
        "content":[
                  {"op": "in",
                    "content":{
                        "field": "analysis.workflow_type",
                        "value": ["HTSeq - Counts"]
                        }
                    }
        ]
    }

    files_df = self.query(end_point=self.files_endpt, fields=self.files_fields, filters=filters, columns=self.files_columns)

    return files_df
  
  def get_files_fields(self, key=None):
    response = requests.get(self.files_mapping_endpt)
    return self.filter_field_list(response, key)

  # Download a file
  # If the file already exists in the target folder, 
  #  download it only if force is True
  def get_file(self, file_id, file_name, folder, force=False):
    file_path = path.join(folder, file_name)
    if not force:
      if path.exists(file_path):
        return
    ids = [file_id]
    params = {"ids": ids}

    response = requests.post(self.data_endpt, 
                            data = json.dumps(params), 
                            headers={"Content-Type": "application/json"})

    response_head_cd = response.headers["Content-Disposition"]
    rcv_file_name = re.findall("filename=(.+)", response_head_cd)[0]
    if rcv_file_name != file_name:
      print(f"Recieved file name\n{rcv_file_name}\nis not as expected\n{file_name}")
    else:      
      with open(file_path, "wb") as output_file:
        output_file.write(response.content) 
        output_file.close()



