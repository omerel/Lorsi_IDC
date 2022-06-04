PATH = '../Pancancer survival analysis of cancer hallmark genes/data_for_LoRSI'
PATH_RESULT = '../Pancancer survival analysis of cancer hallmark genes/lorsi_results'
PATH_OS_RESULTS='../Pancancer survival analysis of cancer hallmark genes/results/OS_survival_results.txt'
PATH_ARTICLE_RESULTS='../Pancancer survival analysis of cancer hallmark genes/suplimental_data/41598_2021_84787_MOESM1_ESM.xlsx'
EVENT_COL = 'event'
TIME_COL = 'time'
GROUP_COL = 'group'
CANCER_DIC = {'Hematopoietic and reticuloendothelial systems':'AML',
                'Bladder':'Bladder',
                'Breast':'Breast',
                'Cervix uteri':'Cervical',
                'Colon':'Colon',
                'Esophagus':'Esophagus',
                'Brain':'Glioblastoma',
                'Larynx':'Head and neck',
                'Liver and intrahepatic bile ducts':'Liver',
                'Skin':'Melanoma',
                'Ovary':'Ovarium',
                'Pancreas':'Pancreas',
                'Prostate gland':'Prostate',
                'Connective, subcutaneous and other soft tissues':'Sarcoma',
                'Stomach':'Stomach',
                'Heart, mediastinum, and pleura':'Thymoma',
                'Thyroid gland':'Thyroid',
                'Uterus, NOS':'Uterine'}