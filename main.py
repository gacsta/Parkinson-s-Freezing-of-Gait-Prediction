 #%%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk(''):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#%%
import glob
path = r'C:\Users\gabr8\Documents\GitHub\Kaggle\Parkinson's Freezing of Gait Prediction\tlvmc-parkinsons-freezing-gait-prediction'


#%%
tdcsfog_paths = glob.glob(path + '/train/tdcsfog/*')
defog_paths = glob.glob(path + '/train/defog/*')

tdcsfog_data = pd.DataFrame([])
defog_data = pd.DataFrame([])


#%%
for p in tdcsfog_paths:
    id = os.path.splitext(os.path.basename(p))[0]
    data = pd.read_csv(p)
    data['Id'] = id
    tdcsfog_data = pd.concat([tdcsfog_data, data], ignore_index = False, axis = 0)
    
for p in defog_paths:
    id = os.path.splitext(os.path.basename(p))[0]
    data = pd.read_csv(p)
    data['Id'] = id
    defog_data = pd.concat([defog_data, data], ignore_index = False, axis = 0)
    
tdcsfog_data.set_index('Id', inplace = True)
defog_data.set_index('Id', inplace = True)

#%%
tdcsfog_meta = pd.read_csv(path + '/tdcsfog_metadata.csv')
defog_meta = pd.read_csv(path + '/defog_metadata.csv')
task_meta = pd.read_csv(path + '/tasks.csv')
events_meta = pd.read_csv(path + '/events.csv')
subjects_meta = pd.read_csv(path + '/subjects.csv')

defog_meta.set_index('Id', inplace = True)
tdcsfog_meta.set_index('Id', inplace = True)
task_meta.set_index('Id', inplace = True)
events_meta.set_index('Id', inplace = True)
subjects_meta.set_index('Subject', inplace = True)


#%%
tdcsfog = tdcsfog_data.join(tdcsfog_meta)
tdcsfog.reset_index(drop = False, inplace = True)
tdcsfog = pd.merge(tdcsfog, subjects_meta, left_on='Subject', right_on='Subject')
tdcsfog.sort_values(by=['Id', 'Time'],inplace = True)

#%%
pd.set_option("display.max_rows", None, "display.max_columns", None)
# tdcsfog.head(1500)
tdcsfog.head(50)