import nibabel as nib
from nilearn.masking import compute_epi_mask
from nilearn.masking import apply_mask
import pandas as pd
import numpy as np
from nilearn.glm.first_level import make_first_level_design_matrix
import readBoldData as rbd

# Load Data
data1 = nib.load('Data/sub-01_task-theoryofmindwithmanualresponse_run-01_bold.nii.gz')

# Mask Data
data1_mask = compute_epi_mask(data1, connected=False)
masked_data1 = apply_mask(data1, data1_mask)

# Load Events
fn_ev1 = 'Data/sub-01_task-theoryofmindwithmanualresponse_run-01_events.tsv'

df_events1 = pd.read_csv(fn_ev1,sep='\t')

stim_ev1 = df_events1[df_events1['trial_type'].notnull()]
df_ev1 = df_events1[df_events1['trial_type'].isnull()].reset_index(drop=True)
for r in range(len(stim_ev1)):
    df_ev1.at[int(stim_ev1.iloc[r,0]/2),'duration'] = stim_ev1.iloc[r,:]['duration']
    df_ev1.at[int(stim_ev1.iloc[r,0]/2),'trial_type'] = stim_ev1.iloc[r,:]['trial_type']

# Create Design Matrix
tr = 2.0 
n_scans = 179 
frame_times = np.arange(n_scans) * tr

conditions = df_ev1['trial_type']
duration = df_ev1['duration']
onsets = df_ev1['onset']

events = pd.DataFrame({'trial_type': conditions, 'onset': onsets, 'duration': duration})

X1 = make_first_level_design_matrix(frame_times, events)

# Calculate SNR
SNR_Image = rbd.SNR(X1,masked_data1)

# Download array
np.save('SNR', SNR_Image)