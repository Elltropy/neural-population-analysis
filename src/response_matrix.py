import pandas as pd
import numpy as np


def build_response_matrix(data, session_id):
    
    subdata = data[data['ophys_session_id'] == session_id]


    if subdata.empty:
        raise ValueError('Session ID does not exist.')

    if not ((subdata['ophys_experiment_id'].nunique() == 1) and (subdata['ophys_experiment_id'].isna().sum() == 0)):
        raise ValueError('Invalid experiment IDs.')

    if not subdata['image_name'].isna().sum() == 0:
        raise ValueError('Missing image labels.')


    subsub = subdata[['cell_specimen_id', 'stimulus_presentations_id', 'mean_response', 'image_name']]


    if not subsub['cell_specimen_id'].isna().sum() == 0:
        raise ValueError('Missing cell IDs.')
    
    if not subsub['stimulus_presentations_id'].isna().sum() == 0:
        raise ValueError('Missing presentation IDs.')

    if not (subsub.groupby('stimulus_presentations_id')['image_name'].nunique() == 1).all():
        raise ValueError('More than one distinct image label associated with a single presentation.')


    piv = subsub.pivot(index = 'cell_specimen_id', columns= ['stimulus_presentations_id', 'image_name'], values='mean_response')

    mask_omitted = piv.columns.get_level_values(1) == 'omitted'
    no_omitted = piv.T.drop(piv.T.index[mask_omitted])

    Y_series = pd.Series(no_omitted.index.get_level_values(1), index=no_omitted.index.get_level_values(0), name='Images')
    no_omitted.index = no_omitted.index.get_level_values(0)
    X_df = no_omitted


    if X_df.empty:
        raise ValueError('No usable population responses.')

    if not np.isfinite(np.array(X_df)).all():
        raise ValueError('Non-finite response values.')

    if not X_df.index.equals(Y_series.index):
        raise ValueError('Index alignment failed.')

    if not X_df.index.is_unique:
        raise ValueError('Duplicate presentation IDs.')

    if not X_df.columns.is_unique:
        raise ValueError('Duplicate neuron IDs.')
    

    return X_df, Y_series
