import numpy as np
import pandas as pd
import re

# note: disk is hard-coded here, change to your own location :)
folder = 'X:\\projects\\Advies_R&D\\Wapens en Munitie\\Ammo_LRs\\data\\input\\'

for trace in ['slagpinindruk_02-', 'stootbodemindruk_01-', 'slagpingatschaaf_01-']:

    # de boel inlezen en samenvoegen
    df_basis = pd.read_csv(folder + trace + 'basis.csv')
    df_extra = pd.read_csv(folder + trace + 'extra.csv')
    df_merged = pd.merge(df_basis, df_extra, on='Id', how='outer')

    # alleen de relevante rijen eruit filteren: fiocchi ammo, en één vergelijking per wapenpaar
    selected_rows = (df_merged['Ammo'] == 'F-F') & df_merged['Pair1st']
    df_filtered = df_merged[selected_rows]

    # verwijder specifieke datapunten/wapens (zie ...\20210501_ammolr\FSM_only\validatie_analyses\analyse_dataset.m),
    # op deze manier worden het dezelfde datasets (aantallen KM en KNM) als in het Scratch-artikel zijn gebruikt.
    # For breech face impression; remove the data points with an ACCF above  0.95. A few data points were identiefied in
    # the region that were accidentally obtained by comparing duplicates of the same image. The ACCF is not exactly 1
    # because a manual cropping step is involved in the score calculation procedure.
    if 'stootbodemindruk' in trace:
        df_filtered = df_filtered.drop(df_filtered.index[df_filtered['ACCF'] > 95])
    # For aperture shear data; remove the weapon that has a very low KM-score for F-F ammo. Visual inspection of the
    # marks that belongs to this score revealed unusual behavior that would not be processed in a default manner, if
    # encountered in a real case.
    if 'slagpingatschaaf' in trace:
        comparison_id = df_filtered.index[(df_filtered['CCF_5_250'] < 0.5) & (df_filtered['Source'] == 'common')].values[0]
        weapon_id = df_filtered.at[comparison_id, 'Weapon1']
        comparison_ids = (df_filtered['Weapon1'] == weapon_id) | (df_filtered['Weapon2'] == weapon_id)
        df_filtered = df_filtered.drop(df_filtered.index[comparison_ids])

    # nieuwe tabel opbouwen
    df_new = pd.DataFrame()
    # geanonimiseerde wapen-IDs, afgekorte hashes; check dat er geen overlap in zit
    df_new['weapon1'] = pd.Series(pd.util.hash_array(df_filtered['Weapon1'].to_numpy(), categorize=True)).astype(str).str[:6].astype(int)
    assert len(np.unique(df_new['weapon1'])) == len(np.unique(df_filtered['Weapon1']))
    df_new['weapon2'] = pd.Series(pd.util.hash_array(df_filtered['Weapon2'].to_numpy(), categorize=True)).astype(str).str[:6].astype(int)
    assert len(np.unique(df_new['weapon2'])) == len(np.unique(df_filtered['Weapon2']))
    # common source (1) of different source (0)
    df_new['hypothesis'] = np.where(df_filtered['Source']=='common', 1, 0)
    # scores
    for score in ['ACCF', 'CMC', 'N', 'CCF_5_250']:
        if score in df_filtered.columns:
            score_short = re.sub('[^a-zA-Z]+', '', score.lower())
            df_new[score_short] = df_filtered[score].to_numpy()
    # splits, met wat kortere teksten
    for split in range(1,4):
        df_new['split' + str(split)] = df_filtered['Split' + str(split)].str[0].to_numpy()

    # nieuwe bestand wegschrijven
    df_new.to_csv(folder + trace + 'FF1st.csv', index = False)
    # (the files are later manually renamed to english)

print('finished')
