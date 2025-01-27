def data_concat():
    import numpy as np
    from glob import glob
    import pandas as pd

    KINEMATIC_PATH = 'processed file/**/kine/*csv'
    FORCE_PATH = 'processed file/**/force/*csv'

    KINEMATIC_DIR = [i.replace('\\','/') for i in glob(KINEMATIC_PATH)]
    FORCE_DIR = [i.replace('\\','/') for i in glob(FORCE_PATH)]

    kdf = pd.DataFrame()
    fdf = pd.DataFrame()

    for kine_dir, force_dir in zip(KINEMATIC_DIR, FORCE_DIR):
        kine = pd.read_csv(kine_dir)
        force = pd.read_csv(force_dir)
        
        _, kday, _, kfname = kine_dir.split('/')
        _, fday, _, ffname = force_dir.split('/')
        
        kfname = kfname.replace('.csv','')
        kplayer_name, ktrial, _, _, _, _, kball, _ = kfname.split('_')
        
        ffname = ffname.replace('.csv','')
        fplayer_name, ftrial, _, _, _, _, fball, _ = ffname.split('_')
        
        kine['player'] = kplayer_name
        kine['day'] = kday
        kine['trial'] = ktrial
        kine['ball_speed'] = kball
        
        force['player'] = fplayer_name
        force['day'] = fday
        force['trial'] = ftrial
        force['ball_speed'] = fball
        
        kdf = pd.concat([kdf, kine])
        fdf = pd.concat([fdf, force])
    return kdf, fdf