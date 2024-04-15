import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
import numpy as np
from glob import glob
import data_concat
from graph_data import check_credentials,show_login_form,transform_list, grf_plotly, one_angle_plotly, kinematic_sequence_plotly

st.set_page_config(page_title = "KMU GOLF REPORT", 
                layout="wide"
                    )
@st.cache_data
def load_data():
    kdf, fdf = data_concat.data_concat()
    return kdf, fdf

if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

# 사용자가 인증되지 않았다면 로그인 폼을 보여줌
if not st.session_state['authenticated']:
    show_login_form()
    
else:
    kdf, fdf = load_data()  
    kdf['trial'] = kdf['trial'].astype(int)
    fdf['trial'] = fdf['trial'].astype(int)

    # 스트림릿 사이드바 설정
    if st.session_state['selected_name'] in ['kookmin']:
        unique_names = kdf['player'].unique()
    else:
        unique_names = [st.session_state['selected_name']]
        
    selected_name = st.sidebar.selectbox('Select Name', unique_names)
    filtered_df_by_name = kdf[kdf['player'] == selected_name]
    unique_dates = sorted(filtered_df_by_name['day'].unique())
    selected_date = st.sidebar.selectbox('Select Date', unique_dates)

    filtered_df_by_name_datas = kdf[(kdf['player'] == selected_name) &
                                    (kdf['day'] == selected_date)]
    unique_trial = sorted(filtered_df_by_name_datas['trial'].unique())
    selected_trial = st.sidebar.selectbox('Select Date', unique_trial)

    kine_filtered = kdf[(kdf['player'] == selected_name) & 
                        (kdf['day'] == selected_date) &
                        (kdf['trial'] == selected_trial)]

    force_filtered = fdf[(fdf['player'] == selected_name) & 
                        (fdf['day'] == selected_date) &
                        (fdf['trial'] == selected_trial)]

    kine_filtered.reset_index(inplace=True, drop=True)
    force_filtered.reset_index(inplace=True, drop=True)
    
    k_sr = 200
    k_address  = kine_filtered['address'][0]
    k_address1 = kine_filtered['address'][0] - k_address
    k_top      = kine_filtered['top'][0]     - k_address
    k_impact   = kine_filtered['impact'][0]  - k_address
    stride_length = round(float(kine_filtered['stride_length'][0]))
    ball_speed = round(float(kine_filtered['ball_speed'][0]) * 1.6)
    k_total_time = k_impact+1 - k_address

    f_sr = k_sr * 5
    f_address    = force_filtered['address'][0]
    f_address1   = force_filtered['address'][0] - k_address
    f_top        = force_filtered['top'][0]     - k_address
    f_impact     = force_filtered['impact'][0]  - k_address
    f_total_time = f_impact+1 - f_address
    
    k_df = kine_filtered.iloc[k_address:int(k_impact + k_address + (k_sr * 0.2)),:].reset_index(drop=True)
    f_df = force_filtered.iloc[f_address:int(f_impact + f_address + (f_sr * 0.2)),:].reset_index(drop=True)

    f_lead_peak_z_time = f_df['lead_peak_z'][0] - f_address
    f_rear_peak_z_time = np.where(f_df['REAR_FORCE_Z'] == f_df['REAR_FORCE_Z'].max())[0][0]
    force_peak_time = round((f_lead_peak_z_time - f_rear_peak_z_time) / 1080 , 4)

    f_rear_peak_y_time = np.where(f_df['REAR_FORCE_Y'] == f_df['REAR_FORCE_Y'].max())[0][0]
    f_lead_min_y_time  = f_df['lead_valley_y'][0] - f_address

    k_df.drop(['address','top','impact','br_time'], axis=1, inplace=True)
    f_df.drop(['address','top','impact','br_time'], axis=1, inplace=True)

    k_time = k_df['TIME']
    f_time = f_df['TIME']

    # ===================================================================================
    # ============================= Using Data ==========================================
    ml_cols = {
    'REAR_FORCE_X' : ['Trail Leg' , 'b'],
    'LEAD_FORCE_X' : ['Stride Leg', 'r'],
    }

    ap_cols = {
        'REAR_FORCE_Y' : ['Trail Leg' , 'b'],
        'LEAD_FORCE_Y' : ['Stride Leg', 'r'],
    }

    vt_cols = {
        'REAR_FORCE_Z' : ['Trail Leg' , 'b'],
        'LEAD_FORCE_Z' : ['Stride Leg', 'r']
    }

    momentum_cols = {
        'REAR_MOMENTUM_Y' : ['Trail Leg' , 'blue'],
        'LEAD_MOMENTUM_Y' : ['Stride Leg', 'red']
    }
    
    ks_cols = {
        'PELVIS_ANGLUAR_VELOCITY_Z'        : ['PELVIS'   , 'r'],
        'TORSO_ANGLUAR_VELOCITY_Z'         : ['TORSO'    , 'g'],
        'LEAD_ELBOW_ANGULAR_VELOCITY_X'    : ['ELBOW'    , 'b'],
        'LEAD_SHOULDER_ANGULAR_VELOCITY_Z' : ['SHOULDER' , 'y'],
    }

    pel_cols = {
                #'pelvis_3d_x' : ['ML', (round(129/255,3), round(185/255,3), round(98/255,3))],
                #'pelvis_3d_y' : ['Vertical' , 'blue'],
                #'pelvis_3d_z' : ['AP', 'red']
                }

    tor_cols = {
                #'proximal_neck_3d_x' : ['ML', (round(129/255,3), round(185/255,3), round(98/255,3))],
                #'proximal_neck_3d_y' : ['Vertical' , 'blue'],
                #'proximal_neck_3d_z' : ['AP', 'red']
                }


    ang_cols = {
        'TORSO_PELVIS_ANGLE_Z'  : 'Hip-Shoulder Separation',
        'LEAD_ELBOW_ANGLE_X'    : 'Front Elbow Flexion',
        'REAR_ELBOW_ANGLE_X'    : 'Back Elbow Flexion',
        'LEAD_SHOULDER_ANGLE_X' : 'Front Arm Horizontal Adduction',
        'REAR_SHOULDER_ANGLE_X' : 'Back Arm Horizontal Adduction',
        'TORSO_ANGLE_Y'         : 'Trunk Lateral Tilt',
        'LEAD_KNEE_ANGLE_X'     : 'Front Leg Knee Extension',  # FLEXION
        'REAR_KNEE_ANGLE_X'     : 'Back Leg Knee Extension',
        'LEAD_SHOULDER_ANGLE_Y' : 'Front Arm Adduction',
        'REAR_SHOULDER_ANGLE_Y' : 'Back Arm Adduction',
    }

    vel_cols = {
        'LEAD_KNEE_ANGULAR_VELOCITY_X' : 'Front Leg Knee Extension Velocity',
        'REAR_KNEE_ANGULAR_VELOCITY_X' : 'Back Leg Knee Extension Velocity',
        #'trunk_twist_clockwise_velocity' : 'X-Factor Velocity', 
        #'right_elbow_flexion_velocity' : 'Back Elbow Flexion Velocity',
        #'left_elbow_flexion_velocity' : 'Front Elbow Flexion Velocity',    
    }

    trunk_cols = {
        'TORSO_ANGLE_Y' : ['Lateral Tilt' , (round(129/255,3), round(185/255,3), round(98/255,3))],
        'TORSO_ANGLE_Z' : ['Rotation', 'red']
    }

    pelvis_cols = {
        'PELVIS_ANGLE_Y' : ['Lateral Tilt', (round(129/255,3), round(185/255,3), round(98/255,3))],
        'PELVIS_ANGLE_Z' : ['Rotation', 'red']
    }
    
    # ============================ 그래프 및 시점 수치 =======================================
    force_ml_fig, force_ml_values = grf_plotly(f_df, ml_cols, f_time, f_address1, f_top, f_impact, axis='ml')
    force_ap_fig, force_ap_values = grf_plotly(f_df, ap_cols, f_time, f_address1, f_top, f_impact, axis='ap')
    force_vt_fig, force_vt_values = grf_plotly(f_df, vt_cols, f_time, f_address1, f_top, f_impact, axis='vt')
    #force_result_fig, force_result_values = grf_plotly(f_df, result_cols, f_time, f_kh_time1, f_fc_time, f_mer_time, f_br_time, axis='result')
    #force_momentum_fig, force_momentum_values = grf_plotly(f_df, momentum_cols, f_time, f_time, f_address1, f_top, f_impact,, axis='momentum')
    kine_values, kine_fig = one_angle_plotly(k_df, ang_cols, k_time, k_address1, k_top, k_impact)
    kinematic_values, kinematic_fig = kinematic_sequence_plotly(k_df, ks_cols, k_time, k_address1, k_top, k_impact)
    
    force_ap_fig.update_layout(
        width=800,  # Set the width to your preference
        height=400  # Set the height to your preference
    )
    force_vt_fig.update_layout(
        width=800,  # Set the width to your preference
        height=400  # Set the height to your preference
    )
    '''
    force_result_fig.update_layout(
        width=800,  # Set the width to your preference
        height=400  # Set the height to your preference
    )
    force_momentum_fig.update_layout(
        width=800,  # Set the width to your preference
        height=400  # Set the height to your preference
    )
    '''
    for col in kine_fig:
        fig = kine_fig[col]
        fig.update_layout(
        width=800,  # Set the width to your preference
        height=400  # Set the height to your preference
        )
    kinematic_fig.update_layout(
        width=800,
        height=400
    )

    peak_pel = round(kinematic_values['peak']['PELVIS_ANGLUAR_VELOCITY_Z']); time_pel = kinematic_values['time']['PELVIS_ANGLUAR_VELOCITY_Z']
    peak_tor = round(kinematic_values['peak']['TORSO_ANGLUAR_VELOCITY_Z']);time_tor = kinematic_values['time']['TORSO_ANGLUAR_VELOCITY_Z']
    peak_elb = round(kinematic_values['peak']['LEAD_ELBOW_ANGULAR_VELOCITY_X']);time_elb = kinematic_values['time']['LEAD_ELBOW_ANGULAR_VELOCITY_X']
    peak_sho = round(kinematic_values['peak']['LEAD_SHOULDER_ANGULAR_VELOCITY_Z']);time_sho = kinematic_values['time']['LEAD_SHOULDER_ANGULAR_VELOCITY_Z']

    pel_time = round(100 * (time_pel - f_address1) / k_total_time)
    tor_time = round(100 * (time_tor - f_address1) / k_total_time)
    elb_time = round(100 * (time_elb - f_address1) / k_total_time)
    sho_time = round(100 * (time_sho - f_address1) / k_total_time)

    tor_gain   = round(peak_tor / peak_pel,2)
    upper_gain = round(peak_elb / peak_tor,2)
    fore_gain  = round(peak_sho / peak_elb,2)
    
    sq_time = [pel_time, tor_time, elb_time, sho_time]
    expected_order = transform_list(sq_time)

    data_as_dict = {
        "Segment"    : ["Pelvic [°/s]", "Torso [°/s]", "Elbow [°/s]", "Shoulder [°/s]"],
        "Pro"        : ["649 ~ 840", "987 ~ 1174", "2211 ~ 2710", "4331 ~ 4884"],
        "Peak"       : [peak_pel, peak_tor, peak_elb, peak_sho],
        "Timing"     : [f"{pel_time} %", f"{tor_time} %", f"{elb_time} %", f"{sho_time} %"],
        "Sequence"   : expected_order,
        "Speed Gain" : [0, tor_gain,upper_gain, fore_gain]
    }
    kinematic_sq = pd.DataFrame(data_as_dict)
    kinematic_sq = kinematic_sq.set_index('Segment')
    kinematic_sq['Speed Gain'] = kinematic_sq['Speed Gain'].astype(float).map('{:.2f}'.format)
    kinematic_sq = kinematic_sq.style.set_properties(**{'text-align': 'center'})
    # ===================================================================================
    # ============================= DashBoard ===========================================
    page_tab1, page_tab2 = st.tabs(["데이터 보기", "피드백 남기기"])