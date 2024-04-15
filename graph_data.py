import plotly.graph_objs as go
import numpy as np
import streamlit as st

username_passward = {
    "kookmin" : "640511",
    "PYS"     : "640511",
    "SBB"     : "640511",
}

def transform_list(nums):
    indexed_nums = list(enumerate(nums))
    indexed_nums.sort(key=lambda x: x[1])
    transformed = [0] * len(nums)
    current_rank = 1
    for i in range(len(nums)):
        if i > 0 and indexed_nums[i][1] != indexed_nums[i-1][1]:
            current_rank += 1
        transformed[indexed_nums[i][0]] = current_rank
    return transformed


def grf_plotly(data, cols, time, address, top, impact, axis):
    if axis == 'ap':
        title = 'GROUND REACTION FORCE (AP-AXIS)'
        ylb = 'Force [%BW]'
    elif axis == 'result':
        title = 'GROUND REACTION FORCE (RESULTANT)'
        ylb = 'Force [%BW]'
    elif axis == 'vt':
        title = 'GROUND REACTION FORCE (Vertical)'
        ylb = 'Force [%BW]'
    elif axis == 'momentum':
        title = 'MOMENTUM (AP-AXIS)'
        ylb = 'Momentum [N*s/BW]'

    y_values = {
        'max'       : {},
        'max_frame' : {},
        'min'       : {},
        'min_frame' : {},
        'address'   : {},
        'top'       : {},
        'impact'    : {},
    }
    
    
    # Create traces
    traces = []
    for col, info in cols.items():
        df = data[col]
        trace = go.Scatter(x=time, y=df, mode='lines', name=info[0], line=dict(color=info[-1],width=4))
        traces.append(trace)
        
        # Perform and store the calculations for max, min and specific times
        y_values['address'][col] = round(df.iloc[address], 2)
        y_values['top'][col]     = round(df.iloc[top], 2)
        y_values['impact'][col]  = round(df.iloc[impact], 2)
        if col == 'LEAD_FORCE_Y':
            y_values['max'][col] = round(df.min(), 2)
            y_values['max_time'][col] = np.where(df == df.min())[0][0]
        else:
            y_values['max'][col] = round(df.max(), 2)
            y_values['max_time'][col] = np.where(df == df.max())[0][0]

    event_times = [address, top, impact]
    event_names = ['address', 'top', 'impact']
    shapes = [
        {
            'type': 'line',
            'xref': 'x',
            'yref': 'paper',
            'x0': time[event_time],
            'y0': 0,
            'x1': time[event_time],
            'y1': 1,
            'opacity' : 0.5,
            'line': {
                'color': 'cyan',
                'width': 4,
                'dash': 'dash',
            }
        } for event_time in event_times
    ]
    annotations = [
        {
            'x': time[event_time + 12],
            'y': 1,
            'xref': 'x',
            'yref': 'paper',
            'text': label,
            'showarrow': False,
            'font': {
                'color': 'cyan',
                'size' : 16
            },
            'textangle': -90
        } for event_time, label in zip(event_times, event_names)
    ]

    # Update the layout with additional elements
    layout = go.Layout(
        title=title,
        xaxis=dict(title='Time [s]',
                    showgrid=False),
        yaxis=dict(
                    title=ylb,
                    showgrid=True,         # This will show the horizontal gridlines
                    gridcolor='lightgrey',
                    gridwidth=1,
                    zeroline=False
                ),
        showlegend=True,
        shapes = shapes,
        legend=dict(
                    # x=1, # Adjust this value to move the legend left or right
                    # y=1, # Adjust this value to move the legend up or down
                    # xanchor='right', # Anchor the legend's right side at the x position
                    # yanchor='top', # Anchor the legend's top at the y position
                    # bgcolor='rgb(43,48,61)', # Set a background color with a bit of transparency
                    orientation = 'h',
                    ),
        margin=dict(l=40, r=40, t=40, b=40),
        height=600,
        hovermode='closest',
        plot_bgcolor='rgb(43,48,61)',
        annotations=annotations
    )

    # Create the figure
    fig = go.Figure(data=traces, layout=layout)

    return fig, y_values

def one_angle_plotly(data, cols, time, address, top, impact):
    ang = {
        'max'       : {},
        'max_frame' : {},
        'min'       : {},
        'min_frame' : {},
        'address'   : {},
        'top'       : {},
        'impact'    : {},
    }
    
    figures = {}
    
    for col in cols:
        df = data[col]
        if 'VELOCITY' in col:
            y_label = 'Angular Velocity [deg/s]'
        elif 'ANGLE' in col:
            y_label = 'Angle [deg]'
        else:
            y_label = 'Distance [CM]'
        # Create the trace for the main data line
        trace = go.Scatter(x=time, y=df, mode='lines', name=cols[col], line=dict(color='firebrick', width=4))
        traces = [trace]
        
        ang['address'][col]  = round(df[address], 2)
        ang['top'][col]      = round(df[top], 2)
        ang['impact'][col]   = round(df[impact], 2)
        ang['max'][col]      = round(df.max(), 2)
        ang['max_time'][col] = np.where(df == df.max())[0][0]
        
        if col in ['TORSO_ANGLE_Y','LEAD_ELBOW_ANGLE_X','LEAD_SHOULDER_ANGLE_Y','LEAD_SHOULDER_ANGLE_Z','LEAD_KNEE_ANGULAR_VELOCITY_X']:
            ang['max'][col]  = round(df[address-40:impact+15].max(), 2)
            ang['max_time'][col] = np.where(df == df[address-40:impact+15].max())[0][0]

        elif col in ['LEAD_KNEE_ANGLE_X', 'HAND_ELBOW_HEIGHT']:
            ang['max'][col]  = round(df[address:impact+1].max(), 2)
            ang['max_time'][col] = np.where(df == df[address:impact+1].max())[0][0]
        
        elif col in [ 'TORSO_PELVIS_ANGLE_Z','LEAD_SHOULDER_ANGLE_X']:
            ang['max'][col]  = round(df.min(), 2)
            ang['max_time'][col] = np.where(df == df.min())[0][0]
        
        event_times = [address, top, impact]
        event_names = [address, top, impact]
        shapes = [
            {
                'type': 'line',
                'xref': 'x',
                'yref': 'paper',
                'x0': time[event_time],
                'y0': 0,
                'x1': time[event_time],
                'y1': 1,
                'opacity' : 0.5,
                'line': {
                    'color': 'cyan',
                    'width': 4,
                    'dash': 'dash',
                }
            } for event_time in event_times
        ]
        annotations = [
            {
                'x': time[event_time + 2],
                'y': 1,
                'xref': 'x',
                'yref': 'paper',
                'text': label,
                'showarrow': False,
                'font': {
                    'color': 'cyan',
                    'size' : 16
                },
                'textangle': -90
            } for event_time, label in zip(event_times, event_names)
        ]
        
        # Define the layout
        layout = go.Layout(
            title=f'{cols[col]}',
            xaxis=dict(title='Time [s]',
                       showgrid=False),
            yaxis=dict(title=y_label,
                       autorange = True,
                       rangemode='tozero',
                        showgrid=True,         # This will show the horizontal gridlines
                        gridcolor='lightgrey',
                        gridwidth=1,
                        zeroline=False,
                        ),                        
            showlegend=False,
            shapes =shapes,
            margin=dict(l=40, r=40, t=40, b=40),
            height=600,
            plot_bgcolor='rgb(43,48,61)',
            annotations=annotations
        )
        
        # Create the figure and add the traces to it
        fig = go.Figure(data=traces, layout=layout)
        
        # Store the figure in the dictionary
        figures[col] = fig
        
    return ang, figures

def kinematic_sequence_plotly(data, ks_cols, time, address, top, impact):
    ks = {
        'peak' : {},
        'time' : {},
    }
    
    # Create traces for each data series
    traces = []
    for col in ks_cols:
        trace = go.Scatter(
            x=time, 
            y=data[col], 
            mode='lines', 
            name=ks_cols[col][0],
            opacity=0.9,
            line=dict(color=ks_cols[col][-1],width= 3), 
        )
        traces.append(trace)
        ks['peak'][col] = round(data[col].max(), 2)
        ks['time'][col] = np.where(data[col] == data[col].max())[0][0]
    
    event_times = [address, top, impact]
    event_names = ['address', 'top', 'impact']
    shapes = [
        {
            'type': 'line',
            'xref': 'x',
            'yref': 'paper',
            'x0': time[event_time],
            'y0': 0,
            'x1': time[event_time],
            'y1': 1,
            'opacity' : 0.5,
            'line': {
                'color': 'cyan',
                'width': 3,
                'dash': 'dash',
            }
        } for event_time in event_times
    ]
    annotations = [
        {
            'x': time[event_time + 2],
            'y': 1,
            'xref': 'x',
            'yref': 'paper',
            'text': label,
            'showarrow': False,
            'font': {
                'color': 'cyan',
                'size' : 16
            },
            'textangle': -90
        } for event_time, label in zip(event_times, event_names)
    ]

    # Define the layout with annotations and shapes
    layout = go.Layout(
        title='KINEMATIC SEQUENCE',
        xaxis=dict(title='Time [s]',
                   showgrid=False),
        yaxis=dict(title='Angular Velocity [Deg/s]', 
                   autorange=True,           
                    rangemode='tozero',
                    showgrid=True,         # This will show the horizontal gridlines
                    gridcolor='lightgrey',
                    gridwidth=1,
                    zeroline=False,),
        annotations=annotations,
        shapes=shapes,
        showlegend=True,
        legend=dict(orientation='h'),
        margin=dict(l=40, r=40, t=40, b=40),
        plot_bgcolor='rgb(43,48,61)'
    )

    # Create the figure and add traces to it
    fig = go.Figure(data=traces, layout=layout)
    
    return ks, fig

def check_credentials(username, passward):
    if username in username_passward and passward == username_passward[username]:
        st.session_state['authenticated'] = True    
        st.session_state['selected_name'] = username
    else:
        st.session_state['authenticated'] = False
        st.error('ID나 PASSWORD가 잘못되었습니다')

# 인증되지 않았을 때 로그인 폼을 보여주는 함수
def show_login_form():
    with st.container():
        st.write("Login")
        username = st.text_input("ID", key='login_username')
        password = st.text_input("PASSWORD", type="password", key='login_password')
        login_button = st.button("login", on_click=check_credentials, args=(username, password))