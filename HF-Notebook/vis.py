import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import plotly.express as px

def visualize_readiness_scores(data):
    # Scatter chart for readiness scores showing only points
    fig = px.scatter(data, x='day', y='score', title='Readiness Score Over Time')

    fig.update_layout(
        xaxis_title='Day',
        yaxis_title='Readiness Score',
        legend_title='Metric'
    )

    return fig

def visualize_activity_log(data, start_date=None, end_date=None):
    # Count the occurrences of each activity per day
    activity_count = data.groupby(['day', 'activity']).size().unstack(fill_value=0).reset_index()

    # Melt the dataframe for easier plotting with Plotly
    activity_count_melted = activity_count.melt(id_vars=['day'], var_name='activity', value_name='count')

    # Create a bar chart for activity categories
    fig = px.bar(activity_count_melted, x='day', y='count', color='activity', title='Activity Log')

    fig.update_layout(
        xaxis_title='Day',
        yaxis_title='Count',
        legend_title='Activity',
        showlegend=True
    )

    # Set the x-axis range if start_date and end_date are provided
    if start_date and end_date:
        fig.update_xaxes(range=[pd.to_datetime(start_date).date(), pd.to_datetime(end_date).date()])
    
    return fig

def visualize_sleep_scores(data):
    # Scatter chart for sleep scores (points only)
    fig = px.scatter(data, x='day', y='score', title='Sleep Score Over Time')

    fig.update_layout(
        xaxis_title='Day',
        yaxis_title='Sleep Score',
        legend_title='Metric'
    )

    return fig

def visualize_respiratory(data, start_date=None, end_date=None):
    # Group and average the data by day
    df = data[["day", "average_breath"]].groupby('day').mean().reset_index()
    fig = go.Figure()

    # Add a line plot for respiratory rates
    fig.add_trace(go.Scatter(
        x=df["day"],
        y=df["average_breath"],
        name='Respiratory Rate',
        mode='markers',
        # line=dict(color='lightblue')
    ))

    fig.update_layout(
        title='Average Respiratory Rate During Sleep',
        xaxis_title='Day',
        yaxis_title='Count/minute',
        legend_title='Metric',
        showlegend=True
    )

    # Set the x-axis range if start_date and end_date are provided
    if start_date and end_date:
        fig.update_xaxes(range=[pd.to_datetime(start_date).date(), pd.to_datetime(end_date).date()])

    return fig

# def visualize_daily_heart_rate(df):
#     # Convert the timestamp column to datetime
#     df['timestamp'] = pd.to_datetime(df['timestamp'])
#     df['date'] = df['timestamp'].dt.date

#     # Calculate the average bpm for each source for each date
#     df_daily_avg = df.groupby(['date', 'source'], as_index=False).agg({'bpm': 'mean'})

#     fig = px.line(df_daily_avg, x='date', y='bpm', color='source', title='Daily Average Heart Rate Chart', markers=True)
    
#     return fig


def visualize_activity_time(data, start_date=None, end_date=None):
    """
    Visualize activity time data as stacked bar charts.

    Parameters:
    data (pd.DataFrame): DataFrame containing activity time information with columns:
                         'day', 'high_activity_time', 'medium_activity_time',
                         'low_activity_time', 'resting_time', 'sedentary_time'
    """
    # Create a copy of the data to convert time units without modifying the original DataFrame
    data_copy = data.copy()
    
    # Convert time from seconds to hours
    data_copy['high_activity_time'] = data_copy['high_activity_time'] / 60 /60
    data_copy['medium_activity_time'] = data_copy['medium_activity_time'] / 60/60
    data_copy['low_activity_time'] = data_copy['low_activity_time'] / 60/60
    data_copy['resting_time'] = data_copy['resting_time'] / 60/60
    data_copy['sedentary_time'] = data_copy['sedentary_time'] / 60/60

    # Create a stacked bar chart
    fig = go.Figure()

    # Add traces for each activity time
    fig.add_trace(go.Bar(
        x=data_copy['day'],
        y=data_copy['high_activity_time'],
        name='High Activity Time',
        marker_color='rgb(26, 118, 255)'
    ))
    fig.add_trace(go.Bar(
        x=data_copy['day'],
        y=data_copy['medium_activity_time'],
        name='Medium Activity Time',
        marker_color='rgb(55, 83, 109)'
    ))
    fig.add_trace(go.Bar(
        x=data_copy['day'],
        y=data_copy['low_activity_time'],
        name='Low Activity Time',
        marker_color='rgb(50, 171, 96)'
    ))
    fig.add_trace(go.Bar(
        x=data_copy['day'],
        y=data_copy['resting_time'],
        name='Resting Time',
        marker_color='rgb(235, 204, 255)'
    ))
    fig.add_trace(go.Bar(
        x=data_copy['day'],
        y=data_copy['sedentary_time'],
        name='Sedentary Time',
        marker_color='rgb(128, 0, 128)'
    ))

    # Update layout for better visualization
    fig.update_layout(
        barmode='stack',
        title='Activity Time Breakdown by Day (in hours)',
        xaxis=dict(title='Day'),
        yaxis=dict(title='Time (hours)'),
        legend_title='Activity Type & Time',
        template='plotly_white'
    )
    if start_date and end_date:
        fig.update_xaxes(range=[pd.to_datetime(start_date).date(), pd.to_datetime(end_date).date()])
    return fig

def visualize_daily_heart_rate(data, start_date=None, end_date = None):
    """
    Visualize heart rate information as scatter plots.
    
    Parameters:
    data (pd.DataFrame): DataFrame containing heart rate information with columns:
                        'bpm', 'source', 'timestamp'
    """
    # Convert timestamp to datetime
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Create scatter plot
    fig = px.scatter(data, x='timestamp', y='bpm', color='source',
                     title='Heart Rate Information',
                     labels={'timestamp': 'Timestamp', 'bpm': 'Beats Per Minute (BPM)'},
                     )

    # Update layout for better visualization
    fig.update_layout(
        xaxis_title='Timestamp',
        yaxis_title='Beats Per Minute (BPM)',
        legend_title='Source',
        # template='plotly_white'
    )
    # Set the x-axis range if start_date and end_date are provided
    if start_date and end_date:
        fig.update_xaxes(range=[pd.to_datetime(start_date).date(), pd.to_datetime(end_date).date()])
    return fig

def visualize_spo2(data, start_date=None, end_date=None):
    """
    Visualize daily SpO2 (oxygen saturation) data.

    Parameters:
    data (pd.DataFrame): DataFrame containing SpO2 data with columns:
                         'day' and 'spo2_percentage.average' (or 'average_spo2')
    start_date (str, optional): Start date for x-axis range
    end_date (str, optional): End date for x-axis range
    """
    # Handle different column naming conventions
    spo2_col = None
    if 'spo2_percentage.average' in data.columns:
        spo2_col = 'spo2_percentage.average'
    elif 'average_spo2' in data.columns:
        spo2_col = 'average_spo2'
    elif 'spo2_percentage' in data.columns:
        spo2_col = 'spo2_percentage'
    else:
        raise ValueError("No SpO2 column found in data. Expected 'spo2_percentage.average', 'average_spo2', or 'spo2_percentage'")

    # Group and average the data by day
    df = data[["day", spo2_col]].groupby('day').mean().reset_index()
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["day"],
        y=df[spo2_col],
        name='SpO2',
        mode='markers',
    ))

    fig.update_layout(
        title='Daily Average SpO2 (Oxygen Saturation)',
        xaxis_title='Day',
        yaxis_title='SpO2 (%)',
        legend_title='Metric',
        showlegend=True
    )

    # Set the x-axis range if start_date and end_date are provided
    if start_date and end_date:
        fig.update_xaxes(range=[pd.to_datetime(start_date).date(), pd.to_datetime(end_date).date()])

    return fig


def visualize_sleep_stages_stacked(data, start_date=None, end_date=None):
    """
    Visualize sleep stages (Deep, REM, Light) as a stacked bar chart.

    Parameters:
    data (pd.DataFrame): DataFrame containing sleep data with columns:
                         'day', 'deep_sleep_duration', 'rem_sleep_duration', 'light_sleep_duration'
                         (durations in seconds)
    start_date (str, optional): Start date for x-axis range
    end_date (str, optional): End date for x-axis range
    """
    # Check for required columns
    required_cols = ['day', 'deep_sleep_duration', 'rem_sleep_duration', 'light_sleep_duration']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Group by day and sum durations (in case of multiple sleep periods per day)
    df = data[required_cols].groupby('day').sum().reset_index()

    # Convert from seconds to hours
    df['deep_hours'] = df['deep_sleep_duration'] / 3600
    df['rem_hours'] = df['rem_sleep_duration'] / 3600
    df['light_hours'] = df['light_sleep_duration'] / 3600

    fig = go.Figure()

    # Add stacked bars - Deep Sleep at bottom, then REM, then Light on top
    fig.add_trace(go.Bar(
        x=df['day'],
        y=df['deep_hours'],
        name='Deep Sleep',
        marker_color='#1E3A5F',  # Dark blue
        hovertemplate='Deep Sleep: %{y:.2f} hours<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        x=df['day'],
        y=df['rem_hours'],
        name='REM Sleep',
        marker_color='#FF6B6B',  # Coral/orange-red
        hovertemplate='REM Sleep: %{y:.2f} hours<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        x=df['day'],
        y=df['light_hours'],
        name='Light Sleep',
        marker_color='#4ECDC4',  # Teal/light blue
        hovertemplate='Light Sleep: %{y:.2f} hours<extra></extra>'
    ))

    fig.update_layout(
        barmode='stack',
        title='Daily Sleep Stages (Deep / REM / Light)',
        xaxis_title='Day',
        yaxis_title='Sleep Duration (hours)',
        legend_title='Sleep Stage',
        showlegend=True,
        hovermode='x unified'
    )

    # Set the x-axis range if start_date and end_date are provided
    if start_date and end_date:
        fig.update_xaxes(range=[pd.to_datetime(start_date).date(), pd.to_datetime(end_date).date()])

    return fig

def visualize_temperature(data, start_date=None, end_date=None):
    # Group and average the data by day
    df = data[["day", "temperature_deviation"]].groupby('day').mean().reset_index()
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["day"],
        y=df["temperature_deviation"],
        name='Temperature Deviation',
        mode='markers',
        # line=dict(color='lightblue')
    ))

    fig.update_layout(
        title='Average temperature deviation',
        xaxis_title='Day',
        yaxis_title='Degrees Celsius',
        legend_title='Metric',
        showlegend=True
    )

    # Set the x-axis range if start_date and end_date are provided
    if start_date and end_date:
        fig.update_xaxes(range=[pd.to_datetime(start_date).date(), pd.to_datetime(end_date).date()])

    return fig

def visualize_daily_hrv(df,start_date=None, end_date=None):
    # Group and average the data by day
    data = df[["day", "average_hrv"]].groupby('day').mean().reset_index()
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data["day"],
        y=data["average_hrv"],
        name='Average HRV During Sleep',
        mode='markers',
        # line=dict(color='green')
    ))

    fig.update_layout(
        title='Daily Average HRV During Sleep',
        xaxis_title='Day',
        yaxis_title='HRV',
        barmode='group',
        legend_title='Metric'
    )

    # Set the x-axis range if start_date and end_date are provided
    if start_date and end_date:
        fig.update_xaxes(range=[pd.to_datetime(start_date).date(), pd.to_datetime(end_date).date()])

    return fig

def visualize_daily_sleep_time(df, start_date=None, end_date=None):
    # Group and average the data by day
    data = df[["day", "total_sleep_duration"]].groupby('day').sum().reset_index()
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data["day"],
        y=data["total_sleep_duration"]/3600,
        name='Daily Total Sleep Duration',
        mode='markers',
        # line=dict(color='green')
    ))

    fig.update_layout(
        title='Daily Total Sleep Duration',
        xaxis_title='Day',
        yaxis_title='Sleep Duration (Unit: hours)',
        barmode='group',
        legend_title='Metric',
        showlegend=True,
    )

    # Set the x-axis range if start_date and end_date are provided
    if start_date and end_date:
        fig.update_xaxes(range=[pd.to_datetime(start_date).date(), pd.to_datetime(end_date).date()])

    return fig


def visualize_daily_lowest_hr(df, start_date=None, end_date=None):
    # Group and average the data by day
    data = df[["day", "lowest_heart_rate"]].groupby('day').min().reset_index()
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data["day"],
        y=data["lowest_heart_rate"],
        name='Daily Lowest Heart Rate',
        mode='markers',
        # line=dict(color='green')
    ))

    fig.update_layout(
        title='Daily Lowest Heart Rate During Sleep',
        xaxis_title='Day',
        yaxis_title='Beat Per Minute',
        # barmode='group',
        # legend_title='Metric'
    )

    # Set the x-axis range if start_date and end_date are provided
    if start_date and end_date:
        fig.update_xaxes(range=[pd.to_datetime(start_date).date(), pd.to_datetime(end_date).date()])

    return fig

def visualize_daily_steps(df,start_date=None, end_date=None):
    # Group and average the data by day
    data = df[["day", "steps"]].groupby('day').sum().reset_index()
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data["day"],
        y=data["steps"],
        name='Daily Total Steps Count',
        mode='markers',
        # line=dict(color='green')
    ))

    fig.update_layout(
        title='Daily Total Steps',
        xaxis_title='Day',
        yaxis_title='Steps',
        barmode='group',
        legend_title='Metric',
        showlegend=True
    )

    # Set the x-axis range if start_date and end_date are provided
    if start_date and end_date:
        fig.update_xaxes(range=[pd.to_datetime(start_date).date(), pd.to_datetime(end_date).date()])

    return fig

def visualize_heart_rate(df):
    # Group and average the data by day
    data = df[["day", "average_heart_rate", "lowest_heart_rate", "average_hrv"]].groupby('day').mean().reset_index()
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=data["day"],
        y=data["average_heart_rate"],
        name='Average Heart Rate',
        marker_color='blue'
    ))

    fig.add_trace(go.Bar(
        x=data["day"],
        y=data["lowest_heart_rate"],
        name='Lowest Heart Rate',
        marker_color='lightblue'
    ))

    fig.add_trace(go.Scatter(
        x=data["day"],
        y=data["average_hrv"],
        name='Average HRV',
        mode='markers'
    ))

    fig.update_layout(
        title='Heart Rate Data Analysis',
        xaxis_title='Day',
        yaxis_title='Heart Rate (BPM)',
        barmode='group',
        legend_title='Metric'
    )

    return fig

def convert_to_datetime(time_str):
    return datetime.fromisoformat(time_str)

def visualize_sleep_start_end(df):
    df_start = pd.DataFrame({
        'Day': df["day"],
        'Time': df["bedtime_start"].apply(convert_to_datetime),
        'Type': 'Start'
    })

    df_end = pd.DataFrame({
        'Day': df["day"],
        'Time': df["bedtime_end"].apply(convert_to_datetime),
        'Type': 'End'
    })

    new_df = pd.concat([df_start, df_end])
    new_df['Time'] = new_df['Time'].apply(lambda t: datetime(2000, 1, 1, t.hour, t.minute, t.second))

    fig = px.scatter(new_df, x='Time', y='Day', color='Type', title='Bedtime Start and End Times')

    fig.update_xaxes(
        tickmode='array',
        tickvals=[datetime(2000, 1, 1, hour) for hour in range(24)],
        ticktext=[f'{hour:02d}:00' for hour in range(24)]
    )

    return fig

def visualize_sleep_breakdowns(data):
    # Convert durations from seconds to hours
    data["deep_sleep_duration"] = data["deep_sleep_duration"] / 3600
    data["light_sleep_duration"] = data["light_sleep_duration"] / 3600
    data["rem_sleep_duration"] = data["rem_sleep_duration"] / 3600
    data["awake_time"] = data["awake_time"] / 3600

    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=data["day"],
        y=data["deep_sleep_duration"],
        name='Deep Sleep',
        marker_color='indianred'
    ))
    fig.add_trace(go.Bar(
        x=data["day"],
        y=data["light_sleep_duration"],
        name='Light Sleep',
        marker_color='lightblue'
    ))
    fig.add_trace(go.Bar(
        x=data["day"],
        y=data["rem_sleep_duration"],
        name='REM Sleep',
        marker_color='orange'
    ))
    fig.add_trace(go.Bar(
        x=data["day"],
        y=data["awake_time"],
        name='Awake Time',
        marker_color='lightgray'
    ))

    # Add secondary y-axis for sleep efficiency
    fig.update_layout(
        yaxis=dict(title='Hours'),
        yaxis2=dict(title='Sleep Efficiency', overlaying='y', side='right', range=[0, 100])
    )

    # Add scatter trace for sleep efficiency (points only)
    fig.add_trace(go.Scatter(
        x=data["day"],
        y=data["efficiency"],
        name='Sleep Efficiency',
        yaxis='y2',
        mode='markers',
        marker=dict(color='green')
    ))

    fig.update_layout(
        barmode='stack',
        title='Sleep Duration and Efficiency Breakdown',
        xaxis_title='Day',
        legend_title='Sleep Types and Efficiency'
    )

    return fig


# flowsheet visualization
def visualize_temporal_flowsheet(df):
    # Ensure the columns are of the correct type
    df["RECORDED_DAY"] = pd.to_datetime(df["RECORDED_DAY"]).dt.date
    df["MEAS_VALUE"] = pd.to_numeric(df["MEAS_VALUE"], errors='coerce')
    
    # Group by "RECORDED_DAY" and "fdgDisplayName", and aggregate "MEAS_VALUE"
    df_grouped = df.groupby(["RECORDED_DAY", "fdgDisplayName"])["MEAS_VALUE"].mean().reset_index()
    
    # Create the scatter chart (points only)
    fig = px.scatter(df_grouped, x="RECORDED_DAY", y="MEAS_VALUE", color="fdgDisplayName",
                     title="Temporal Categorical Value Changes",
                     labels={"RECORDED_DAY": "Recorded Day", "MEAS_VALUE": "Mean Measure Value", "fdgDisplayName": "Category"})
    
    # Update layout for better visualization
    fig.update_layout(xaxis_title='Day',
                      yaxis_title='Mean Measure Value',
                      legend_title_text='Category')
    
    fig.show()

def visualize_individual_temporal_flowsheet(df, start_date=None, end_date=None):
    import pandas as pd
    import plotly.express as px
    
    # Ensure the columns are of the correct type
    df["RECORDED_DAY"] = pd.to_datetime(df["RECORDED_DAY"]).dt.date
    df["MEAS_VALUE"] = pd.to_numeric(df["MEAS_VALUE"], errors='coerce')
    
    # Group by "RECORDED_DAY" and "fdgDisplayName", and aggregate "MEAS_VALUE"
    df_grouped = df.groupby(["RECORDED_DAY", "fdgDisplayName"])["MEAS_VALUE"].mean().reset_index()
    
    # Get the unique categories
    categories = df_grouped["fdgDisplayName"].unique()

    # Generate a scatter chart for each category
    for category in categories:
        df_category = df_grouped[df_grouped["fdgDisplayName"] == category]
        
        fig = px.scatter(df_category, x="RECORDED_DAY", y="MEAS_VALUE",
                         title=f"Temporal Changes for {category}",
                         labels={"RECORDED_DAY": "Recorded Day", "MEAS_VALUE": "Mean Measure Value"})
        
        # Update layout for better visualization
        fig.update_layout(xaxis_title='Day',
                          yaxis_title='Mean Measure Value',
                          legend_title_text='Category')
        
        # Set the x-axis range if start_date and end_date are provided
        if start_date and end_date:
            fig.update_xaxes(range=[pd.to_datetime(start_date).date(), pd.to_datetime(end_date).date()])
        
        fig.show()


def visualize_individual_temporal_flowsheetv2(df, start_date=None, end_date=None):
    # Ensure the columns are of the correct type
    df["RECORDED_DAY"] = pd.to_datetime(df["RECORDED_DAY"]).dt.date
    df["RECORDED_TIME"] = pd.to_datetime(df["RECORDED_TIME"])
    df["MEAS_VALUE"] = pd.to_numeric(df["MEAS_VALUE"], errors='coerce')
    value_threshold = 2

    # Get the unique categories
    categories = df["fdgDisplayName"].unique()
    valid_categories = df['fdgDisplayName'].value_counts()[lambda x: x > value_threshold].index.tolist()

    # Generate a scatter plot and mean line for each category
    for category in valid_categories:
        df_category = df[df["fdgDisplayName"] == category]
        
        # Calculate the mean value per day for the mean line
        df_mean = df_category.groupby("RECORDED_DAY")["MEAS_VALUE"].mean().reset_index()
        
        # Create the scatter plot
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_category["RECORDED_TIME"],
            y=df_category["MEAS_VALUE"],
            mode='markers',
            name='Individual Measurements',
            marker=dict(color='blue')
        ))
        
        # # Add the mean value line
        # fig.add_trace(go.Scatter(
        #     x=df_mean["RECORDED_DAY"],
        #     y=df_mean["MEAS_VALUE"],
        #     mode='lines',
        #     name='Mean Value',
        #     line=dict(color='red')
        # ))

        # Update the layout
        fig.update_layout(
            title=f"Temporal Changes for {category}",
            xaxis_title='Timestamp',
            yaxis_title='Measure Value',
            legend_title='Legend',
            showlegend=True
        )

        # Set the x-axis range if start_date and end_date are provided
        if start_date and end_date:
            fig.update_xaxes(range=[pd.to_datetime(start_date), pd.to_datetime(end_date)])

        fig.show()

def visualize_individual_temporal_flowsheetv3(df, start_date=None, end_date=None):
    # Ensure the columns are of the correct type
    df["RECORDED_DAY"] = pd.to_datetime(df["RECORDED_DAY"]).dt.date
    df["RECORDED_TIME"] = pd.to_datetime(df["RECORDED_TIME"])
    df["MEAS_VALUE"] = pd.to_numeric(df["MEAS_VALUE"], errors='coerce')

    # Filter the dataframe to include only SpO2, Resp, and MAP (mmHg)
    valid_categories = ['SpO2', 'Resp', 'MAP (mmHg)']
    df_filtered = df[df["fdgDisplayName"].isin(valid_categories)]

    # Generate a scatter plot for each category
    for category in valid_categories:
        df_category = df_filtered[df_filtered["fdgDisplayName"] == category]
        
        # Skip this category if there's no data
        if df_category.empty:
            print(f"No data available for {category}")
            continue

        # Create the scatter plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_category["RECORDED_TIME"],
            y=df_category["MEAS_VALUE"],
            mode='markers',
            name='Individual Measurements',
            marker=dict(color='blue')
        ))

        # Update the layout
        fig.update_layout(
            title=f"Temporal Changes for {category}",
            xaxis_title='Timestamp',
            yaxis_title='Measure Value',
            showlegend=True
        )

        # Set the x-axis range if start_date and end_date are provided
        if start_date and end_date:
            fig.update_xaxes(range=[pd.to_datetime(start_date), pd.to_datetime(end_date)])

        fig.show()

import pandas as pd
import plotly.graph_objects as go

def visualize_daily_lowest_hr_v2(df, start_date=None, end_date=None):
    # Ensure the columns are of the correct type
    df["day"] = pd.to_datetime(df["day"]).dt.date

    # Extract heart rate values and create a new DataFrame
    heart_rate_data = []
    for index, row in df.iterrows():
        heart_rate_values = row["heart_rate"]["items"]
        interval = row["heart_rate"]["interval"]
        timestamp = pd.to_datetime(row["heart_rate"]["timestamp"])
        day = row["day"]
        for i, hr in enumerate(heart_rate_values):
            if hr is not None:
                hr_time = timestamp + pd.to_timedelta(i * interval, unit='s')
                heart_rate_data.append({"timestamp": hr_time, "heart_rate": hr, "day": day})

    heart_rate_df = pd.DataFrame(heart_rate_data)
    
    # Calculate the daily lowest heart rate
    daily_lowest_hr = heart_rate_df.groupby("day")["heart_rate"].min().reset_index()

    fig = go.Figure()

    # Add scatter plot for individual heart rate values
    fig.add_trace(go.Scatter(
        x=heart_rate_df["timestamp"],
        y=heart_rate_df["heart_rate"],
        mode='markers',
        name='Heart Rate',
        marker=dict(color='blue')
    ))

    # # Add line plot for daily lowest heart rate
    # fig.add_trace(go.Scatter(
    #     x=daily_lowest_hr["day"],
    #     y=daily_lowest_hr["heart_rate"],
    #     mode='lines+markers',
    #     name='Daily Lowest Heart Rate',
    #     line=dict(color='red')
    # ))

    fig.update_layout(
        title='Daily Heart Rate During Sleep',
        xaxis_title='Timestamp',
        yaxis_title='Beat Per Minute',
        legend_title='Metric',
        showlegend=True
    )

    # Set the x-axis range if start_date and end_date are provided
    if start_date and end_date:
        fig.update_xaxes(range=[pd.to_datetime(start_date), pd.to_datetime(end_date)])

    return fig

def visualize_sedentary_time(data, start_date=None, end_date=None):
    """
    Visualize sedentary time data as a scatter plot.
    
    Parameters:
    data (pd.DataFrame): DataFrame containing activity time information with columns:
                         'day' and 'sedentary_time'
    start_date (str, optional): Start date for x-axis range
    end_date (str, optional): End date for x-axis range
    """
    # Create a copy of the data to convert time units without modifying the original DataFrame
    data_copy = data.copy()
    
    # Convert sedentary time from seconds to hours
    data_copy['sedentary_time'] = data_copy['sedentary_time'] / 3600  # 3600 seconds in an hour

    # Create a scatter chart
    fig = go.Figure()

    # Add trace for sedentary time
    fig.add_trace(go.Scatter(
        x=data_copy['day'],
        y=data_copy['sedentary_time'],
        mode='markers',
        name='Sedentary Time',
        # line=dict(color='rgb(128, 0, 128)', width=2),
        marker=dict(size=6)
    ))

    # Update layout for better visualization
    fig.update_layout(
        title='Sedentary Time by Day',
        xaxis=dict(title='Day'),
        yaxis=dict(title='Time (hours)'),
        # template='plotly_white'
    )

    # Update x-axis range if start_date and end_date are provided
    if start_date and end_date:
        fig.update_xaxes(range=[pd.to_datetime(start_date).date(), pd.to_datetime(end_date).date()])

    return fig

def visualize_daily_hrv_v2(df, start_date=None, end_date=None):
    # Ensure the columns are of the correct type
    df["day"] = pd.to_datetime(df["day"]).dt.date

    # Extract HRV values and create a new DataFrame
    hrv_data = []
    for index, row in df.iterrows():
        hrv_values = row["hrv"]["items"]
        interval = row["hrv"]["interval"]
        timestamp = pd.to_datetime(row["hrv"]["timestamp"])
        day = row["day"]
        for i, hrv in enumerate(hrv_values):
            if hrv is not None:
                hrv_time = timestamp + pd.to_timedelta(i * interval, unit='s')
                hrv_data.append({"timestamp": hrv_time, "hrv": hrv, "day": day})

    hrv_df = pd.DataFrame(hrv_data)

    # Calculate the daily average HRV
    daily_avg_hrv = hrv_df.groupby("day")["hrv"].mean().reset_index()

    fig = go.Figure()

    # Add scatter plot for individual HRV values
    fig.add_trace(go.Scatter(
        x=hrv_df["timestamp"],
        y=hrv_df["hrv"],
        mode='markers',
        name='HRV',
        marker=dict(color='blue')
    ))

    # # Add line plot for daily average HRV
    # fig.add_trace(go.Scatter(
    #     x=daily_avg_hrv["day"],
    #     y=daily_avg_hrv["hrv"],
    #     mode='lines+markers',
    #     name='Daily Average HRV',
    #     line=dict(color='red')
    # ))

    fig.update_layout(
        title='Daily HRV During Sleep',
        xaxis_title='Timestamp',
        yaxis_title='HRV',
        legend_title='Metric',
        showlegend=True
    )

    # Set the x-axis range if start_date and end_date are provided
    if start_date and end_date:
        fig.update_xaxes(range=[pd.to_datetime(start_date), pd.to_datetime(end_date)])

    return fig
