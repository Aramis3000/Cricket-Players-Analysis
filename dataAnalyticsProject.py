#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def load_dataset(file_path):
    dataset = pd.read_csv(file_path)
    return dataset

def preprocess_dataset(dataset, is_batsman=True):
    dataset = dataset.dropna(subset=['Stadium_Name', 'Player_Name'])
    dataset = pd.get_dummies(dataset, columns=['Role', 'Country'])
    
    if is_batsman:
        # Drop columns related to bowling stats for batsmen
        dataset = dataset.drop(columns=['Total_Wickets', 'Economy', 'Dots'])
    else:
        # Drop columns related to batting stats for bowlers
        dataset = dataset.drop(columns=['Highest_Runs', 'Average_Runs', 'Strike_Rate'])
    
    return dataset

def calculate_moving_average(data, window_size=3):
    return data.rolling(window=window_size).mean()

def get_player_stats(dataset, player_name):
    player_stats = dataset.loc[dataset['Player_Name'] == player_name].copy()
    return player_stats

def visualize_player_performance(player_stats, is_batsman=True):
    if is_batsman:
        metrics = ['Highest_Runs', 'Average_Runs', 'Strike_Rate']
        yaxis_title = 'Performance Metrics'
    else:
        metrics = ['Highest_Runs', 'Economy', 'Total_Wickets']
        yaxis_title = 'Bowling Stats'

    colors = px.colors.qualitative.Set2
    fig = go.Figure()

    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(x=player_stats['Stadium_Name'], y=player_stats[metric],
                             name=f'{player_stats.iloc[0]["Player_Name"]} - {metric}', marker_color=colors[i],
                             text=player_stats[metric], textposition='auto'))

    fig.update_layout(title=f'{player_stats.iloc[0]["Player_Name"]} Performance in Different Stadiums',
                      xaxis_title='Stadium Name', yaxis_title=yaxis_title,
                      xaxis_tickangle=-45, barmode='group')
    fig.show()


def visualize_career_trajectory_multiple_players(player_data_list, is_batsman=True):
    colors = px.colors.qualitative.Dark24

    if is_batsman:
        metric = 'Average_Runs'
        yaxis_title = 'Average Runs'
    else:
        metric = 'Total_Wickets'
        yaxis_title = 'Total Wickets'

    fig = go.Figure()

    for i, player_data in enumerate(player_data_list):
        player_name = player_data.iloc[0]['Player_Name']
        fig.add_trace(go.Scatter(x=player_data['Match_ID'], y=player_data[metric],
                                 mode='lines+markers', name=player_name, line=dict(color=colors[i]),
                                 marker=dict(color=colors[i], size=8),
                                 text=player_data['Stadium_Name'], hoverinfo='text+y'))

    fig.update_layout(title=f'Career Trajectory of {"Batsmen" if is_batsman else "Bowlers"}',
                      xaxis_title='Match ID', yaxis_title=yaxis_title)
    fig.show()


def train_test_split_data(player_stats, test_size=0.2, random_state=42):
    X = player_stats[['Stadium_ID']]
    y = player_stats['Average_Runs']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def build_predictive_model(X_train, y_train):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

def suggest_players_in_stadium(dataset, stadium_name, is_batsman=True, top_n=5):
    stadium_data = dataset[dataset['Stadium_Name'] == stadium_name]
    if stadium_data.empty:
        print(f"No data found for stadium {stadium_name}.")
        return

    dataset = preprocess_dataset(dataset, is_batsman=is_batsman)

    if is_batsman:
        key_metric = 'Average_Runs'
        players_in_stadium = stadium_data.sort_values(by='Average_Runs', ascending=False)['Player_Name'].unique()
    else:
        key_metric = 'Total_Wickets'
        players_in_stadium = stadium_data.sort_values(by='Total_Wickets', ascending=False)['Player_Name'].unique()

    if is_batsman:
        print(f"\nSuggested Batsmen in {stadium_name} based on Average Runs:")
    else:
        print(f"\nSuggested Bowlers in {stadium_name} based on Total Wickets:")

    for player in players_in_stadium[:top_n]:
        player_data = get_player_stats(dataset, player)
        if not player_data.empty:
            if is_batsman:
                actual_runs = player_data['Average_Runs'].iloc[0]
                print(f"{player} - Average Runs: {actual_runs:.2f}")
            else:
                total_wickets = player_data['Total_Wickets'].iloc[0]
                print(f"{player} - Total Wickets: {total_wickets}")

def main():
    file_path = r'Player vs Stadium Dataset 2.csv'
    dataset = load_dataset(file_path)

    analysis_choice = input("Do you want to analyze batsmen or bowlers? (Batsmen/Bowlers): ").lower()

    if analysis_choice == 'batsmen':
        input_player_names = input("Enter the batsman names separated by commas (e.g., Player 1,Player 2): ").split(',')
        player_data_list = []
        for player_name in input_player_names:
            player_data = get_player_stats(dataset, player_name.strip())
            if not player_data.empty:
                player_data_list.append(player_data)
                print("\nBatsman Information:")
                print(player_data[['Player_Name', 'Role', 'Country']].iloc[0])

                print("\nBatsman Performance Statistics:")
                print(player_data[['Stadium_Name', 'Highest_Runs', 'Average_Runs', 'Strike_Rate']])

                # Visualize individual player performance
                visualize_player_performance(player_data)

        if not player_data_list:
            print("No data found for the given batsmen.")
            return

        # Create a comparison graph for average runs of input batsmen
        compare_fig = go.Figure()
        colors = px.colors.qualitative.Plotly

        for i, player_data in enumerate(player_data_list):
            player_name = player_data.iloc[0]['Player_Name']
            compare_fig.add_trace(go.Bar(x=player_data['Stadium_Name'], y=player_data['Average_Runs'],
                                         name=player_name, marker_color=colors[i]))

        compare_fig.update_layout(title="Comparison of Average Runs for Input Batsmen",
                                  xaxis_title="Stadium Name", yaxis_title="Average Runs",
                                  xaxis_tickangle=-45, barmode='group')
        compare_fig.show()

        # Merge career trajectories of batsmen into one plot
        visualize_career_trajectory_multiple_players(player_data_list, is_batsman=True)

        input_stadium_name = input("Enter the stadium name to suggest batsmen: ")
        suggest_players_in_stadium(dataset, input_stadium_name, is_batsman=True)

    elif analysis_choice == 'bowlers':
        input_player_names = input("Enter the bowler names separated by commas (e.g., Player 1,Player 2): ").split(',')
        player_data_list = []
        for player_name in input_player_names:
            player_data = get_player_stats(dataset, player_name.strip())
            if not player_data.empty:
                player_data_list.append(player_data)
                print("\nBowler Information:")
                print(player_data[['Player_Name', 'Role', 'Country']].iloc[0])

                print("\nBowler Performance Statistics:")
                print(player_data[['Stadium_Name', 'Economy', 'Total_Wickets']])

                # Visualize individual player performance
                visualize_player_performance(player_data, is_batsman=False)

        if not player_data_list:
            print("No data found for the given bowlers.")
            return

        # Create a comparison graph for bowlers based on total wickets
        compare_fig = go.Figure()
        colors = px.colors.qualitative.Plotly

        for i, player_data in enumerate(player_data_list):
            player_name = player_data.iloc[0]['Player_Name']
            compare_fig.add_trace(go.Bar(x=player_data['Stadium_Name'], y=player_data['Total_Wickets'],
                                         name=player_name, marker_color=colors[i]))

        compare_fig.update_layout(title="Comparison of Total Wickets for Input Bowlers",
                                  xaxis_title="Stadium Name", yaxis_title="Total Wickets",
                                  xaxis_tickangle=-45, barmode='group')
        compare_fig.show()

        # Merge career trajectories of bowlers into one plot
        visualize_career_trajectory_multiple_players(player_data_list, is_batsman=False)

        input_stadium_name = input("Enter the stadium name to suggest bowlers: ")
        suggest_players_in_stadium(dataset, input_stadium_name, is_batsman=False)

    else:
        print("Invalid choice. Please choose either 'Batsmen' or 'Bowlers'.")

if __name__ == "__main__":
    main()


# In[ ]:




