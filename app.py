from flask import Flask, jsonify, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)


dataset = pd.read_csv('data/CSV_FILE_PATH.csv')

def find_player_data(stadium_name, player_name):
    filtered_data = dataset[
        (dataset['Stadium_Name'] == stadium_name) &
        (dataset['Player_Name'] == player_name) &
        (dataset.apply(lambda row: all(cell != 'NA' for cell in row), axis=1))
    ]
    return filtered_data.replace({np.nan: None}).to_dict(orient='records')

def generate_bar_chart(stadium_name, player_name):
    filtered_data = find_player_data(stadium_name, player_name)

    if not filtered_data.empty:
        statistics = ['Highest_Runs', 'Average_Runs', 'Strike_Rate', 'Total_Wickets', 'Economy', 'Dots']
        player_values = [filtered_data.iloc[0][statistic] or 0 for statistic in statistics]

        plt.figure(figsize=(8, 6))
        plt.bar(statistics, player_values)
        plt.title(f"Statistics Comparison for {player_name} in {stadium_name}")
        plt.xlabel("Statistic")
        plt.ylabel("Value")
        plt.xticks(rotation=45)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        chart_data = base64.b64encode(buf.read()).decode()
        buf.close()
        return chart_data

    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/player_data/<stadium_name>/<player_name>')
def get_player_data(stadium_name, player_name):
    filtered_data = find_player_data(stadium_name, player_name)
    return jsonify(filtered_data)

@app.route('/api/compare_players', methods=['POST'])
def compare_players():
    data = request.json
    player_name1 = data['player_name1']
    player_name2 = data['player_name2']
    stadium_name = data['stadium_name']

    player1_chart_data = generate_bar_chart(stadium_name, player_name1)
    player2_chart_data = generate_bar_chart(stadium_name, player_name2)

    response_data = {
        'player1_chart_data': player1_chart_data,
        'player2_chart_data': player2_chart_data
    }
    return jsonify(response_data)

@app.route('/api/stadiums')
def get_stadiums():
    stadiums = dataset['Stadium_Name'].dropna().unique().tolist()
    return jsonify(stadiums)

@app.route('/api/players')
def get_players():
    players = dataset['Player_Name'].dropna().unique().tolist()
    return jsonify(players)

if __name__ == '__main__':
    app.run(debug=True)
