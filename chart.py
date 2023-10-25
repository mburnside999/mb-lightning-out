from flask import Flask, render_template
import plotly.graph_objs as go

app = Flask(__name__)

@app.route('/')
def index():
    # Create a simple Plotly chart
    data = [go.Scatter(x=[1, 2, 3], y=[1, 3, 2], mode='lines+markers')]
    layout = go.Layout(title='Sample Chart')
    chart = go.Figure(data=data, layout=layout)

    # Render the chart using the plotly.offline.plot function
    chart_div = chart.to_html(full_html=False, default_height=400, default_width=600)

    return render_template('index.html', chart_div=chart_div)

if __name__ == '__main__':
    app.run(port=5004)
