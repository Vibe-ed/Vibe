# import os
from flask import Flask, Response, request, render_template
import convert_data

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_data')
def get_algo_data():
    return convert_data.get_sample_date_json()


if __name__ == '__main__':
    app.run(debug=True)

