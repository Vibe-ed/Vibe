import os
from flask import Flask, Response, request, render_template
import convert_data

app = Flask(__name__)

@app.route('/')
def index():
    return 'Hello, World!'


@app.route('/get_data')
def get_algo_data():
    return convert_data.get_sample_date_json()


if __name__ == '__main__':
    # app.run(debug=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

