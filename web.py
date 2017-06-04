
import flask
from flask import Flask, render_template

import load_corpora

app = Flask(__name__)

@app.route("/")

def hello():
    return render_template('index.html', ngram="2, 5")



@app.route("/go", methods=["POST"])

def run_with_params():

	strip = lambda x: int(x.strip())

	ngram_range = tuple(map(strip, (flask.request.form['ngram'].split(','))))

	res = load_corpora.main(ngram_range=ngram_range)

	print(res)

	return render_template('index.html', res=res, ngram=flask.request.form['ngram'])


if __name__ == "__main__":
    app.run(debug=True)