from flask import Flask, render_template, request
from trainer import Feedback

class_feedback = Feedback()
app = Flask(__name__)


@app.route("/")
def hello():
    return render_template('home.html')


@app.route("/api/review", methods=['POST'])
def get_result():
    text = request.form['review']
    print("Text: ", text)
    result = class_feedback.run(text=text)

    return result


if __name__ == '__main__':
    app.run(host='0.0.0.0')
