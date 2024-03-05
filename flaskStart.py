from flask import Flask,render_template,redirect

app=Flask(__name__)

@app.route('/')
def hello():
    print("Hello")
    return render_template('index3.html')

app.run(debug=True)