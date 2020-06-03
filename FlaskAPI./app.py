from flask.ext.api import FlaskAPI
app = FlaskAPI(__name__)

@app.route('/getData/')
def getData():
    return {'name':'roy'}

if __name__=="__main__":
    app.run(debug=True)

