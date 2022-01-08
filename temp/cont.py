from flask import Flask, render_template, request, Response, session, send_from_directory
from flask_bootstrap import Bootstrap
import os, signal
import time
import datetime
import uuid
from model import InputForm

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(days=100)
sessionkey = str(uuid.uuid4())
#sessionkey = "63c70dbc-d331-471a-a014-bb2036e32649"
app.secret_key = sessionkey

from app import f

@app.route('/', methods=['GET', 'POST'])
def index():
    form = InputForm(request.form)
    print(1, session)

    if "a" not in session: session["a"]=-1

    if request.method == 'POST' and form.validate():
        session["a"]=10
        session["b"]=456
        print(1.5, session)
        f(form.filter.data)
        session["a"]=2

    print(2, session)

    return render_template("temp" + '.html', test=session["a"], form=form)

if __name__ == '__main__':
    app.run(debug=True)