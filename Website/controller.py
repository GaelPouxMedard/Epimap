from model import InputForm
from flask import Flask, render_template, request, Response, session, send_from_directory, redirect
from flask_bootstrap import Bootstrap
import Analyse as Analyse
import os, signal
import time
import datetime
import uuid
import shutil
import threading
import multiprocessing
import pickle

#os.chdir("/home/gpouxmed/EpiMap/")

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(days=1)

sessionkey = str(uuid.uuid4())
sessionkey = "63c70dbc-d330-471a-a014-bb2036e32649"
app.secret_key = sessionkey

sizecache=300*1051084  # En Mo*conversion

template_name = 'niceTemplate'

Bootstrap(app)


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate,public, max-age=0"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r

# Je le garde juste comme exemple ; qd qqn fait une requete GET (pour DL un fichier apr exemple) depuis la
# direction donnée, on peut récupérer les variables dans la fonction, et envoyer le fichier ensuite ac send_from_dir
@app.route("/static/users/<username>/<file>", methods=['GET'])
def resetCounterSession(username, file):
    now=datetime.datetime.now()
    session['last_active'] = now

    return send_from_directory(f"./static/users/{username}", file)

def size(path):
    total_size = 0

    for dirpath, dirnames, filenames in os.walk(path):
        for i in filenames:
            f = os.path.join(dirpath, i)
            total_size += os.path.getsize(f)
    return total_size

def maybewipecache():
    if size(f"./static/users/")>sizecache:
        listDirs = os.listdir("./static/users/")
        for dir in listDirs:
            shutil.rmtree(f"./static/users/"+dir)
            if not len(listDirs)>1000:
                os.mkdir(f"./static/users/"+dir)

def wipeUsrFolder(foldername, toKeep):
    if foldername[0]!="./static/": foldername = "./static/"+foldername
    listFiles = [foldername+f for f in os.listdir(foldername) if os.path.isfile(os.path.join(foldername, f))]

    for file in listFiles:
        rem=True
        for k in toKeep:
            if k in file:
                rem=False
                break
        if rem:
            try:
                os.remove(file)
            except:
                pass

def login(ipUsr):
    if "id" not in session:
        maybewipecache()
        session["id"] = str(uuid.uuid4())
        if ipUsr is not None:
            session["ip"]=ipUsr
    if not os.path.isdir(f"./static/users"):
        os.mkdir(f"./static/users")
    if not os.path.isdir(f"./static/users/{session['id']}"):
        os.mkdir(f"./static/users/{session['id']}")
    if "ip" not in session and ipUsr is not None:
        session["ip"]=ipUsr

    if 'id' in session and 'ip' in session and not "registered" in session:
        with open(f"./static/visits.txt", "a") as f:
            f.write(str(datetime.datetime.now())+"\t"+session["id"]+"\t"+session["ip"]+"\n")
        session["registered"]=True


@app.route('/progress/<var>')
def progress(var):
    time.sleep(0.5)
    login(None)
    def generate():
        try:
            with open(f"./static/users/{var}/prog.txt", "r") as f:
                age, lage, uage = f.read().replace("\n", "").split("\t")
                age, lage, uage = float(age), float(lage), float(uage)-1
                p = int((age-lage)*100./(uage-lage))
        except Exception as e:
            p=0

        yield "data:" + str(p) + "\n\n"

    response = Response(generate(), mimetype='text/event-stream')
    return response

@app.route("/get-dataset")
def get_file():
    filterTreated = session["filter"].replace(",", "").replace("\r", "").lower().split("\n")
    filterStatusTreated = session["filterStatus"].replace(",", "").replace("\r", "").lower().split("\n")
    filterTreated = [flt for flt in filterTreated if flt != '']
    filterStatusTreated = [flt for flt in filterStatusTreated if flt != '']
    if session["zoom"]: bbUsr = [session["xmin"], session["ymin"], session["xmax"], session["ymax"]]
    else: bbUsr = [0, 0, 6698000, 4896000]
    data = Analyse.getData(filterTreated, session["filterOperator"], filterStatusTreated, session["filterStatusOperator"], bbUsr, session["exclureRome"], session["datasetApprox"], session["noDates"])
    #generator = yield data[7]

    return Response(data[7], mimetype="text/plain", headers={"Content-Disposition": "attachment;filename=dataset.csv"})


@app.route("/doneComputing")
def doneComputing():
    with open(f"./static/users/{session['id']}/stop.txt", "w+") as f:
        f.write(str(1))
    with open(f"./static/users/{session['id']}/submitted.txt", "w+") as f:
        f.write(str(0))
    with open(f"./static/users/{session['id']}/prog.txt", "w+") as f:
        f.write(str(0)+"\t"+str(0)+"\t"+str(1))

    return redirect("/", code=302)

@app.route('/', methods=['GET', 'POST'])
def index():
    form = InputForm(request.form)

    try:out = open(f"./static/users/{session['id']}/out.html", "r").read()
    except:out = None
    try:
        filename = f"./static/users/{session['id']}/{session['lastfilename']}"
        figHistWords = open(f"./static/users/{session['id']}/histNames.html", "r").read()
        figHistStat = open(f"./static/users/{session['id']}/histStat.html", "r").read()
        figMetrics = open(f"./static/users/{session['id']}/metrics.html", "r").read()
    except:
        filename = "./static/demo/Ages.mp4"
        figMetrics = open("./static/demo/metrics.html", "r").read()
        figHistStat = open("./static/demo/histStat.html", "r").read()
        figHistWords = open("./static/demo/histNames.html", "r").read()

    login(None)
    session['last_active'] = datetime.datetime.now()

    try:
        with open(f"./static/users/{session['id']}/submitted.txt", "r") as f:
            session["submitted"] = int(f.read())
    except: session["submitted"] = False

    if request.method == 'GET':
        try:
            for field in form:
                field.data = session[field.id]
            if form.noDates.data == True:
                form.anim.data = False
                form.lage.data = 0
                form.uAge.data = 1

        except Exception as e:
            form.plotPoints.data = False
            form.plotKde.data = True
            form.plotHist2d.data = False
            form.anim.data = True
            form.weighted.data = True
            form.cities.data = True
            form.fixedvmax.data = False
            form.plotClus.data = True
            form.imageOnly.data = False
            form.exclureRome.data = False
            form.datasetApprox.data = False
            form.noDates.data = False


    if request.method == 'POST' and form.validate():
        with open(f"./static/users/{session['id']}/stop.txt", "w+") as f:
            f.write(str(0))

        ipUsr = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
        login(ipUsr)
        session['last_active'] = datetime.datetime.now()

        with open(f"./static/users/{session['id']}/submitted.txt", "w+") as f:
            f.write(str(1))

        for field in form:
            session[field.id] = field.data
        if form.noDates.data == True:
            form.anim.data = False
            form.lAge.data = 0
            form.uAge.data = 1

        filterTreated = form.filter.data.replace(",", "").replace("\r", "").lower().split("\n")
        filterStatusTreated = form.filterStatus.data.replace(",", "").replace("\r", "").lower().split("\n")
        filterTreated = [flt for flt in filterTreated if flt!='']
        filterStatusTreated = [flt for flt in filterStatusTreated if flt!='']
        if form.zoom.data: bbUsr = [form.xmin.data, form.ymin.data, form.xmax.data, form.ymax.data]
        else: bbUsr = [0, 0, 6698000, 4896000]

        data = Analyse.getData(filterTreated, form.filterOperator.data, filterStatusTreated, form.filterStatusOperator.data, bbUsr, form.exclureRome.data, form.datasetApprox.data, form.noDates.data)
        histWords = data[5]
        histStat = data[6]
        nbInscriptions = len(data[1])
        figHistWords = Analyse.plotFigHistWords(histWords, nbInscriptions, folder=f"./static/users/{session['id']}")
        figHistStat = Analyse.plotFigHistStat(histStat, nbInscriptions, folder=f"./static/users/{session['id']}")

        print("Started", filterTreated, filterStatusTreated)
        session["plots"] = []
        if form.plotPoints.data: session["plots"].append("points")
        if form.plotKde.data: session["plots"].append("kde")
        if form.plotHist2d.data: session["plots"].append("hist2d")
        fig, cb, bg = Analyse.init(f"./static/users/{session['id']}", data, session["plots"], form.style.data, cities=form.cities.data, bbUsr=bbUsr)

        try:
            args = (data, fig, cb,
                    form.gridSize.data,
                    form.lAge.data,
                    form.uAge.data,
                    filterTreated,
                    form.anim.data,
                    form.noDates.data,
                    form.weighted.data,
                    session["plots"],
                    form.fps.data,
                    form.fixedvmax.data,
                    form.vmax.data,
                    form.style.data,
                    form.sizeScatter.data,
                    f"./static/users/{session['id']}",
                    form.imageOnly.data,
                    form.filterOperator.data,
                    filterStatusTreated,
                    form.filterStatusOperator.data,
                    form.plotClus.data,
                    form.radClus.data,
                    bg,
                    bbUsr)


            # import pprofile
            # profiler = pprofile.Profile()
            # with profiler:
            filename, out, figMetrics = Analyse.run(*args)
            # profiler.dump_stats("Benchmark.txt")


        except Exception as e:
            with open(f"./static/users/{session['id']}/submitted.txt", "w+") as f:
                f.write(str(0))
            return f"There was en error during the computation of the results: {e}.<br>This might happen for the following reasons:<br>\
                - The request was too large to handle (too many elements to display, too much resolution asked). Note that the hosting website (PythonAnywhere) stops uncompleted requests after 5min.\n\
                - You launched several requests after reloading the page.<br>\
                - There are too many persons using EpiMap right now.<br><br>\
                Taking this in consideration, you can <a href='https://epimap.msh-lse.fr'>try EpiMap again</a>."

        with open(f"./static/users/{session['id']}/submitted.txt", "w+") as f:
            f.write(str(0))

        print("Finished")

    folderShort = filename[filename[2:].find("/")+2:]
    folderShort = folderShort[:folderShort.rfind("/")]+"/"

    session['lastfilename'] = filename[filename.rfind("/"):]
    picname = filename[filename.rfind("/"):]
    wipeUsrFolder(folderShort, toKeep=[picname, picname.replace('.jpg', '.pdf'), "prog", "stop", "submitted", "histNames", "histStat", "metrics"])

    outputFileName = folderShort+filename[filename.rfind("/"):]
    outputFileNamePdf = outputFileName.replace(".jpg", ".pdf")

    return render_template(template_name + '.html', form=form, filename=filename, outputFileName=outputFileName, outputFileNamePdf=outputFileNamePdf, folderShort=folderShort, out=out,
                           figHistWords=figHistWords, figHistStat=figHistStat, figMetrics=figMetrics, pending=session["submitted"])

if __name__ == '__main__':
    app.run(debug=True)