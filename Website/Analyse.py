import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap
import mpld3
from mpld3 import plugins
from copy import copy
import re
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.stats import gaussian_kde
import datetime


cyclecolors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#e372e2", "#73cf7f", "#b9ed22", "#1a5ecf", "#9400bd", "#8cd94b", "#ec37c2", "#7f0e7f", "#bcb3a2", "#1703cf"]*10
#cyclecolors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf']*10


import os
#os.chdir("/home/gpouxmed/EpiMap/")

'''
import pprofile
profiler = pprofile.Profile()
with profiler:
    RhovsT(g, 0.1, 0.28999, 1, 1000)
profiler.print_stats()
profiler.dump_stats("Benchmark.txt")
'''


# Modified classes to make video writing faster

class FasterFFMpegWriter(FFMpegWriter):
    '''FFMpeg-pipe writer bypassing figure.savefig.'''
    def __init__(self, **kwargs):
        '''Initialize the Writer object and sets the default frame_format.'''
        super().__init__(**kwargs)
        self.frame_format = 'argb'

    def grab_frame(self, **savefig_kwargs):
        '''Grab the image information from the figure and save as a movie frame.

        Doesn't use savefig to be faster: savefig_kwargs will be ignored.
        '''
        try:
            # re-adjust the figure size and dpi in case it has been changed by the
            # user.  We must ensure that every frame is the same size or
            # the movie will not save correctly.
            self.fig.set_size_inches(self._w, self._h)
            self.fig.set_dpi(self.dpi)
            # Draw and save the frame as an argb string to the pipe sink
            #self.fig.canvas.draw()

            # self._frame_sink().write(self.fig.canvas.tostring_argb())
            self._proc.stdin.write(self.fig.canvas.tostring_argb())
        except (RuntimeError, IOError) as e:
            out, err = self._proc.communicate()
            raise IOError('Error saving animation to file (cause: {0}) '
                      'Stdout: {1} StdError: {2}. It may help to re-run '
                      'with --verbose-debug.'.format(e, out, err))

class FasterFuncAnimation(FuncAnimation):
    '''FFMpeg-pipe writer bypassing figure.savefig.'''
    def __init__(self, fig, func, background, **kwargs):
        '''Initialize the Writer object and sets the default frame_format.'''
        super().__init__(fig, func, **kwargs)
        self.background=background

    def _post_draw(self, framedata, blit):
        if blit and self._drawn_artists:
            self._blit_draw(self._drawn_artists, self._blit_cache)
        else:
            #self._fig.canvas.draw_idle()

            renderer = self._fig._cachedRenderer
            # https://matplotlib.org/_modules/matplotlib/axis.html#YAxis

            self._fig.canvas.restore_region(self.background[0])
            #self._fig.canvas.restore_region(self.background[1])

            self._fig.axes[0].draw_artist(self._fig.axes[0].title)
            for c in self._fig.axes[0].collections:
                self._fig.axes[0].draw_artist(c)

            #print(self._fig.axes[1].get_yaxis().get_children())
            #print(self._fig.axes[1].get_yaxis().get_major_ticks()[0].get_children())
            #print(self._fig.axes[1].get_yaxis().get_label())

            self._fig.axes[1].draw_artist(self._fig.axes[1].get_yaxis())  # Meme en creusant dur à optim.

            for c in self._fig.axes[1].collections:
                self._fig.axes[1].draw_artist(c)

            #self._fig.canvas.blit(self._fig.axes[0].bbox)


# Script (local) global variables
try:
    PS
except:
    PS={}  # parameters session

# Compute backgrounds and convert coordinate systems
def initMaps():
    from mpl_toolkits.basemap import Basemap
    bounds = np.array([-10.424482, 25.728333, 49.817223, 57.523959])
    matplotlib.use("TkAgg")
    plt.figure(figsize=(8, 5))
    minLon, minLat, maxLon, maxLat = bounds

    m = Basemap(projection="merc", llcrnrlon=minLon, llcrnrlat=minLat, urcrnrlon=maxLon, urcrnrlat=maxLat, resolution="c", area_thresh=2000.)


    labCit, longCit, latCit = [], [], []
    with open("Cities.txt") as f:
        for l in f:
            lab, lat, long = l.replace("\n", "").split(", ")
            labCit.append(lab)
            x,y = m(float(long), float(lat))
            longCit.append(x)
            latCit.append(y)

    cities=True
    #m.drawcoastlines()
    #m.drawlsmask(land_color='grey', ocean_color='white', lakes=False)
    m.bluemarble()
    font = {'fontname': 'Helvetica'}


    if cities:
        plt.scatter(longCit, latCit, color='darkorange', s=20, marker="*", zorder=20)
        for i in range(len(labCit)):
            align = "center"
            shifty=0.5
            if labCit[i]=="Lugdunum":
                align = "right"
            if labCit[i]=="Mediolanum":
                align = "left"
            if labCit[i]=="Ravenna":
                align = "left"
            if labCit[i]=="Ephesus":
                align = "left"
            if labCit[i]=="Athens":
                align = "right"
            if labCit[i]=="Constantinople":
                align = "left"
            #plt.text(longCit[i], latCit[i]+shifty, labCit[i], horizontalalignment=align, color="black", **font, zorder=20)

    def getco(arr):
        arr = eval(arr)
        y, x = m(arr[1], arr[0])
        return [x,y]

    df = pd.read_csv("dataLongLat.csv", sep="\t")
    dfTemp = df["coords"].apply(getco)
    df2=df.copy()
    df2["coords"]=dfTemp
    df2.to_csv("dataMerc.csv", sep="\t", index=False)
    pause()

    style="bluemarbleCities"
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.savefig(f"static/img/map_{style}.jpg", bbox_inches='tight', dpi=600)
    plt.close()


# Get data
def filt(filters, filtersJoint, inscr, operator):
    res = set(re.findall(filtersJoint[0], inscr))
    resneg = set(re.findall(filtersJoint[1], inscr))
    res = list(res-resneg)

    if operator=="or":
        pres=False
        if len(res)!=0:
            pres=True
        else:
            return pres, []
    else:
        pres=True
        for flt in filters:
            if flt not in inscr:
                pres=False
                return pres, []
    return pres, res

def getData(filter, filterOperator, filterStatus, filterStatusOperator, bbUsr, exclureRome, datApprox, noDates):

    if filter==[""] or filter==[]: filter=None
    if filterStatus==[""] or filterStatus==[]: filterStatus=None
    if filterStatus is not None:
        filterStatusJoint = '|'.join(filterStatus).lower()

    histWords = {}
    histStat = {}
    fn = "data"
    if exclureRome:
        fn += "_ssRome"
    if noDates:
        fn += "_noDates"
    elif datApprox:
        fn += "_approx"

    df = pd.read_csv(fn+".csv", sep="\t")

    if filter is not None:
        ft = [s.lower() for s in filter if s[0]!="-"]
        filterTrue = ["\w*"+s+"\w*" for s in ft]
        filterTrue = '|'.join(filterTrue)
        ff = [s.replace("-", "").lower() for s in filter if s[0]=="-"]
        filterFalse = ["\w*"+s+"\w*" for s in ff]
        filterFalse = '|'.join(filterFalse)
        filtersJoint = [filterTrue, filterFalse]

        if len(ft)!=0:
            df = df[df['inscription'].str.contains('|'.join(ft))]
    if filterStatus is not None:
        df = df[df['status'].str.contains(filterStatusJoint)]

    arrDatesMean, arrCoords = [], []
    arrDatesL, arrDatesU = [], []
    arrTxts = "EDCS;date-;date+;province;coord_x;coord_y;inscr;category\n"

    for j, line in enumerate(df.values):
        EDCS, dates, province, coords, inscr, statusStr = line
        if statusStr=="0":
            status=None
        else:
            status=statusStr

        coordsNum = np.array(re.sub(r'[\[|\]]', ' ', coords).split(", "), dtype=float)

        if coordsNum[1]<bbUsr[0] or coordsNum[1]>bbUsr[2] or coordsNum[0]<bbUsr[1] or coordsNum[0]>bbUsr[3]:
            continue

        if filter is None:
            resHist = ""
            if inscr is not None:
                resHist = inscr.strip().split(" ")
            pass
        else:
            if inscr is np.nan:
                continue
            else:
                pres, resHist = filt(filter, filtersJoint, inscr, filterOperator)
                if not pres:
                    continue

        if filterStatus is None:
            resStat = ""
            if status is not None:
                resStat = status.replace("et ", "").strip().split(" ")
            pass
        else:
            if status is None:
                continue
            else:
                pres, resStat = filt(filterStatus, filterStatusJoint, status, filterStatusOperator)
                if not pres:
                    continue

        for wd in resHist:
            if wd not in histWords: histWords[wd] = 0
            histWords[wd] += 1
        for wd in resStat:
            if wd not in histStat: histStat[wd] = 0
            histStat[wd] += 1

        datesNum = np.array(re.sub(r'[\[|\]]', ' ', dates).split(", "), dtype=float)

        arrDatesMean.append(datesNum[0]+(datesNum[1]-datesNum[0])/2)  # On associe une inscription avec 40 ans de vie
        arrCoords.append([coordsNum[1], coordsNum[0]])
        arrDatesL.append(datesNum[0])
        arrDatesU.append(datesNum[1])

        arrTxts += str(EDCS)+";"+str(datesNum[0])+";"+str(datesNum[1])+";"+province+";"+str(coordsNum[0])+";"+str(coordsNum[1])+";"+inscr+";"+statusStr+"\n"


    arrDatesMean, arrCoords, arrDatesL, arrDatesU = np.array(arrDatesMean), np.array(arrCoords), np.array(arrDatesL), np.array(arrDatesU)
    bounds = np.array([-10.424482, 25.728333, 49.817223, 57.523959])
    bounds = np.array([0, 0, 6698568.814169849, 4897013.401366526])

    return arrDatesMean, arrCoords, arrDatesL, arrDatesU, bounds, histWords, histStat, arrTxts


# Plot metadata
def plotFigHistWords(histWords, nbInscriptions, folder=""):
    fig, ax = plt.subplots(figsize=(5,4))
    listWds, cntWds = list(histWords.keys()), list(histWords.values())

    if len(cntWds) == 0:
        return "No inscription found."
    sortedCntWds, sortedListWds = zip(*sorted(zip(cntWds, listWds)))
    sortedCntWds, sortedListWds = list(sortedCntWds), list(sortedListWds)
    lg = len(sortedCntWds)
    ax.barh(list(range(lg))[-100:], sortedCntWds[-100:], color="#ecc882", tick_label=sortedListWds[-100:])

    ax.set_ylim([lg-30, lg])
    ax.set_xlabel("Count")
    ax.set_title(f"Keywords - from {nbInscriptions} inscriptions")
    fig.tight_layout()
    fig.savefig(f"{folder}/histNames.pdf")
    fig.savefig(f"{folder}/histNames.jpg")
    plugins.clear(fig)
    plugins.connect(fig, plugins.Reset(), plugins.BoxZoom(), plugins.Zoom(enabled=True))
    html = mpld3.fig_to_html(fig)
    w, h = fig.get_size_inches()*fig.dpi
    html = html.replace(f"\"width\": {round(w, 1)}", "\"width\": document.getElementById(\"divHistNames\").clientWidth*0.9")
    with open(f"{folder}/histNames.html", "w+") as f:
        f.write(html)

    plt.close(fig)

    return html

def plotFigHistStat(histStat, nbInscriptions, folder=""):
    fig, ax = plt.subplots(figsize=(5,4))
    listWds, cntWds = list(histStat.keys()), list(histStat.values())

    if len(cntWds) == 0:
        return "No inscription found."
    sortedCntWds, sortedListWds = zip(*sorted(zip(cntWds, listWds)))
    sortedCntWds, sortedListWds = list(sortedCntWds), list(sortedListWds)
    lg = len(sortedCntWds)
    ax.barh(list(range(lg))[-100:], sortedCntWds[-100:], color="#ecc882", tick_label=sortedListWds[-100:])

    ax.set_ylim([lg-30, lg])
    ax.set_xlabel("Count")
    ax.set_title(f"Status - from {nbInscriptions} inscriptions")
    fig.tight_layout()
    fig.savefig(f"{folder}/histStat.pdf")
    fig.savefig(f"{folder}/histStat.jpg")
    plugins.clear(fig)
    plugins.connect(fig, plugins.Reset(), plugins.BoxZoom(), plugins.Zoom(enabled=True))
    html = mpld3.fig_to_html(fig)
    w, h = fig.get_size_inches()*fig.dpi
    html = html.replace(f"\"width\": {round(w, 1)}", "\"width\": document.getElementById(\"divHistStat\").clientWidth*0.9")
    with open(f"{folder}/histStat.html", "w+") as f:
        f.write(html)

    plt.close(fig)
    return html

def plotMetrics(folder):
    if len(PS[folder]["metrics"])==0:
        return "No metrics computed."
    elif len(PS[folder]["metrics"]["age"])==0:
        return "No metrics computed."

    fig, axs = plt.subplots(2, 1, figsize=(8,4))

    axs[0].plot(PS[folder]["metrics"]["age"], PS[folder]["metrics"]["pop"], "-", c="blue", label="Population")
    axs[0].set_xlim([min(PS[folder]["metrics"]["age"]), max(PS[folder]["metrics"]["age"])+1])
    axs[0].set_xlabel("Age")
    axs[0].set_ylabel("Population")

    axs[1].plot(PS[folder]["metrics"]["age"], PS[folder]["metrics"]["ent"], "-", c="red", label="Entropy")
    axs[1].plot(PS[folder]["metrics"]["age"], PS[folder]["metrics"]["entMBC"], "-", c="#ecc882", label="Entropy without isolated points")
    axs[1].plot(PS[folder]["metrics"]["age"], PS[folder]["metrics"]["avgDist"], "-", c="g", label="Norm. avg. interpoints distance")
    axs[1].set_xlim([min(PS[folder]["metrics"]["age"]), max(PS[folder]["metrics"]["age"])+1])
    axs[1].set_ylim([0, 1])
    axs[1].set_xlabel("Age")
    axs[1].set_ylabel("Entropy")

    axs[1].legend()

    fig.tight_layout()
    fig.savefig(f"{folder}/metrics.pdf")
    plugins.clear(fig)
    plugins.connect(fig, plugins.Reset(), plugins.BoxZoom(), plugins.Zoom(enabled=True))
    html = mpld3.fig_to_html(fig)
    w, h = fig.get_size_inches()*fig.dpi
    html = html.replace(f"\"width\": {round(w, 1)}", "\"width\": document.getElementById(\"divMetrics\").clientWidth*0.9")
    with open(f"{folder}/metrics.html", "w+") as f:
        f.write(html)

    plt.close(fig)

    return html


# Treat data
def getMetrics(folder, age, coords, weights, writeRes=True):
    if "age" not in PS[folder]["metrics"]: PS[folder]["metrics"]["age"]=[]
    if "ent" not in PS[folder]["metrics"]: PS[folder]["metrics"]["ent"]=[]
    if "entMBC" not in PS[folder]["metrics"]: PS[folder]["metrics"]["entMBC"]=[]
    if "pop" not in PS[folder]["metrics"]: PS[folder]["metrics"]["pop"]=[]
    if "avgDist" not in PS[folder]["metrics"]: PS[folder]["metrics"]["avgDist"]=[]

    from sklearn.metrics import pairwise_distances
    dists = pairwise_distances(coords)
    radclus = 50
    if PS[folder]["plotClus"]==True:
        radclus = PS[folder]["radClus"]
    db = DBSCAN(eps=radclus*1000*(2)**0.5, min_samples=2, metric="precomputed").fit(dists, sample_weight=weights)  # racine pour convertir en km en europe (approx)
    clus = db.labels_

    unique, counts = np.unique(clus, return_counts=True)
    #unique, counts = unique[unique!=-1], counts[unique!=-1]

    pops = {}
    for u in unique:
        pops[u] = np.sum(weights[clus == u])

    ent, entMBC = 0., 0.
    pop = np.sum(weights)
    popMBC = pop
    if -1 in pops: popMBC -= pops[-1]
    nbClus = len(pops)
    for u in pops:
        adj, adjPop, adjPopMBC=0, 0, 0
        popclus = pops[u]

        if popclus==0:
            adj=1e-20

        ent += (popclus/pop)*np.log(popclus/pop+adj)

        if u != -1:
            entMBC += (popclus/popMBC)*np.log(popclus/popMBC+adj)

    ent /= np.log(1./nbClus)+1e-20
    if nbClus>=2:
        entMBC /= np.log(1. / (nbClus-1))+1e-20

    if writeRes:
        PS[folder]["metrics"]["age"].append(age)
        PS[folder]["metrics"]["ent"].append(ent)
        PS[folder]["metrics"]["entMBC"].append(entMBC)
        PS[folder]["metrics"]["pop"].append(pop)
        PS[folder]["metrics"]["avgDist"].append(np.mean(dists)/(PS[folder]["bounds"][2]-PS[folder]["bounds"][0]))

    return clus

def aggregatePoints(toPlot, cols):
    if len(toPlot)==0:
        return toPlot, cols

    coordsUniques, count = np.unique(toPlot, axis=0, return_counts=True)
    coordsToAggregate = coordsUniques[count>20]  # Useless to consider them all

    for c in coordsToAggregate:
        ones = np.zeros((len(toPlot)))
        w = np.where(toPlot == c)[0]
        ones[w]=1
        bols = np.array(ones-1, dtype=bool)
        bols[w[0]]=True  # To avoid stacking arrays

        newWeight = cols[:, 3].dot(ones)

        cols = cols[bols]
        toPlot = toPlot[bols]

        cols[w[0]]=np.array([1, 0, 0, newWeight])

    return toPlot, cols

def treatDataAge(folder, age):
    maxtransp = 20  # La fourchette de dates minimale pour avoir un poids de 1
    inds = (int(age) <= PS[folder]["arrDatesU"]) & (int(age) >= PS[folder]["arrDatesL"]-maxtransp)  # Période sur "maxtransp" ans

    toPlot = copy(PS[folder]["arrCoords"][inds])
    if not PS[folder]["weighted"] or PS[folder]["noDates"] or not PS[folder]["anim"]:
        a = np.ones((len(PS[folder]["arrDatesMean"][inds])))
    else:
        # a = np.abs(PS[folder]["arrDatesMean"][inds] - age)
        # div = (PS[folder]["arrDatesU"][inds] - PS[folder]["arrDatesL"][inds]) / 2
        # div[div == 0] = 1
        # a /= div
        # a = 1. - np.sqrt(a)

        a = maxtransp / (PS[folder]["arrDatesU"][inds] - PS[folder]["arrDatesL"][inds] + 1e-20)
        a[a>1] = 1.

    cols = np.zeros((len(toPlot), 4))
    cols[:, 0] = 1.
    cols[:, 3] = a

    toPlot, cols = aggregatePoints(toPlot, cols)

    clus = []
    if len(toPlot)!=0 and ((age%5==0 and not PS[folder]["anim"]) or PS[folder]["anim"]):
        clus = getMetrics(folder, age, toPlot, cols[:, 3])

    return [age, toPlot, cols, clus]


# Plot tools
def quantile_to_level(data, quantile):
    """Return data levels corresponding to quantile cuts of mass."""
    isoprop = np.asarray(quantile)
    values = np.ravel(data)
    sorted_values = np.sort(values)[::-1]
    normalized_values = np.cumsum(sorted_values) / values.sum()
    idx = np.searchsorted(normalized_values, 1 - isoprop)
    levels = np.take(sorted_values, idx, mode="clip")
    return levels

def tohtml(long, lat, z):
    long, lat, z = np.round(long, 5), np.round(lat, 5), np.round(z, 5)
    code = f"""
        <table border="1" class="table">
          <thead>
            <tr style="text-align: right;" class="tr">
              <th class="th"></th>
              <th class="th">Inscriptions</th>
            </tr>
          </thead>
          <tbody>
            <tr class="tr">
              <th class="th">X</th>
              <td class="td">{long}</td>
            </tr>
            <tr class="tr">
              <th class="th">Y</th>
              <td class="td">{lat}</td>
            </tr>
            <tr class="tr">
              <th class="th">Qty</th>
              <td class="td">{z}</td>
            </tr>
          </tbody>
        </table>
    """
    return code

def connectToLabelsOnFly(folder, hist):
    css = """
    .table
    {
      border-collapse: collapse;
    }
    .th
    {
      color: #ffffff;
      background-color: #5d3700;
    }
    .td
    {
      background-color: #ecc882;
    }
    .td
    {
      background-color: #ecc882;
    }
    .table, .th, .td
    {
      font-family:Arial, Helvetica, sans-serif;
      border: 1px solid black;
      text-align: right;
    }
    """
    ax = PS[folder]["fig"].gca()

    bins, xedges, yedges, _ = hist

    edgeSqr = (PS[folder]["fig"].axes[0].get_xlim()[1]-PS[folder]["fig"].axes[0].get_xlim()[0])/PS[folder]["gridSize"]

    x = np.array([xedges[i] + edgeSqr / 2 for i in range(len(xedges) - 1)])
    y = np.array([yedges[i] + edgeSqr / 2 for i in range(len(yedges) - 1)])
    bins[np.isnan(bins)]=0
    nnz = bins.nonzero()
    N = len(nnz[0])

    dflong, dflat, dfz = x[nnz[0]], y[nnz[1]], bins[nnz]

    labels = []
    for i in range(N):
        labels.append(str(tohtml(dflong[i], dflat[i], dfz[i])))

    axmin, axmax = ax.get_xlim()
    s = (edgeSqr*1.0 * (ax.get_window_extent().width / (axmax - axmin) * 72. / PS[folder]["fig"].dpi)) ** 2

    points = ax.scatter(dflong, dflat, marker='s', color='k', s=s, alpha=.0, zorder=10)

    tooltip = plugins.PointHTMLTooltip(points, labels, voffset=10, hoffset=10, css=css)
    return tooltip


# Plots
def wipePlot(folder):
    for i in reversed(range(0, len(PS[folder]["fig"].axes[0].collections))):
        PS[folder]["fig"].axes[0].collections[i].remove()

def init(folder, data, typePlot, styleLoc, cities, bbUsr):
    style=styleLoc
    arrCoords, bounds = data[1], data[4]
    strCities=""
    if cities:
        strCities="Cities"

    minLon, minLat, maxLon, maxLat = bounds
    im = mpimg.imread(f'static/img/map_{style+strCities}.jpg')

    fig, ax = plt.subplots(figsize=(9, 6))
    shiftx, shifty = 95800, 94800  # Because white margin on the maps
    minLon, maxLon, minLat, maxLat = minLon-shiftx, maxLon+shiftx, minLat-shifty, maxLat+shifty

    ax.imshow(im, extent = [minLon, maxLon, minLat, maxLat])
    ax.set_xlim([minLon+shiftx, maxLon-shiftx])
    ax.set_ylim([minLat+shifty, maxLat-shifty])

    ax.set_xlim([bbUsr[0], bbUsr[2]])
    ax.set_ylim([bbUsr[1], bbUsr[3]])


    a = np.array([[0, 100]])
    img = ax.imshow(a, cmap="afmhot_r")
    img.set_visible(False)
    cb=plt.colorbar(img)
    if typePlot==["points"]:
        fig.axes[1].cla()
    fig.axes[1].set_visible(False)


    fig.tight_layout()

    pos = fig.axes[1].get_position()
    pos = [pos.x0, fig.axes[0].get_position().y0,  pos.width, fig.axes[0].get_position().height]
    fig.axes[1].set_position(pos)

    if typePlot==["points"]:
        fig.axes[1].axis('off')
    else:
        fig.axes[1].axis("on")

    fig.canvas.draw()

    bg = [fig.canvas.copy_from_bbox(fig.axes[0].bbox.expanded(10,2)), fig.canvas.copy_from_bbox(fig.axes[1].bbox.expanded(1.,1.))]
    fig.axes[1].set_visible(True)
    fig.canvas.draw()

    return fig, cb, bg

def plotFrameHist2d(folder, age, toPlot, cols, clus):
    bb = [[PS[folder]["bounds"][0], PS[folder]["bounds"][2]], [PS[folder]["bounds"][1], PS[folder]["bounds"][3]]]
    fac = (bb[1][1]-bb[1][0])/(bb[0][1]-bb[0][0])
    binx, biny = PS[folder]["gridSize"], int(PS[folder]["gridSize"]*fac)#0.693798712)

    if PS[folder]["fixedvmax"]: cmax=PS[folder]["vmax"]
    else: cmax=None

    cmap = pl.cm.afmhot_r
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:,-1] = np.linspace(0, 1, cmap.N)**0.5 # Set alpha
    my_cmap = ListedColormap(my_cmap)

    #hist = PS[folder]["fig"].axes[0].hist2d(toPlot[:, 0], toPlot[:, 1], weights=cols[:, 3], bins=[binx, biny], cmap="afmhot_r", alpha=0.8, cmin=0.005, vmax=cmax, range=bb)
    hist = PS[folder]["fig"].axes[0].hist2d(toPlot[:, 0], toPlot[:, 1], weights=cols[:, 3], bins=[binx, biny], cmap=my_cmap, cmin=0.005, vmax=cmax, range=bb)

    if not PS[folder]["anim"]:
        PS[folder]["cb"].set_label(f"Population (total: {round(np.sum(cols[:, 3]), 2)})", rotation=270, size=25, labelpad=25)
    else:
        PS[folder]["cb"].set_label(f"Population (total: {round(np.sum(cols[:, 3]), 2)})", rotation=270, size=15, labelpad=25)
    if cmax is None: m = np.nanmax(hist[0])
    else: m = cmax
    PS[folder]["cb"].set_ticks([c*100/m for c in np.linspace(0, m, 6)])  # cbar always has 100 height here.
    PS[folder]["cb"].set_ticklabels([str(np.round(c, 3)) for c in np.linspace(0, m, 6)])

    if not PS[folder]["anim"]:
        plugins.clear(PS[folder]["fig"])
        tooltip = connectToLabelsOnFly(folder, hist)
        plugins.connect(PS[folder]["fig"], plugins.Reset(), plugins.BoxZoom(), plugins.Zoom(enabled=True), tooltip)

def plotFramePoints(folder, age, toPlot, cols, clus):

    colorPoints = copy(cols)
    transpmax = 1.
    if PS[folder]["fixedvmax"]:
        transpmax = PS[folder]["vmax"]
    colorPoints[:, 3] = colorPoints[:, 3]/transpmax
    cols[:, 3] = cols[:, 3]/transpmax

    colorPoints[cols[:, 3] > 1, 3] = 1.  # Si plusieurs tombes à un endroit
    cols[cols[:, 3] > 1, 3] = 1.

    if not PS[folder]["anim"]:
        plugins.clear(PS[folder]["fig"])
        plugins.connect(PS[folder]["fig"], plugins.Reset(), plugins.BoxZoom(), plugins.Zoom(enabled=True))

    if not PS[folder]["plotClus"]:
        PS[folder]["fig"].axes[0].scatter(toPlot[:, 0], toPlot[:, 1], c=cols, s=PS[folder]["sizeScatter"])
    else:
        for numCol, c in enumerate(set(clus)):
            rgb=matplotlib.colors.to_rgb(cyclecolors[numCol])
            colorPoints[:, 0] = rgb[0]
            colorPoints[:, 1] = rgb[1]
            colorPoints[:, 2] = rgb[2]
            PS[folder]["fig"].axes[0].scatter(toPlot[:, 0][clus == c], toPlot[:, 1][clus == c], s=PS[folder]["sizeScatter"], c=colorPoints[clus==c])

    if PS[folder]["typePlot"] == ["points"]:
        PS[folder]["cb"].set_ticks([0, 0.0001])
        PS[folder]["cb"].set_ticklabels(["", ""])

def plotFrameKde(folder, age, toPlot, cols, clus):
    if len(toPlot)<=2:
        return -1



    if "hist2d" not in PS[folder]["typePlot"]: cbarBool = True
    else: cbarBool=False

    cmax=None
    if PS[folder]["fixedvmax"]: cmax=PS[folder]["vmax"]

    scipyKde = gaussian_kde([toPlot[:, 0], toPlot[:, 1]], weights=cols[:, 3], bw_method=((PS[folder]["bounds"][2]-PS[folder]["bounds"][0])/(1000000*PS[folder]["gridSize"])))
    x = np.linspace(PS[folder]["bounds"][0], PS[folder]["bounds"][2], 100)
    y = np.linspace(PS[folder]["bounds"][1], PS[folder]["bounds"][3], 100)
    X, Y = np.meshgrid(x, y)
    grid_coords = np.append(X.reshape(-1, 1), Y.reshape(-1, 1), axis=1)
    Z = scipyKde(grid_coords.T)  # Ca prend du temps.
    Z = Z.reshape(len(X),len(Y))

    common_levels = quantile_to_level(Z, np.array(list(range(11)))/10)[1:]


    PS[folder]["fig"].axes[0].contour(X, Y, Z, levels=common_levels, cmap="afmhot_r", vmin=0, vmax=cmax)

    if cbarBool:
        if not PS[folder]["anim"]:
            PS[folder]["cb"].set_label(f"Density estimation (population: {round(np.sum(cols[:, 3]), 2)})", size=25, labelpad=30, rotation=270)
        else:
            PS[folder]["cb"].set_label(f"Density estimation (population: {round(np.sum(cols[:, 3]), 2)})", size=15, labelpad=30, rotation=270)
        if cmax is not None: m = cmax
        else: m = np.max(common_levels)
        PS[folder]["cb"].set_ticks([c*100/m for c in common_levels])
        PS[folder]["cb"].set_ticklabels(["{:.1e}".format(c) for c in common_levels])

    PS[folder]["fig"].axes[0].set_xlim([PS[folder]["bounds"][0], PS[folder]["bounds"][2]])
    PS[folder]["fig"].axes[0].set_ylim([PS[folder]["bounds"][1], PS[folder]["bounds"][3]])

    if not PS[folder]["anim"]:
        plugins.clear(PS[folder]["fig"])
        plugins.connect(PS[folder]["fig"], plugins.Reset(), plugins.BoxZoom(), plugins.Zoom(enabled=True))

def getFrame(age=None, folder="", uAge=None, toPlot=None, cols=None, clus=None):

    with open(folder+"/stop.txt", "r") as f:
        stop = int(f.read())
        if stop == 1:
            print(PS[folder]["filter"], age)
            return

    donotredoframe = False
    if toPlot is None:
        age, toPlot, cols, clus = treatDataAge(folder, age)
        if len(PS[folder]["prevData"]) == len(toPlot):
            if np.allclose(PS[folder]["prevData"], toPlot):
                donotredoframe = True
        PS[folder]["prevData"] = toPlot
        strTitle="Year " + str(int(age))
    else:
        if PS[folder]["noDates"] == True:
            strTitle = "All years"
        else:
            strTitle = "Years " + str(int(age)) +" to " + str(int(uAge))

    print(age, PS[folder]["filter"], len(toPlot))

    if PS[folder]["filterStatus"] is not None:
        strTitle="Status : " + str(PS[folder]["filterStatus"]) + " - " + strTitle
    if PS[folder]["filter"] is not None:
        strTitle="Filter : " + str(PS[folder]["filter"]) + " - " + strTitle
    PS[folder]["fig"].axes[0].set_title(strTitle)

    if not donotredoframe:
        wipePlot(folder)

        if "points" in PS[folder]["typePlot"]:
            try:
                plotFramePoints(folder, age, toPlot, cols, clus)
            except Exception as e:
                print("Points - ", e)
                pass
        if "kde" in PS[folder]["typePlot"]:
            try:
                plotFrameKde(folder, age, toPlot, cols, clus)
            except Exception as e:
                print("Kde - ", e)
                pass
        if "hist2d" in PS[folder]["typePlot"]:
            try:
                plotFrameHist2d(folder, age, toPlot, cols, clus)
            except Exception as e:
                print("Hist - ", e)
                pass

    with open(f"{folder}/prog.txt", "w+") as f:
        f.write(str(age)+"\t"+str(PS[folder]["lage"])+"\t"+str(PS[folder]["uage"]))


# Requests

def getAnim(folder, filename, lAge=None, uAge=None):
    minAge, maxAge = min(PS[folder]["arrDatesL"]), max(PS[folder]["arrDatesU"])
    if lAge is not None and uAge is not None: ageBounds = [lAge, uAge]
    else: ageBounds = [minAge, maxAge]

    ages = list(range(ageBounds[0], ageBounds[1]))
    ani = FasterFuncAnimation(PS[folder]["fig"], getFrame, PS[folder]["bg"], fargs=(folder,), frames=ages, repeat=False, blit=False, interval=0.001)


    ani.save(filename, writer=PS[folder]["writervideo"])

    return ani

def getMeanAges(folder, lAge, uAge):
    rangeAge = uAge - lAge
    toPlotTot = [[-1, -1]]
    colsTot = [[0, 0, 0, 0]]
    for age in range(lAge, lAge+rangeAge):
        age, toPlot, cols, clus = treatDataAge(folder, age)
        toPlotTot += list(toPlot)
        colsTot += list(cols)

        with open(f"{folder}/prog.txt", "w+") as f:
            f.write(str(age) + "\t" + str(lAge) + "\t" + str(uAge))

    toPlotTot, colsTot = np.array(toPlotTot), np.array(colsTot)
    toPlotTot, colsTot = aggregatePoints(toPlotTot, colsTot)

    colsTot[:, 3] /= rangeAge

    clusTot = getMetrics(folder, 0, toPlotTot, colsTot[:, 3], writeRes=False)

    getFrame(lAge, folder, uAge, toPlotTot, colsTot, clusTot)



def run(data, fig, cb, gridSize=30, lage=-50, uage=1000, filter=None, anim=True, noDates=False, weighted=True,
        typePlot=["points"], fps=10, fixedvmax=False, vmax=100, style="lines",
        sizeScatter=20, folder="./", imageOnly=False, filterOperator="or", filterStatus=None, filterStatusOperator="or",
        plotClus=False, radClus=100, bg=None, bbUsr=None):
    arrDatesMean, arrCoords, arrDatesL, arrDatesU, bounds, figHistWords, figHistStat, txtDS = data
    writervideo = FasterFFMpegWriter(fps=fps)
    bounds = bbUsr

    tabVars = {"fig":fig, "bounds":bounds, "weighted":weighted, "gridSize":gridSize, "arrDatesU":arrDatesU, "arrDatesL":arrDatesL,
               "arrDatesMean":arrDatesMean, "filter":filter, "anim":anim, "noDates":noDates, "typePlot":typePlot, "writervideo":writervideo, "vmax":vmax,
               "fixedvmax":fixedvmax, "style":style, "lage":lage, "uage":uage, "sizeScatter":sizeScatter, "arrCoords":arrCoords,
               "filterOperator":filterOperator, "filterStatus":filterStatus, "filterStatusOperator":filterStatusOperator,
               "plotClus":plotClus, "radClus":radClus, "cb":cb, "metrics":{}, "bg":bg, "prevData":np.array([])}

    global PS
    PS[folder]=tabVars

    if not PS[folder]["anim"]:
        if len(PS[folder]["arrDatesMean"])==0:
            return f"{folder}/NoFigToShow", "No inscription found.", "No metrics computed."
        getMeanAges(folder, lage, uage)

        now = datetime.datetime.now()
        hhmmss = "%s-%s-%s" % (now.hour, now.minute, now.second)
        filename = f"{folder}/Pic_{hhmmss}.jpg".replace(":", "-")
        w, h = PS[folder]["fig"].get_size_inches()*PS[folder]["fig"].dpi
        rapp = h/w
        html=""
        if not imageOnly:
            PS[folder]["fig"].delaxes(PS[folder]["fig"].axes[1])
            html = mpld3.fig_to_html(PS[folder]["fig"])
            html = html.replace(f"\"width\": {round(w, 1)}", "\"width\": document.getElementById(\"colMap\").clientWidth" )
            html = html.replace(f"\"height\": {round(h, 1)}", f"\"height\": document.getElementById(\"colMap\").clientWidth*{rapp}")
            with open(f"{folder}/out.html", "w+") as f: f.write(html)
        PS[folder]["fig"].set_size_inches(15, 15*5/8)

        PS[folder]["fig"].savefig(filename, dpi=300)
        PS[folder]["fig"].savefig(filename.replace(".jpg", ".pdf"))
        htmlMetrics = plotMetrics(folder)
        return filename, html, htmlMetrics

    else:
        if len(PS[folder]["arrDatesMean"])==0:
            return f"{folder}/Pic.jpg", "No inscription found.", "No metrics computed."



        now = datetime.datetime.now()
        hhmmss = "%s-%s-%s" % (now.hour, now.minute, now.second)
        filename = f'{folder}/Ages_{hhmmss}.mp4'

        ani = getAnim(folder, filename, lAge=lage, uAge=uage)

        htmlMetrics = plotMetrics(folder)
        return filename, ani, htmlMetrics


if __name__ == "__main__":
    import pprofile

    profiler = pprofile.Profile()
    with profiler:
        data = getData(["Iulius"], "or", [''], "or")
        fig, cb = init(folder, data, "linesFilled", cities=False)
        filename, out, htmlMetrics = run(data, fig, cb,
                                    gridSize=30,
                                    lage=-100,
                                    uage=-85,
                                    filter=["Iulius"],
                                    anim=True,
                                    weighted=True,
                                    typePlot=["points", "kde",],
                                    fixedvmax=False,
                                    vmax=5,
                                    fps=5,
                                    style="linesFilled",
                                    sizeScatter=10,
                                    folder=f"./temp",
                                    imageOnly=True,
                                    filterOperator="or",
                                    filterStatus=[''],
                                    filterStatusOperator="or",
                                    plotClus=True,
                                    radClus=200
                                    )

    profiler.dump_stats("temp/Benchmark.txt")




