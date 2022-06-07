import numpy as np
import matplotlib.pyplot as plt
import html2text

rome = True
approxDate = True
noDates = True

def initMaps():
    from mpl_toolkits.basemap import Basemap
    import pandas as pd
    bounds = np.array([-10.424482, 25.728333, 49.817223, 57.523959])
    minLon, minLat, maxLon, maxLat = bounds

    m = Basemap(projection="merc", llcrnrlon=minLon, llcrnrlat=minLat, urcrnrlon=maxLon, urcrnrlat=maxLat, resolution="c", area_thresh=2000.)

    def getco(arr):
        arr = eval(arr)
        y, x = m(arr[1], arr[0])
        return [x,y]

    name = "data"
    df = pd.read_csv(name+".csv", sep="\t")
    dfTemp = df["coords"].apply(getco)
    df2=df.copy()
    df2["coords"]=dfTemp
    df2.to_csv(name+".csv", sep="\t", index=False)
    pause()

    style="bluemarbleCities"
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.savefig(f"static/img/map_{style}.jpg", bbox_inches='tight', dpi=600)
    plt.close()


# initMaps()
# pause()
dic = {}
kw = ["dating", "EDCS-ID", "province", "**to**", "**place:**", "**lieu:**", "**inscription genus / personal status:**"]
numErrInsc = 0
num = -1
for numDoc in [1, 2, 3, 4, 5]:
    print("Doc", numDoc)
    f = open(f"AggregatedRegions/{numDoc}.html", "r", encoding="utf-8")
    txt = f.read()#[:10000000]
    tombs = html2text.html2text(txt).split("\n\n")[11:]
    for t in tombs:
        num += 1
        d = {"dating":np.nan, "EDCS-ID":np.nan, "province":np.nan, "**to**":np.nan, "place":np.nan, "inscription":np.nan, "status":np.nan}
        tab = t.split("  ")
        d["text"] = t

        for num2, i in enumerate(tab):
            if len(i) == 0: continue
            #print("-", i)
            for k in kw:
                if k in i:
                    if k == "**to**":
                        d[k] = i[i.find(k) + len(k):].replace(" ", "")
                    elif k == "**place:**":
                        d["place"] = i[i.find(k) + len(k):].replace(" ", "")
                        #print(d["place"])
                        try:
                            d["inscription"] = tab[num2+1].replace("\n", " ")
                        except Exception as e:
                            print("ERREUUUUUUUUUUUUUUR 1", e)
                            numErrInsc+=1
                            pass
                    elif k == "**inscription genus / personal status:**":
                        indStatus=1
                        d["status"]=""
                        while True:
                            try:
                                d["status"] += " "
                                for stat in tab[num2+indStatus].split(";"):
                                    d["status"]+=stat.replace("\n", " ")+" "
                                if tab[num2+indStatus][-1]!=";":
                                    break
                                indStatus+=1
                            except Exception as e:
                                print("ERREUUUUUUUUUUUUUUR 2", e)
                                numErrInsc+=1
                                break
                                pass
                    else:
                        d[k] = i[i.find("**" + k + ":**") + len(k) + 5:].replace(" ", "")

        if len(d)!=0:
            dic[num] = d
        if d["dating"] is not np.nan:
            print(d["dating"])

print("NOMBRE ERREURS :", numErrInsc)
print()
print()
print
print()
#pause()

import re
def treatText(s):
    s = s.replace("]", "")
    s = s.replace("[", "")
    s = s.replace("(", "")
    s = s.replace(")", "")
    s = s.replace("{", "")
    s = s.replace("}", "")
    s = s.replace("?", "")
    for i in range(10):
        s = s.replace(str(i), "")

    s = re.sub('=.+?>', '', s)
    s = re.sub("(?<=\w)/(?=\w)", "", s)
    s = s.replace("<", "")
    s = s.replace(">", "")

    s = s.replace("   ", " ")
    s = s.replace("  ", " ")
    s = s.lower()
    s = s.replace("augustiaugustae", "augusti augustae")
    s = s.replace("officiumprofessio", "officium professio")
    s = s.replace("libertilibertae", "liberti libertae")
    s = s.replace("serviservae", "servi servae")
    if len(s) != 0:
        if s[0] != " ":
            s = " " + s

    return s

numErrStruct = 0
tabierr, tabnonierr = [], []
dicPropre = {}
kwPropres = ["EDCS", "dates", "province", "coords", "inscription", "status"]
nbTombesDates, nbTombesCoords, nbTombesTot = 0, 0, 0
for i in dic:
    dicPropre[i] = {}
    err = False
    nbTombesTot += 1
    try:
        dicPropre[i]["EDCS"] = dic[i]["EDCS-ID"].split("-")[1]
    except Exception as e:
        print("EDCS-ID", e, dic[i]["EDCS-ID"])
        err = True
        dicPropre[i]["EDCS"] = np.nan
    try:
        if dic[i]["dating"] is not np.nan:
            dicPropre[i]["dates"] = [int(dic[i]["dating"]), int(dic[i]["**to**"])]
            nbTombesDates+=1
            if int(dic[i]["dating"]) > int(dic[i]["**to**"]):
                provokeErr()
        elif approxDate and not noDate:
            treatedtxt = treatText(dic[i]["inscription"])
            if "hse" in treatedtxt or "hic situs est" in treatedtxt:
                dicPropre[i]["dates"] = [-200, 50]
                nbTombesDates+=1
            elif "dis manibus et memoriae aeternae" in treatedtxt:
                dicPropre[i]["dates"] = [200, 300]
                nbTombesDates+=1
            elif "dis manibus" in treatedtxt:
                dicPropre[i]["dates"] = [70, 200]
                nbTombesDates+=1
            else:
                provokeErr()
        elif noDates:
            treatedtxt = treatText(dic[i]["inscription"])
            dicPropre[i]["dates"] = [-1000, 1000]
        else:
            provokeErr()

    except Exception as e:
        print("dating", e, dic[i]["EDCS-ID"], dic[i]["dating"])
        err = True
        dicPropre[i]["dates"] = [np.nan, np.nan]
    try:
        if dic[i]["province"][0] == " ": dic[i]["province"] = dic[i]["province"][1:]
        dicPropre[i]["province"] = dic[i]["province"]
    except Exception as e:
        print("province", e, dic[i]["province"])
        err = True
        dicPropre[i]["province"] = np.nan
    try:
        if dic[i]["place"] is np.nan:
            provokeErr()
        elif dic[i]["place"] == "?":
            provokeErr()
        else:
            ilat = dic[i]["place"].find("latitude=") + len("latitude=")
            ilon = dic[i]["place"].find("longitude=") + len("longitude=")
            lat = dic[i]["place"][ilat:ilat + 9].replace("&", "")
            lon = dic[i]["place"][ilon:ilon + 9].replace("&", "")
            lat, lon = float(lat), float(lon)
            dicPropre[i]["coords"] = [float(lat), float(lon)]
            nbTombesCoords += 1

            if not rome and (lat>41.464135 and lat<42.196783 and lon>11.976857 and lon < 12.905822):
                provokeErr()

    except Exception as e:
        print("place", e, dic[i]["place"])
        err = True
        dicPropre[i]["coords"] = [np.nan, np.nan]
    try:
        dicPropre[i]["inscription"] = treatText(dic[i]["inscription"])
    except Exception as e:
        print("inscription", e, dic[i]["inscription"])
        err = True
        dicPropre[i]["inscription"] = np.nan
    try:
        if dic[i]["status"] is not np.nan:
            dicPropre[i]["status"] = treatText(dic[i]["status"])
        else:
            dicPropre[i]["status"] = str(0)
    except Exception as e:
        print("status", e, dic[i]["status"])
        err = True
        dicPropre[i]["status"] = np.nan

    if err:
        numErrStruct += 1
        tabierr.append(i)
        print(i)
    else:
        tabnonierr.append(i)

    # print(dicPropre[i])


fn = "data"
if not rome:
    fn+="_ssRome"
if approxDate and not noDates:
    fn += "_approx"
elif noDates:
    fn += "_noDates"
with open(fn+".csv", "w+", encoding="utf-8") as f:
    f.write(kwPropres[0])
    for key in kwPropres[1:]:
        f.write("\t"+key)
    f.write("\n")
    for i in tabnonierr:
        f.write(dicPropre[i][kwPropres[0]])
        for key in kwPropres[1:]:
            f.write("\t" + str(dicPropre[i][key]))
        f.write("\n")

with open(fn+"Err.csv", "w+", encoding="utf-8") as f:
    for i in tabierr:
        f.write(str(i)+"\t"+str(dic[i]["text"].replace("\n", "   "))+"\n")


for i in tabnonierr:
    pass
    #print(i, dicPropre[i])#["province"])

print("NB err inscription", numErrInsc)
print("NB err struct", numErrStruct)
print("# tombes datées", nbTombesDates)
print("# tombes placées", nbTombesCoords)
print("# tombes tot", nbTombesTot)
print("NB TOMBES JUSTES", len(tabnonierr))
print("NB TOMBES FAUSSES", len(tabierr))



