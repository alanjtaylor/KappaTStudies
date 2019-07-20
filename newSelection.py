import ROOT
import pandas
import uproot
from matplotlib import pyplot as plt
import matplotlib as mpl
import math
import numpy as np
import inspect
from optparse import OptionParser, OptionGroup, OptionValueError
import argparse


def computeSignificance(signal, background):

    Z = math.sqrt( 2.0*( (signal+background)*math.log(1 + signal/background) - signal )   )
    return Z


def getHisto(aDataframe, **kwargs):

    normalised = False
    if 'normed' in kwargs.keys() :
        normalised = kwargs.pop('normed')

    label_str = ""
    if 'label' in kwargs.keys():
        label_str = kwargs.pop('label')

    ## variables needed for the histogram
    histkwargs = {}
    for key, value in kwargs.items():
        if key in inspect.getfullargspec(np.histogram).args:
            histkwargs[key] = value

    histvals, binedges = np.histogram(aDataframe, **histkwargs )
    yerrs = np.sqrt(histvals)

    if normalised:
        nevents = float(sum(histvals))
        binwidth = (binedges[1]-binedges[0])
        histvals = histvals/nevents/binwidth
        yerrs = yerrs/nevents/binwidth

    bincenters = (binedges[1:]+binedges[:-1])/2

    ## variables needed for matplotlib
    ebkwargs = {}
    for key, value in kwargs.items() :
        #if key in inspect.getargspec(plt.errorbar).args :
        if key in inspect.getfullargspec(plt.errorbar).args :
            ebkwargs[key] = value

    out = plt.errorbar(bincenters, histvals, yerrs, **ebkwargs, label=label_str)
    return out

def doCosmetics():


    mpl.rcParams['lines.linewidth'] = 1
    mpl.rcParams['figure.figsize'] = 8.75, 5.92

    # not very useful
    mpl.rcParams['figure.facecolor'] = 'white'
    mpl.rcParams['figure.subplot.bottom'] = 0.16
    mpl.rcParams['figure.subplot.top'] = 0.95
    mpl.rcParams['figure.subplot.left'] = 0.16
    mpl.rcParams['figure.subplot.right'] = 0.95

    # font
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['mathtext.fontset'] = 'stixsans'
    mpl.rcParams['mathtext.default'] = 'rm'
    # helvetica usually not present on linux
    #mpl.rcParams['font.sans-serif'] = 'helvetica, Helvetica, Nimbus Sans L, Mukti Narrow, FreeSans'


    mpl.rcParams['axes.labelsize'] = 20
    mpl.rcParams['xtick.labelsize'] = 19
    mpl.rcParams['ytick.labelsize'] = 19

    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['xtick.major.size'] = 12
    mpl.rcParams['xtick.minor.size'] = 6
    mpl.rcParams['ytick.major.size'] = 14
    mpl.rcParams['ytick.minor.size'] = 7

    mpl.rcParams['xtick.top'] = True
    mpl.rcParams['ytick.right'] = True

    mpl.rcParams['xtick.minor.visible'] = True
    mpl.rcParams['ytick.minor.visible'] = True

    mpl.rcParams['legend.frameon'] = False

parser = argparse.ArgumentParser()

parser.add_argument('-p', '--priority', action='store', default=True, help='Give priority to tH or ttH in the optimisation')
parser.add_argument('-nlept', '--nleptons', action='store', default=1, help='Number of leptons used in tH study')
parser.add_argument('-njets', '--njets', action='store', default=2, help='Number of central jets')
parser.add_argument('-scan', '--scan', action='store', default=False, help='Choose whether to do the scan over the eta max variable')
parser.add_argument('-f', '--fit', action='store', help='dummy')
parser.add_argument('-s', '--selection', action='store', help='dummy')

#parser.parse_known_args(['--priority', '--nleptons', '--njets', '--scan'])
args = parser.parse_args()

## options for the code
## decides whether to give priority to the tH categories or not
print('tH priority status:', args.priority)
print('number of leptons:', args.nleptons)
print('number of jets:', args.njets)
print('do the scan:', args.scan)

prioritytH = args.priority
# accept events with number of leptons equal to the below 
nLeptons = int(args.nleptons)
# accept events with a number of central jets equal to or less than the below 
nCenJets = int(args.njets)
# whether to do an optimisation scan of max |eta|
doScan = args.scan

#prioritytH = True 
#nLeptons = 1 
#nCenJets = 2 
#doScan = True

SAMPLES = ["ggH", "VBF", "ZH", "WH", "ttH", "tHjb", "tWH", "ggZH", "bbH"]

# lists for pandas dataframes, the yields in each category and the purities
frames = [ ]
yieldsPerCategory = [ ]
puritiesPerCategory = [ ]
efficienciesPerCategory = [ ] 

# get the data and turn into dataframe
data_tree = uproot.open("skimmedFiles/data_total.root")["CollectionTree"]
dataDF = data_tree.pandas.df(flatten=None)

# loop over higgs samples, creating dataframes and lists
for sample in SAMPLES:
    
    tree = uproot.open("skimmedFiles/"+sample+"_total.root")["CollectionTree"]
    DF = tree.pandas.df(flatten=None)
    
    yields = [ ]
    purity = [ ]
    effs = [ ] 

    frames.append(DF)
    yieldsPerCategory.append(yields)
    puritiesPerCategory.append(purity)
    efficienciesPerCategory.append(effs)

# create dicts of data frames and the lists
dictDF = dict(zip(SAMPLES, frames))
dictYields = dict(zip(SAMPLES,yieldsPerCategory))
dictPurities = dict(zip(SAMPLES,puritiesPerCategory))
dictEfficiencies = dict(zip(SAMPLES,efficienciesPerCategory))

totalHiggsYields = [ ]
dataYields = [ ]

## ATLAS Style numpy cosmetics 
doCosmetics()

## inspect some variables for the leptonic categories
fig, ax = plt.subplots()

bin_jets = [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5]
dataHisto = getHisto(dataDF[(dataDF["nLep"] == 1)]["n_j_central"],normed=True, fmt='ko', label='Data', bins=bin_jets)
tHjbHisto = getHisto(dictDF["tHjb"][(dictDF["tHjb"]["nLep"] == 1)]["n_j_central"],normed= True, fmt='ro', label='tHjb', bins=bin_jets)
ttHHisto = getHisto(dictDF["ttH"][(dictDF["ttH"]["nLep"] == 1)]["n_j_central"],normed= True, fmt='bo', label='ttH', bins=bin_jets)

plt.legend(bbox_to_anchor=(0.65, 0.7, 0.2, 0.2), fontsize=18, labelspacing=0.8, handletextpad=0., frameon=False)

ax.set_xlabel('Number of central jets', horizontalalignment='right', x=1.0)
ax.set_ylabel(r'Fraction of Events', horizontalalignment='right', y=1.0)

plt.savefig('outputPlots/n_j_centralLept.png')

## clear the figure 
plt.cla()

## plot the jet which has the maximum |eta| in the event. 

dataHisto = getHisto(dataDF[(dataDF["nLep"] == 1) & (dataDF["n_j_central"] <=3 )]["maxEta"],normed=True, fmt='ko', label='Data', bins=np.linspace(0.0, 5.0, 20))
tHjbHisto = getHisto(dictDF["tHjb"][(dictDF["tHjb"]["nLep"] == 1) & (dictDF["tHjb"]["n_j_central"] <=3 )]["maxEta"],normed= True, fmt='ro', label='tHjb', bins=np.linspace(0.0, 5.0, 20))
ttHHisto = getHisto(dictDF["ttH"][(dictDF["ttH"]["nLep"] == 1) & (dictDF["ttH"]["n_j_central"] <=3 )]["maxEta"],normed= True, fmt='bo', label='ttH', bins=np.linspace(0.0, 5.0, 20))

plt.legend(bbox_to_anchor=(0.65, 0.7, 0.2, 0.2), fontsize=18, labelspacing=0.8, handletextpad=0., frameon=False)

ax.set_xlabel('$|\eta^{max}|$', horizontalalignment='right', x=1.0)
ax.set_ylabel(r'Fraction of Events / 0.25', horizontalalignment='right', y=1.0)
plt.savefig('outputPlots/maxEtaLept.png')

plt.cla()

#######
#######
## Do the same thing now but for the hadronic case

bin_jets = [2.5,3.5,4.5,5.5,6.5,7.5,8.0,8.5]
dataHisto = getHisto(dataDF[(dataDF["nLep"] == 0)]["n_j_central"],normed=True, fmt='ko', label='Data', bins=bin_jets)
tHjbHisto = getHisto(dictDF["tHjb"][(dictDF["tHjb"]["nLep"] == 0)]["n_j_central"],normed= True, fmt='ro', label='tHjb', bins=bin_jets)
ttHHisto = getHisto(dictDF["ttH"][(dictDF["ttH"]["nLep"] == 0)]["n_j_central"],normed= True, fmt='bo', label='ttH', bins=bin_jets)

plt.legend(bbox_to_anchor=(0.65, 0.7, 0.2, 0.2), fontsize=18, labelspacing=0.8, handletextpad=0., frameon=False)

ax.set_xlabel('Number of central jets', horizontalalignment='right', x=1.0)
ax.set_ylabel(r'Fraction of Events', horizontalalignment='right', y=1.0)

plt.savefig('outputPlots/n_j_centralHad.png')

## clear the figure 
plt.cla()

## plot the jet which has the maximum |eta| in the event.

dataHisto = getHisto(dataDF[(dataDF["nLep"] == 0)]["maxEta"],normed=True, fmt='ko', label='Data', bins=np.linspace(0.0, 5.0, 20))
tHjbHisto = getHisto(dictDF["tHjb"][(dictDF["tHjb"]["nLep"] == 0)]["maxEta"],normed= True, fmt='ro', label='tHjb', bins=np.linspace(0.0, 5.0, 20))
ttHHisto = getHisto(dictDF["ttH"][(dictDF["ttH"]["nLep"] == 0)]["maxEta"],normed= True, fmt='bo', label='ttH', bins=np.linspace(0.0, 5.0, 20))

plt.legend(bbox_to_anchor=(0.65, 0.7, 0.2, 0.2), fontsize=18, labelspacing=0.8, handletextpad=0., frameon=False)

ax.set_xlabel('$|\eta^{max}|$', horizontalalignment='right', x=1.0)
ax.set_ylabel(r'Fraction of Events / 0.25', horizontalalignment='right', y=1.0)
plt.savefig('outputPlots/maxEtaHad.png')

plt.cla()

## compute the nominal sensitivity
Z_ttH_Cats = [ ]
Z_tH_Cats = [ ] 
Z_ttH_LeptonicCats = [ ]
Z_tH_LeptonicCats = [ ] 
Z_ttH_HadronicCats = [ ]
Z_tH_HadronicCats = [ ] 

for catNo in range(27,34):

    bkgHiggsYield = 0.0

    dataYield = dataDF[dataDF['xgboostCat'] == catNo]['totalWeight'].sum()*(10.0/45.0)
    tHjbYield = dictDF["tHjb"][dictDF["tHjb"]['xgboostCat'] == catNo]['totalWeight'].sum()
    
    tWHYield = dictDF["tWH"][dictDF["tWH"]['xgboostCat'] == catNo]['totalWeight'].sum()
    ttHYield = dictDF["ttH"][dictDF["ttH"]['xgboostCat'] == catNo]['totalWeight'].sum()
    ## compute the Single Higgs background
    for sample in SAMPLES:
        if sample != "tHjb" and sample != "ttH" and sample != "tWH":
            bkgHiggsYield += dictDF[sample][dictDF[sample]['xgboostCat'] == catNo]['totalWeight'].sum()

    Z_ttH_Cats.append( computeSignificance(ttHYield, tWHYield + tHjbYield + dataYield + bkgHiggsYield) )
    Z_tH_Cats.append( computeSignificance(tWHYield + tHjbYield, ttHYield + dataYield + bkgHiggsYield) )
    
    if catNo < 31:
        Z_ttH_HadronicCats.append( computeSignificance(ttHYield, tWHYield + tHjbYield + dataYield + bkgHiggsYield) )
        Z_tH_HadronicCats.append( computeSignificance(tWHYield + tHjbYield, ttHYield + dataYield + bkgHiggsYield) )
    else:
        Z_ttH_LeptonicCats.append( computeSignificance(ttHYield, tWHYield + tHjbYield + dataYield + bkgHiggsYield) )
        Z_tH_LeptonicCats.append( computeSignificance(tWHYield + tHjbYield, ttHYield + dataYield + bkgHiggsYield) )

print('nominal leptonic significances :: ')
## ttH leptonic significance 
print('ttH lept significance: ' , round(math.sqrt(sum(Z*Z for Z in Z_ttH_LeptonicCats)),3))
## tH leptonic significance 
print('tH lept significance: ' , round(math.sqrt(sum(Z*Z for Z in Z_tH_LeptonicCats)),3))
print('nominal hadronic significances!')
## ttH hadronic significance 
print('ttH hadronic significance : ' , round(math.sqrt(sum(Z*Z for Z in Z_ttH_HadronicCats)),3))
## tH hadronic significance 
print('tH hadronic significance : ' , round(math.sqrt(sum(Z*Z for Z in Z_tH_HadronicCats)),3))

## create lists to create a tgraph of significance vs eta cut
maxEtaSelection = [ ]
tHZs = [ ]
ttHZs = [ ] 

maxEtaVal = 0.0

## scan over the maximum |eta| of a jet in the event.

if doScan:
    while maxEtaVal < 4.5:
        bkgHiggsYield = 0.0 
        for sample in SAMPLES:

            ## give priority to tH or not 
            if prioritytH:
                selection = (dictDF[sample]["nLep"] == nLeptons) & (dictDF[sample]["n_j_central"] <= nCenJets ) & (dictDF[sample]["maxEta"] >= maxEtaVal )
                data_selection = (dataDF["nLep"] == nLeptons) & (dataDF["n_j_central"] <= nCenJets ) & (dataDF["maxEta"] >= maxEtaVal )
            else:
                # if 0 leptons and we want to give priority to XGBoost categories, we need to avoid picking events with xgboost >= 27
                if nLeptons == 0:
                    selection = (dictDF[sample]["nLep"] == nLeptons) & (dictDF[sample]["n_j_central"] <= nCenJets ) & (dictDF[sample]["maxEta"] >= maxEtaVal ) & (dictDF[sample]["xgboostCat"] < 27 )
                    data_selection =  (dataDF["nLep"] == nLeptons) & (dataDF["n_j_central"] <= nCenJets ) & (dataDF["maxEta"] >= maxEtaVal ) & (dataDF["xgboostCat"] < 27 )
                # if >= 1 lepton and we want to give priority to XGBoost, we need to avoid picking events with xgboostcat >= 31 
                if nLeptons >= 1:
                    selection = (dictDF[sample]["nLep"] == nLeptons) & (dictDF[sample]["n_j_central"] <= nCenJets ) & (dictDF[sample]["maxEta"] >= maxEtaVal ) & (dictDF[sample]["xgboostCat"] < 31 )
                    data_selection = (dataDF["nLep"] == nLeptons) & (dataDF["n_j_central"] <= nCenJets ) & (dataDF["maxEta"] >= maxEtaVal ) & (dataDF["xgboostCat"] < 31 )

            if sample == "tHjb":
                tHjbYield = dictDF[sample][selection]['totalWeight'].sum()
            elif sample == "tWH":
                tWHYield = dictDF[sample][selection]['totalWeight'].sum()
            elif sample == "ttH":
                ttHYield = dictDF[sample][selection]['totalWeight'].sum()
            else:
                bkgHiggsYield += dictDF[sample][selection]['totalWeight'].sum()

        dataYield = dataDF[data_selection]['totalWeight'].sum()*(10.0/45.0)
        Z_tH_newCat = computeSignificance(tHjbYield + tWHYield,ttHYield + dataYield + bkgHiggsYield)
        Z_ttH_newCat = computeSignificance(ttHYield,tHjbYield + tWHYield + dataYield + bkgHiggsYield)

        ## calculate the significance from the nominal selection but removing the events which satisfy our selection with ~selection
        Z_ttH_Cats = [ ]
        Z_tH_Cats = [ ]

        ## xgboost categorisation 
        minRange = 27
        maxRange = 34

        if nLeptons >= 1:
            minRange = 31
            maxRange = 34
        else:
            minRange = 27
            maxRange = 31

        ## calculate the significance from the nominal selection but removing the events which satisfy our selection with ~selection
        for catNo in range(minRange,maxRange):

            bkgHiggsYieldCat = 0.0

            dataYield = dataDF[(~data_selection) & (dataDF["xgboostCat"] ==  catNo )]['totalWeight'].sum()*(10.0/45.0)
            for sample in SAMPLES:

                if prioritytH:
                    selection = (dictDF[sample]["nLep"] == nLeptons) & (dictDF[sample]["n_j_central"] <= nCenJets ) & (dictDF[sample]["maxEta"] >= maxEtaVal )
                else:
                    # if 0 leptons and we want to give priority to XGBoost categories, we need to avoid picking events with xgboost >= 27
                    if nLeptons == 0:
                        selection = (dictDF[sample]["nLep"] == nLeptons) & (dictDF[sample]["n_j_central"] <= nCenJets ) & (dictDF[sample]["maxEta"] >= maxEtaVal ) & (dictDF[sample]["xgboostCat"] < 27 )
                    # if >= 1 lepton and we want to give priority to XGBoost, we need to avoid picking events with xgboostcat >= 31 
                    if nLeptons >= 1:
                        selection = (dictDF[sample]["nLep"] == nLeptons) & (dictDF[sample]["n_j_central"] <= nCenJets ) & (dictDF[sample]["maxEta"] >= maxEtaVal ) & (dictDF[sample]["xgboostCat"] < 31 )

                if sample == "tHjb":
                    tHjbYield = dictDF[sample][(~selection) & (dictDF[sample]["xgboostCat"] ==  catNo )]['totalWeight'].sum()
                elif sample == "tWH":
                    tWHYield = dictDF[sample][(~selection) & (dictDF[sample]["xgboostCat"] ==  catNo )]['totalWeight'].sum()
                elif sample == "ttH":
                    ttHYield = dictDF[sample][(~selection) & (dictDF[sample]["xgboostCat"] ==  catNo )]['totalWeight'].sum()
                else:
                    bkgHiggsYieldCat += dictDF[sample][(~selection) & (dictDF[sample]["xgboostCat"] ==  catNo )]['totalWeight'].sum()

            Z_ttH_Cats.append( computeSignificance(ttHYield, tWHYield + tHjbYield + dataYield + bkgHiggsYieldCat) )
            Z_tH_Cats.append( computeSignificance(tWHYield + tHjbYield, ttHYield + dataYield + bkgHiggsYieldCat) )

        Z_ttH_Cats.append(Z_ttH_newCat)
        Z_tH_Cats.append(Z_tH_newCat)

        ttHSignificance = math.sqrt(sum(Z*Z for Z in Z_ttH_Cats))
        tHSignificance = math.sqrt(sum(Z*Z for Z in Z_tH_Cats))

        print('new significances for max eta cut :' , round(maxEtaVal,2))
        print('ttH significance : ' , round(ttHSignificance,3))
        print('tH significance : ', round(tHSignificance,3))

        maxEtaSelection.append( maxEtaVal )
        ttHZs.append( ttHSignificance )
        tHZs.append( tHSignificance )
        maxEtaVal += 0.1

    ax.plot(maxEtaSelection, ttHZs, label='ttH significance')
    ax.set_ylabel('ttH significance')
    ax.set_xlabel('$|\eta^{max}|$ selection')

    if prioritytH == False:
        # modify the y-axis of the ttH plot
        if nLeptons >= 1:
            ax.set_ylim(3.17, 3.19)
        else:
            ax.set_ylim(2.99,3.01)

    plt.savefig('outputPlots/ttH_significance.png')
    plt.cla()

    ax.plot(maxEtaSelection, tHZs, label='tH significance')
    ax.set_ylabel('tH significance')
    ax.set_xlabel('$|\eta^{max}|$ selection')

    plt.savefig('outputPlots/tH_significance.png')
    plt.cla()

### compute the purities with a new selection
### for now only add a single tH leptonic selection
### selection - N_j_central <= 2, N_lep == 1, N_j_btag >= 1, |eta^{Max}| >= 2.4.

# loop over higgs samples, creating dataframes and lists

yieldsPerCategory = [ ]
puritiesPerCategory = [ ]
efficienciesPerCategory = [ ]

for sample in SAMPLES:
    
    yields = [ ]
    purity = [ ]
    effs = [ ] 

    yieldsPerCategory.append(yields)
    puritiesPerCategory.append(purity)
    efficienciesPerCategory.append(effs)

# create dicts of data frames and the lists
dictYields = dict(zip(SAMPLES,yieldsPerCategory))
dictPurities = dict(zip(SAMPLES,puritiesPerCategory))
dictEfficiencies = dict(zip(SAMPLES,efficienciesPerCategory))

totalHiggsYields = [ ]
dataYields = [ ]

## for the fit
newNCATEGORIES = 8
NPROCESS = 9
lumi = (36104.16 + 43593.8 + 58450.1) / 1000.0

# note: i think it should be the cross sections with the rapidity of the Higgs < |2.5| rather than the full cross sections as done below
xsec_production_modes = [ ] 
for sample in SAMPLES:
    xsec_production_modes.append( dictDF[sample]['crossSection'][0]*1000.0 )

# number of true events
NTRUE = np.array(xsec_production_modes) * lumi

### compute the yields and efficiences of the new category

### selection - N_j_central <= 2, N_lep == 1, N_j_btag >= 1, |eta^{Max}| >= 2.4.
selection = (dictDF[sample]["nLep"] == 1) & (dictDF[sample]["n_j_central"] <= 2 ) & (dictDF[sample]["maxEta"] >= 2.4 )
## avoid picking events that satisfy this selection
data_selection = (dataDF["nLep"] == 1) & (dataDF["n_j_central"] <= 2 ) & (dataDF["maxEta"] >= 2.4 )

# loop over the categories currently in use 
for catNo in range(27,34):
    
    totalHiggsYield = 0.0
    dataYield = dataDF[(~data_selection) & (dataDF["xgboostCat"] ==  catNo )]['totalWeight'].sum()*(10.0/45.0)

    dataYields.append( dataYield )

    for i, sample in enumerate(SAMPLES):

        ## avoid picking events that satisfy this selection
        selection = (dictDF[sample]["nLep"] == 1) & (dictDF[sample]["n_j_central"] <= 2 ) & (dictDF[sample]["maxEta"] >= 2.4 )

        yieldFromSample = dictDF[sample][(~selection) & (dictDF[sample]["xgboostCat"] ==  catNo )]['totalWeight'].sum()
        dictYields[sample].append( yieldFromSample )
        totalHiggsYield += dictDF[sample][(~selection) & (dictDF[sample]["xgboostCat"] ==  catNo )]['totalWeight'].sum()
        
        dictEfficiencies[sample].append( yieldFromSample / NTRUE[i] )

    totalHiggsYields.append(totalHiggsYield)


### add on the tH lep category now

dataYield = dataDF[(data_selection)]['totalWeight'].sum()*(10.0/45.0)
dataYields.append(dataYield)
totalHiggsYield = 0.0

for i, sample in enumerate(SAMPLES):

    ## pick events that satisfy this!
    selection = (dictDF[sample]["nLep"] == 1) & (dictDF[sample]["n_j_central"] <= 2 ) & (dictDF[sample]["maxEta"] >= 2.4 )
    yieldFromSample = dictDF[sample][(selection)]['totalWeight'].sum()
    dictYields[sample].append( yieldFromSample )
    totalHiggsYield += dictDF[sample][(selection)]['totalWeight'].sum()

    dictEfficiencies[sample].append( yieldFromSample / NTRUE[i] )

totalHiggsYields.append(totalHiggsYield)

# compute the purities

purities = [ ] 
for sample in SAMPLES:
    for i in range(len(totalHiggsYields)):
        dictPurities[sample].append( dictYields[sample][i] / totalHiggsYields[i] )

    purities.append(dictPurities[sample])

CatLabels = [ "C27","C28","C29","C30", "C31", "C32", "C33", "tH lep"]
purities = pandas.DataFrame(purities, index=SAMPLES, columns=CatLabels)
fig, ax = plt.subplots(figsize=(10, 8))
purities = purities.reindex(SAMPLES, axis=0)
purities.T.plot.barh(stacked=True, ax=ax, width=0.9)
ax.set_xlim(0, 1)
ax.set_xlabel('Fraction of Signal Process / Category', horizontalalignment='right', x=1.0)
plt.savefig('outputPlots/newPurities.png')


## get the expected number of bkg events for the fit
newEXPECTED_BKG_CAT = np.array(dataYields)

## for each category, find the efficiency of each sample

efficienciesForWS = [ ]

# loop over each category
for i in range(len(dictYields['ttH'])):
    # the efficiencies for each sample for a single category
    effSingleCat = [ ] 
    for sample in SAMPLES:
        effSingleCat.append( dictEfficiencies[sample][i] )
    efficienciesForWS.append(effSingleCat)

# and the cross sections:
newEFFICIENCIES = np.array(efficienciesForWS) * 0.9





