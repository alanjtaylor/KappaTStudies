import ROOT
import pandas
import uproot
from matplotlib import pyplot as plt
import matplotlib as mpl
import math
import numpy as np


SAMPLES = ["ggH", "VBF", "ZH", "WH", "ttH", "tHjb", "tWH", "ggZH", "bbH"]

# lists for pandas dataframes, the yields in each category and the purities
frames = [ ]
yieldsPerCategory = [ ]
puritiesPerCategory = [ ]
efficienciesPerCategory = [ ] 

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

## do the fit
NCATEGORIES = 7
NPROCESS = 9
lumi = (36104.16 + 43593.8 + 58450.1) / 1000.0

# note: i think it should be the cross sections with the rapidity of the Higgs < |2.5| rather than the full cross sections as done below
xsec_production_modes = [ ] 
for sample in SAMPLES:
    xsec_production_modes.append( dictDF[sample]['crossSection'][0]*1000.0 )

# number of true events
NTRUE = np.array(xsec_production_modes) * lumi

# loop over the categories currently in use 
for catNo in range(27,34):
    
    totalHiggsYield = 0.0
    dataYields.append( dataDF[dataDF['xgboostCat'] == catNo]['totalWeight'].sum()*(10.0/45.0) )

    for i, sample in enumerate(SAMPLES):

        yieldFromSample = dictDF[sample][dictDF[sample]['xgboostCat'] == catNo]['totalWeight'].sum()
        dictYields[sample].append( yieldFromSample )
        totalHiggsYield += dictDF[sample][dictDF[sample]['xgboostCat'] == catNo]['totalWeight'].sum()
        
        dictEfficiencies[sample].append( yieldFromSample / NTRUE[i] )

    totalHiggsYields.append(totalHiggsYield)

# compute the purities

purities = [ ] 
for sample in SAMPLES:
    for i in range(len(totalHiggsYields)):
        dictPurities[sample].append( dictYields[sample][i] / totalHiggsYields[i] )

    purities.append(dictPurities[sample])

CatLabels = [ "C27","C28","C29","C30", "C31", "C32", "C33" ]
purities = pandas.DataFrame(purities, index=SAMPLES, columns=CatLabels)
fig, ax = plt.subplots(figsize=(10, 8))
purities = purities.reindex(SAMPLES, axis=0)
purities.T.plot.barh(stacked=True, ax=ax, width=0.9)
ax.set_xlim(0, 1)
ax.set_xlabel('Fraction of Signal Process / Category', horizontalalignment='right', x=1.0)
plt.savefig('outputPlots/nominalPurities.png')

## get the expected number of bkg events for the fit
EXPECTED_BKG_CAT = np.array(dataYields)

## for each category, find the efficiency of each sample

efficienciesForWS = [ ]

for i in range(len(dictYields['ttH'])):
    # the efficiencies for each sample for a single category
    effSingleCat = [ ] 
    for sample in SAMPLES:
        effSingleCat.append( dictEfficiencies[sample][i] )
    efficienciesForWS.append(effSingleCat)

# and the cross sections:
EFFICIENCIES = np.array(efficienciesForWS) * 0.9



