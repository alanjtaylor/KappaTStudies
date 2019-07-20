import yaml
import ROOT
import os
import sys

from optparse import OptionParser, OptionGroup, OptionValueError
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--process', action='store', required=True, help='Give name of process (eg ttH)')
parser.add_argument('-c', '--campaign', action='store', required=True, help='Give the MC campaign (eg mc16a)')
args = parser.parse_args()

### function for computing the max |eta| of all the jets in the event:

getMaxEta_code ='''
float getMaxEta (ROOT::VecOps::RVec<float> eta_jets)
{
    float maxEta = 0.0;
    for(int i=0;i<eta_jets.size();i++){
    if (std::abs(eta_jets[i]) >= maxEta) 
    {
    maxEta = std::abs(eta_jets[i]);
    }
    }
    return maxEta;

}
'''
ROOT.gInterpreter.Declare(getMaxEta_code)


getNBtags_code ='''
int getNBtags (ROOT::VecOps::RVec<float> btag_jets)
{
    int NBtags = 0;
    for(int i=0;i<btag_jets.size();i++){

    if (btag_jets[i] == 1 ) {
    NBtags++;
    }
    }
    
    return NBtags;

}
'''
ROOT.gInterpreter.Declare(getNBtags_code)


### start skimming

print 'Creating RDF for process ', args.process, ' and for MC / DS  ' , args.campaign
configFile = yaml.safe_load(open('files.yaml'))

## check if the data year and DS exist, if not then exit.
if args.campaign in configFile["Files"][args.process]:
    print('Got dataset!')
else:
    sys.exit('Exiting, that data year and DS number does not exist')


scaleFactor = 1.0
if args.process[:-2] != "data":

    lumi = configFile[args.campaign+"_lumi"]
    fileName = configFile["EOSPrefix"] + args.campaign + "/Nominal/" + configFile["Files"][args.process][args.campaign]
    histoName = configFile["Files"][args.process]["histogram"]
    
    ## get histogram from file for normalisation
    aFile = ROOT.TFile(fileName)
    sumWeights = (  aFile.Get(histoName).GetBinContent(1) / aFile.Get(histoName).GetBinContent(2) )*aFile.Get(histoName).GetBinContent(3)
    scaleFactor = lumi / sumWeights

    treeName = "CollectionTree"
    d = ROOT.RDataFrame(treeName, fileName)

if args.process[:-2] == "data":

    fileName = configFile["EOSPrefix"] + args.process + "/" + configFile["Files"][args.process][args.campaign]
    print fileName
    treeName = "CollectionTree"
    d = ROOT.RDataFrame(treeName, fileName)


# filter the dataframe with the cuts
cut_str = "HGamEventInfoAuxDyn.isPassed == 1 && getNBtags(HGamAntiKt4EMTopoJetsAuxDyn.MV2c10_FixedCutBEff_77) >= 1 && (HGamEventInfoAuxDyn.N_lep >= 1 || HGamEventInfoAuxDyn.N_j_central >= 3 )"

# gets it correct!
#cut_str = "HGamEventInfoAuxDyn.isPassed == 1 && HGamEventInfoAuxDyn.catCoup_XGBoost_ttH >= 27" 

## total weight expression -- this is different for data and MC

totalWeightStr = "HGamEventInfoAuxDyn.crossSectionBRfilterEff*HGamEventInfoAuxDyn.weight*"+str(scaleFactor)
xsecStr = "HGamEventInfoAuxDyn.crossSectionBRfilterEff"

if args.process[:-2] == "data":
    # remove the signal region in data
    cut_str += " && ( HGamEventInfoAuxDyn.m_yy*0.001 < 120.0 || HGamEventInfoAuxDyn.m_yy*0.001 > 130.0 ) "
    totalWeightStr = "HGamEventInfoAuxDyn.weight"

    ## no cross section in data so just set it to weight which is equal to 1.0 
    xsecStr = "HGamEventInfoAuxDyn.weight"

d_cut = d.Filter(cut_str)

dOut =  d_cut.Define("myy", "HGamEventInfoAuxDyn.m_yy*0.001") \
            .Define("n_j_central", "HGamEventInfoAuxDyn.N_j_central") \
            .Define("n_j_central30", "HGamEventInfoAuxDyn.N_j_central30") \
            .Define("n_j", "HGamEventInfoAuxDyn.N_j") \
            .Define("n_j_30", "HGamEventInfoAuxDyn.N_j_30") \
            .Define("maxEta", "getMaxEta(HGamAntiKt4EMTopoJetsAuxDyn.eta)") \
            .Define("nBTags", "getNBtags(HGamAntiKt4EMTopoJetsAuxDyn.MV2c10_FixedCutBEff_77)") \
            .Define("nLep", "HGamEventInfoAuxDyn.N_lep") \
            .Define("weight", "HGamEventInfoAuxDyn.weight") \
            .Define("crossSection", xsecStr) \
            .Define("totalWeight", totalWeightStr) \
            .Define("xgboostCat", "HGamEventInfoAuxDyn.catCoup_XGBoost_ttH")


if not os.path.exists("skimmedFiles"):
    print 'making skimmedFiles directory...'
    os.makedirs("skimmedFiles")
                
outFileName = "skimmedFiles/"+args.process+"_"+args.campaign+".root"
dOut.Snapshot(treeName, outFileName, dOut.GetDefinedColumnNames())
