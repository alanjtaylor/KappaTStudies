import ROOT
from countingworkspace import create_workspace, create_variables
import countingworkspace.utils
countingworkspace.utils.silence_roofit()
from nominalSelection import NCATEGORIES, NPROCESS, NTRUE, EFFICIENCIES, EXPECTED_BKG_CAT, SAMPLES, xsec_production_modes
from newSelection import newNCATEGORIES, newEFFICIENCIES, newEXPECTED_BKG_CAT
from optparse import OptionParser, OptionGroup, OptionValueError
import argparse


### arguments
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--fit', action='store', required=True, help='What to fit (kappa or mu)')
parser.add_argument('-s', '--selection', action='store', required=True, help='Selection to be used')
args = parser.parse_args()


########
# ATLAS Run 2 Luminosity
lumi = (36104.16 + 43593.8 + 58450.1) / 1000.0
combWS = ROOT.RooWorkspace("combWS","combWS")
combWS.factory('lumi[%f]' % lumi)

## add cross sections into workspace
ntrue = create_variables(combWS, 'xsec_{proc}',bins=SAMPLES,values=xsec_production_modes)

if args.selection == "nominal":

    create_workspace(NCATEGORIES, SAMPLES, efficiencies=EFFICIENCIES, nexpected_bkg_cat = EXPECTED_BKG_CAT,expression_nsignal_gen='prod:nsignal_gen_proc{proc}(mu_{proc}[1, -4, 5], lumi, xsec_{proc})',ws=combWS)

if args.selection == "new":

    create_workspace(newNCATEGORIES, SAMPLES, efficiencies=newEFFICIENCIES, nexpected_bkg_cat = newEXPECTED_BKG_CAT,expression_nsignal_gen='prod:nsignal_gen_proc{proc}(mu_{proc}[1, -4, 5], lumi, xsec_{proc})',ws=combWS)

combWS.set('all_exp').Print('V')

## set all the POIs to constant with the exception of ttH

for sample in SAMPLES:
    if sample != "ttH":
        combWS.obj("mu_"+sample).setConstant(True)

poi = ""
if args.fit == "mu":
    poi = "mu_ttH"

if args.fit == "kappa":
    
    poi = "kT"
    combWS.obj("mu_ttH").setConstant(True)

    combWS.factory('kT[1, -2, +2]')
    combWS.factory('kW[1]')
    combWS.factory('expr:k_tWH("2.91 * @0 * @0 + 2.21 * @1 * @1 - 4.22 * @0 * @1", kT, kW)')
    combWS.factory('expr:k_tHjb("2.63 * @0 * @0 + 3.58 * @1 * @1 - 5.21 * @0 * @1", kT, kW)')
    combWS.factory('expr:k_ttH("@0 * @0", kT)')
    combWS.factory("EDIT::model_kappa(model, mu_ttH=prod(mu_ttH, k_ttH), mu_tHjb=prod(mu_tHjb, k_tHjb), mu_tWH=prod(mu_tWH, k_tWH))")

    mc = combWS.obj("ModelConfig")
    mc.SetPdf(combWS.pdf("model_kappa"))


mc = combWS.obj("ModelConfig")
mc.SetParametersOfInterest(ROOT.RooArgSet(combWS.var(poi)))
pdf = combWS.obj('ModelConfig').GetPdf()
obs = combWS.obj('ModelConfig').GetObservables()

data_asimov = ROOT.RooStats.AsymptoticCalculator.GenerateAsimovData(pdf, obs)
data_asimov.Print('V')

# fit the pdf with the Asimov dataset
fr = pdf.fitTo(data_asimov, ROOT.RooFit.Save())

# print the results
fr.Print()

pl = ROOT.RooStats.ProfileLikelihoodCalculator(data_asimov,mc)
pl.SetConfidenceLevel(0.683)

firstPOI = mc.GetParametersOfInterest().find(poi)
interval = pl.GetInterval()

lowerLimit = interval.LowerLimit(firstPOI)
upperLimit = interval.UpperLimit(firstPOI)

print("68% interval on {0} is [{1},{2}]".format(firstPOI.GetName(),lowerLimit,upperLimit))

plot = ROOT.RooStats.LikelihoodIntervalPlot(interval)
plot.SetTitle(args.fit)
plot.SetRange(0.0,2.0)

if args.fit == "kappa":
    plot.SetRange(-2.0,2.0)

c = ROOT.TCanvas()
ROOT.gPad.SetLogy(True)
plot.Draw("tf1")
c.Draw()
c.SaveAs("outputPlots/"+poi+"_" + args.selection + "Selection" +".png")

## write the plot to a tfile
outputFile = ROOT.TFile( poi+"_" + args.selection + "Selection" +".root", 'recreate')
histoScan = plot.GetPlottedObject()
histoScan.SetName("histoScan")
histoScan.Write()
outputFile.Write()
outputFile.Close()

del pl



