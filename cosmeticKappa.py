import ROOT

ROOT.gROOT.LoadMacro("/afs/cern.ch/work/a/altaylor/public/ATLASStyle/AtlasStyle.C")
ROOT.gROOT.LoadMacro("/afs/cern.ch/work/a/altaylor/public/ATLASStyle/AtlasLabels.C")
ROOT.gROOT.LoadMacro("/afs/cern.ch/work/a/altaylor/public/ATLASStyle/AtlasUtils.C")

ROOT.SetAtlasStyle()
ROOT.gROOT.SetBatch(ROOT.kTRUE)

afile = ROOT.TFile("kT_nominalSelection.root")
afileNew = ROOT.TFile("kT_newSelection.root")

kappaScan = afile.Get("histoScan")
kappaScanNew = afileNew.Get("histoScan")

c1 = ROOT.TCanvas()

kappaScan.Draw("PL")
kappaScanNew.Draw("PL SAME")

c1.SetLogy()

kappaScan.SetMarkerColor(1)
kappaScan.SetLineColor(1)
kappaScan.SetMarkerSize(1.0)

kappaScanNew.SetMarkerColor(ROOT.kRed)
kappaScanNew.SetLineColor(ROOT.kRed)
kappaScanNew.SetMarkerSize(1.0)

leftBin = kappaScan.FindBin(-1.6)
rightBin = kappaScan.FindBin(+1.6)

kappaScan.GetXaxis().SetRange(leftBin,rightBin)
kappaScan.GetYaxis().SetRangeUser(0.05,10.0)
kappaScan.GetXaxis().SetTitle("#kappa_{T}")
kappaScan.GetYaxis().SetTitle("-ln(L_{#kappa_{T}} / L_{#kappa_{T}=1})")

aline = ROOT.TLine(-1.6,0.5,+1.6,0.5)
aline.SetLineColor(ROOT.kBlue)
aline.SetLineStyle(ROOT.kDashed)
aline.SetLineWidth(2)
aline.Draw()

## add a legend

leg = ROOT.TLegend(0.40,0.20,0.65,0.35)
leg.SetLineColor(ROOT.kWhite)
leg.SetFillColor(ROOT.kWhite)
leg.SetTextSize(0.04)
leg.SetBorderSize(0)

leg.AddEntry(kappaScan,"Current","pl")
leg.AddEntry(kappaScanNew,"Current + tH lep","pl")
leg.Draw() 

c1.SaveAs("outputPlots/kappaScan.png")
