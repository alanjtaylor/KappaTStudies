# KappaTStudies
Some scripts for studying the sign of the coupling between the Higgs boson and top quark. 

## Skim files

To skim the files into small ntuples, suitable for analysis, do:

    source setup.sh
    . RunAll.sh
    
This will run the skimFiles.py script over all data and MC samples. The skimFiles.py script can also be ran standalone over a single MC / data sample with the following:

    python skimFiles.py -p ggH -c mc16a

where p is the process (can also be data15-18) and c is the MC campaign (can also be DS1-8)

## Do analysis

On lxplus, it is necessary to do the following for the first time and in a separate terminal from the above:

    setupATLAS
    lsetup "lcgenv -p LCG_93python3 x86_64-centos7-gcc62-opt ROOT" "lcgenv -p LCG_93python3 x86_64-centos7-gcc62-opt pip"
    python3 -m venv venv
    source venv/bin/activate 
    pip install numpy scipy matplotlib pandas uproot countingworkspace
    mkdir outputPlots
    
After the first time, it is only necessary to do

    setupATLAS
    lsetup "lcgenv -p LCG_93python3 x86_64-centos7-gcc62-opt ROOT" "lcgenv -p LCG_93python3 x86_64-centos7-gcc62-opt pip"
    source venv/bin/activate

If we run:

    python nominalSelection.py 
    
It will compute the signal effiencies and background estimates for the nominal selection and calculate the purities. 

There is also a script that finds a new event selection, and can be run with (for example):

    python newSelection.py -p True -nlept 1 -njets 2 -scan True

where p decides whether or not to give priority to tH or ttH categories, nlept is the number of leptons, njets is the maximum number of jets allowed in the event and scan decides whether to do a scan over the maximum pseudorapidity of a jet in the event.
 
 This script will create plots of the central jet multiplicity and the maximum pseudorapidity of a jet in the event. It will also compute the signal efficiencies and background estimates for a new selection (defined in the script) and calculate the purities for the new selection.
 
 A maximum likelihood counting experiment fit can be run with (for example):

     python doFit.py -f kappa -s new
     
where f decides whether to fit the kappa or the mu and s decides whether to use the new or nominal selection. If the fit is ran over kappa for both the new and the nominal selection, the likelihood scans can be compared with `cosmeticKappa.py`
 
 
 
 
 
 
