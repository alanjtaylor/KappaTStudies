#!/bin/bash
# skim the data
echo 'Skimming all data and all MC, takes around 40mins or so...'
for data in data15 data16 data17 data18; do
    for DS in DS1 DS2 DS3 DS4 DS5 DS6 DS7 DS8; do
	python skimFiles.py -p ${data} -c ${DS}
    done
done
# hadd all data files together
hadd skimmedFiles/data_total.root skimmedFiles/data??_DS?.root
# skim the MC
for process in ggH VBF WpH WmH ZH ggZH ttH bbH tWH tHjb; do
    for mc in a d e; do 
	python skimFiles.py -p ${process} -c mc16${mc}
    done
    # for each process, merge the MC campaigns together
    hadd skimmedFiles/${process}_total.root skimmedFiles/${process}_mc16?.root
done
hadd skimmedFiles/WH_total.root skimmedFiles/WmH_total.root skimmedFiles/WpH_total.root

