path = "/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/171126_m90/";
fName = "171126_m90_2018.01.18_000_CheckMovement_015_ImageStack.tif";
savePath = "/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/171126_m90/";
avgName = "AVG_"+fName;

open(path+fName);
run("Z Project...", "stop=1 projection=[Average Intensity]");
arg = "value=208 downsample_value=0 template=AVG_"+fName+" stack="+fName+" log=[Generate log file] plot=[Plot RMS]"
run("moco ", arg)
selectWindow("New Stack");
fNameLength = lengthOf(fName);
saveName = substring(fName,0, (fNameLength-4)); 
saveAs("Results", savePath+saveName+"_moco.csv");
selectWindow("Errors");
close();
selectWindow("New Stack");
save(savePath+saveName+"_moco.tif");
close();
selectWindow("AVG_"+fName);
close();
selectWindow(fName);
close();
//selectWindow("Results");
//close();
