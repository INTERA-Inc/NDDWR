this directory contains all raw data used to build/calibrate wahpeton model
Raw means this data can be redownlaoded in its orignal form from its original source. 


Water levels digitized from gws_76.pdf and plates 
They are assumed to be from 1970
All water levels were digitized except for those with a (-) by them which indicates that the water level is lower than the shown elevation

For data downloaded from the web tool (for transient calibration)

All available data was downloaded. 
Data was not used if:
-there was no measuring pt/depth to water information (if either of these was missing - only two data pts had a depth to water with no measuring pt. site index 127048 and 127049, data pts from nov 2006. These are not used)
-the well was outside of the grid extent

A significant amount of data passed these tests (800,000 data pts) (wahp_waterlevels.csv). File size was 700 MB, too big for git. An initial low pass filter was used to reduce data (daily average) - wahp_waterlevels_daily.csv. All of this is kept on the S drive:

