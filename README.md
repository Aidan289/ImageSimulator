# CASTOR Image Simulator

FORECASTOR Team - 2022-2024

This repository contains code developed to make mock astrophysical images for [CASTOR](https://www.castormission.org/) 
using the GalSim python package, as presented in the manuscript ["FORECASTOR -- II. Simulating Galaxy Surveys 
with CASTOR"](https://ui.adsabs.harvard.edu/abs/2024arXiv240217163M/abstract). Also included are the scripts used to make comparison images 
for LSST and GALEX.

These codes are also hosted on [CANFAR](https://www.canfar.net/en/), for ease of use. The code is designed to be used with simulated light-cones 
from the Santa Cruz Semi-Analytic Model, which are proprietry and not publicly available. The mock lightcones are available on CANFAR for members of the CASTOR 
team for the sole purpose of creating simulated images for CASTOR science planning. These codes can be adapted for use with other lightcones/input galaxy catalogues, 
for those external to the CASTOR team. 

Please contact Madeline Marshall at madeline_marshall@outlook.com or 
Tyrone Woods at Tyrone.Woods@umanitoba.ca for questions about access.

To use on CANFAR:
1. Ensure you have a Canadian Astronomy Data Centre account (or
   [request one](https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/auth/request.html) if you
   do not have one yet), with membership to the CASTOR project.
2. Go to [CANFAR](https://www.canfar.net/en/) and sign in to the
   [Science Portal](https://www.canfar.net/science-portal/). If you cannot access this,
   then you must send an email to [support@canfar.net](mailto:support@canfar.net)
   requesting access to the Science Portal.
3. Inside the [Science Portal](https://www.canfar.net/science-portal/), click the "`+`"
   icon to launch a new session. Under "`type`", select "`notebook`". If multiple
   `castor_etc` versions are available, you can select the specific version you would like
   to use under the "`container image`" field; these codes were designed to work on versions
   1.2.8 and above.
6. Click the blue "`Launch`" button to start your new `castor_etc` notebook session. 
7. These codes are set up to work "out of the box" within this [JupyterLab](https://jupyter.org/) 
   environment. The codes  can be found at /arc/projects/CASTOR/MockPhotometry, alongside the mock
   lightcones which are found in /arc/projects/CASTOR/Simulations. 
