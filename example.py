# -*- coding: utf-8 -*-

import pickle as pk
import numpy as np
from astroquery.gaia import Gaia

Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

def corr_bprp_excess(bprp, bprp_excs):
    """
    calculate the C* value as defined in Riello et.al.2022
    bprp is bp_rp of Gaia DR3
    bprp_excs is phot_bp_rp_excess_factor of Gaia DR3
    """
    if bprp < 0.5:
        corr = 1.154360 + 0.033772*bprp + 0.032277*bprp**2
    elif bprp < 4:
        corr = 1.162004 + 0.011464*bprp + 0.049255*bprp**2 - 0.005879*bprp**3
    else:
        corr = 1.057572 + 0.140537*bprp
    return bprp_excs-corr    

#load XGBoost-250 model
xgb_250 = pk.load(open("models\gaia_teff_xgb250.pkl", "rb"))

#load data from Gaia DR3
query = Gaia.launch_job_async("SELECT * \
        FROM gaiadr3.gaia_source AS g, gaiadr3.astrophysical_parameters AS ap \
        WHERE g.source_id = ap.source_id \
        AND CONTAINS(POINT(g.ra,g.dec), CIRCLE({ra},{dec},{r}))=1\
        AND ap.teff_gspphot > 0;".format(ra=81.28, dec=-69.78, r=1.0/60.0), dump_to_file=False)


data_gaia = query.get_results()
data_gaia['C*'] = [corr_bprp_excess(a, b) for a,b in data_gaia['bp_rp', 'phot_bp_rp_excess_factor']]

#define columns needed for model
columns = ['ra', 'dec', 'l', 'b', 'parallax','parallax_error', 'pm',
 'pmra', 'pmdec', 'ruwe', 
 'ipd_frac_multi_peak',
 'ipd_frac_odd_win',
 'phot_g_mean_mag',  
  'bp_rp', 'bp_g', 'g_rp','teff_gspphot',
 'teff_gspphot_lower',
 'teff_gspphot_upper',
 'logg_gspphot',
 'logg_gspphot_lower',
 'logg_gspphot_upper',
 'mh_gspphot',
 'mh_gspphot_lower',
 'mh_gspphot_upper', 'azero_gspphot', 'C*']

data_gaia_cols = data_gaia[columns]

#convert astropy Table to unstructured numpy array
X = np.lib.recfunctions.structured_to_unstructured(np.array(data_gaia_cols))

#predict
flags_pred = xgb_250.predict(X)

#show Gaia entries with positive quality flags
print(data_gaia[flags_pred==1])