# refer to the ReadMe.md
# set conda environment to python 3.9
# conda install/pip climada under conda
# crowd_process wont be used in the test.py, comment them in scClim: __init__.py
# plot method is adjusted in order to work in .py state management

# Config dir and force climada to look for environment variable in specific dir
import json
config = {
    "log_level": "INFO",
    "local_data": {
        "func_dir": "/Users/ray/Desktop/202309_hail_damage_model/scClim",
        "data_dir": "/Users/ray/Desktop/202309_hail_damage_model/test_data",
        "out_dir": "/Users/ray/Desktop/202309_hail_damage_model/out_files",
        "crowd_url": "placeholder"
    }
}
with open("/Users/ray/.config/climada.conf", "w") as f:
    json.dump(config, f, indent=2)

import os
os.environ["CLIMADA_CONF"] = "/Users/ray/.config/climada.conf"

# Test environment variable
from climada.util.config import CONFIG
print(CONFIG.local_data.data_dir)

import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
from climada.entity import ImpactFuncSet, ImpactFunc
from climada.engine import ImpactCalc
import scClim as sc
from scClim.constants import UNIT_DICT,CUT_OFF_DICT,INT_RANGE_DICT,INT_LABEL_DICT,W_SIZE_DICT # thresholds and def of the model
import matplotlib.pyplot as plt

data_dir = "./test_data"
haz_var = 'MESHS' #Maximum Expected Severe Hail Size
exposure = 'KGV' # "Kantonale Gebäudeversicherung" (data format of cantonal building insurances)

years = [2021]
n_years = 1


#set cut_off and intensity_range
unit = UNIT_DICT[haz_var]
cut_off = CUT_OFF_DICT[haz_var]
intensity_range= INT_RANGE_DICT[haz_var]
intensity_label= INT_LABEL_DICT[haz_var]

#windowsize (rolling) in hazard units:
assert(len(np.unique(np.diff(intensity_range[1:])))==1)
stepsize = np.unique(np.diff(intensity_range[1:]))[0]
window_size = W_SIZE_DICT[haz_var]
unit_windowsize = stepsize*window_size

# Create Hazard object
meshs = xr.open_dataset(f'{data_dir}/test_meshs.nc')
meshs.load()
haz = sc.hazard_from_radar(
    meshs, extent=[5.8, 10.6, 45.7, 47.9], varname=haz_var)
haz.plot_intensity(0)

# Create Exposure object
exp = sc.read_gvz_exposure(f'{data_dir}/test_exp.csv',
                            crs = 'EPSG:4326')
exp.assign_centroids(haz)
exp.gdf=exp.gdf.rename(columns={'id_col':'VersicherungsID'})
exp.tag.description = "Exposure Hexbin" 
exp.plot_hexbin(gridsize=40)
plt.show()

# Create Impact object
imp_path = f'{data_dir}/test_dmg.csv'
imp_measured = sc.read_gvz_dmg(imp_path, 'KGV',return_type='imp',
                                years=(years[0],years[-1]),
                                baujahr_filter='',index_dmgs=False,
                                crs='EPSG:4326',id_col='id_col')

eai_exp_obs = imp_measured._build_exp()                # Exposures object
eai_exp_obs.tag.description = "Observed expected annual impact"  # ONE title
ax = eai_exp_obs.plot_hexbin(gridsize=40)              # no title kwarg
plt.show()

# create idendity impact function (to determine dates
# with nonzero hazard intensity over an exposed asset)
imp_fun_set = ImpactFuncSet()
imp_fun_identity = ImpactFunc.from_step_impf((0, 1, max(intensity_range)*2))
imp_fun_identity.haz_type = haz.haz_type   # e.g. 'HAIL'
imp_fun_identity.id = 1
imp_fun_set.append(imp_fun_identity)

# calculate impact (with identitiy impact function)
imp = ImpactCalc(exp, imp_fun_set, haz).impact(save_mat=True)

#get dates with nonzero modelled impact
dates_modeled_imp = np.array([dt.datetime.fromordinal(d) for d in imp.date[imp.at_event > 0]])

# Perform empirical calibration
calib_tuple = sc.empirical_calibration_per_exposure(hazard_object = haz,
    exposure_object = exp, damages = imp_measured, exposure_type = 'buildings',
    variable = haz_var,filter_year=None,dates_modeled_imp=dates_modeled_imp,
    roll_window_size=window_size,get_PVA=True)
ds, df_all, ds_roll, ds_roll_cut, values_at_centroid_all, intensity_range = calib_tuple

#make sure zero values are included (not NaN)
values_at_centroid_all['PAA'] = (values_at_centroid_all['n_dmgs']/
                                 values_at_centroid_all['n_exp'])
values_at_centroid_all['MDR'] = (values_at_centroid_all['dmg_val']/
                                 values_at_centroid_all['exp_val'])
#save values_at_centroids to csv
values_at_centroid_all.to_csv(f'{data_dir}/temp/at_centr.csv',index=False)

n_samples=200 #number of bootstrapping samples (1000 in the paper)
out_tuple=sc.bootstrapping(ds, ds_roll,haz_var,n_samples,intensity_range,
                           log_fit=False,cut_off=cut_off,keep_raw_values=True)
ds_boot,ds_boot_roll,ds_boot_roll_cut, fit_data = out_tuple

# Get empirical damage functions
df, df_roll, df_roll_cut, n_dmgs = sc.compute_empirical_damage_functions(
    ds, ds_roll, ds_roll_cut,get_monotonic_fit=False)

# fit Sigmoidal function
y_bounds = [min(intensity_range),max(intensity_range)]
v_tresh_bounds = (0,20) #visually detemined (MDR/PAA is >0 at MESHS=20)
pbounds={'v_thresh': v_tresh_bounds, 'v_half': y_bounds,
         'scale': [df.MDR.max()/10, min(1,df.MDR.max()*10)],
          'power': (3,3)
          }
p,res,impf_emanuel = sc.calib_opt.fit_emanuel_impf_to_emp_data(df,pbounds,plot=False)

pboundsPAA={'v_thresh': v_tresh_bounds, 'v_half': y_bounds,
            'scale': [df_roll.PAA.max()/10, min(1,df.PAA.max()*10)]}
p_PAA,resPAA,impf_emanuelPAA = sc.calib_opt.fit_emanuel_impf_to_emp_data(
    df,pboundsPAA,opt_var='PAA',plot=False)

df_roll.loc[df_roll.MDD==np.inf,'MDD']=np.nan
pboundsMDD={'v_thresh': v_tresh_bounds, 'v_half': y_bounds,
            'scale': [df.MDD.max()/10, df.MDD.max()*5]}
p_MDD,resMDD,impf_emanuelMDD = sc.calib_opt.fit_emanuel_impf_to_emp_data(
    df_roll,pboundsMDD,opt_var='MDD',plot=False)

# Emanuel fit for every bootstrapped sample (here for MDR only)
df_boot_emanuel = pd.DataFrame(index = intensity_range, dtype=float,
                               columns = [f'b_{i}' for i in range(n_samples)])
for i in range(n_samples):
    df_now = ds_boot.isel(b_sample=i).to_dataframe()
    pNow,resNow,impf_emanuelNow = sc.calib_opt.fit_emanuel_impf_to_emp_data(
        df_now,pbounds,plot=False,verbose=False)
    df_boot_emanuel.loc[impf_emanuelNow.intensity,f'b_{i}']=impf_emanuelNow.mdd

# plot rolling MDR
dmg_bin_size=sc.constants.DMG_BIN_DICT[haz_var]
title=f'Empirical Mean Damage Ratio; {unit_windowsize} {unit} moving average'

# MDR plot
fig,ax=sc.impf_plot2(df_all,df_roll,df_roll_cut,ds_boot_roll,ds_boot_roll_cut,
                     haz_var,impf_emanuel,cut_off,'MDR',title,dmg_bin_size,
                     intensity_label,color='blue',df_boot_emanuel=df_boot_emanuel)
plt.show()

intensity = np.arange(0, 100, 1)

df_impf = pd.DataFrame(index=intensity, columns=['PAA', 'MDD', 'MDR'])
for var in ['PAA', 'MDD', 'MDR']:
    df_impf.loc[1:, var] = df_roll[var].loc[intensity_range[1:]]

    # add smoothed function
    smooth = sc.smooth_monotonic(
        df_roll_cut.index[1:], df_roll_cut[var].loc[intensity_range[1:]])
    # plt.plot(df_roll_cut.index[1:],smooth)

    df_impf.loc[intensity_range[1]:, var+'_smooth'] = smooth

#add Sigmoidal function fit
np.testing.assert_array_equal(impf_emanuelPAA.intensity,intensity_range)
df_impf.loc[intensity_range[1:], 'PAA_emanuel'] = impf_emanuelPAA.mdd[1:]
df_impf.loc[intensity_range[1:], 'MDR_emanuel'] = impf_emanuel.mdd[1:]
df_impf.loc[intensity_range[1:], 'MDD_emanuel'] = impf_emanuelMDD.mdd[1:]

# set values above cut_off to values at cut_off
cut_vars = ['PAA', 'MDD', 'MDR']
if haz_var == 'MESHS':  # cut off at 59 instead of 60 for realistic values
    df_impf.loc[cut_off-1:,
                cut_vars] = df_impf.loc[cut_off-1, cut_vars].values
elif haz_var != 'POH': # for POH there is no cut off!
    df_impf.loc[cut_off:, cut_vars] = df_impf.loc[cut_off, cut_vars].values

#Save impact function data as .csv file
df_impf.fillna(0).to_csv(
    data_dir+'/temp/test_impf.csv')

impf_path = data_dir+'/temp/test_impf.csv'
imp_fun_set = sc.impf_from_csv(impf_path, smooth=False,
                                emanuel_fit=True, plot=True)
plt.show()

# Step5 Compare modelled and observed damages
imp = ImpactCalc(exp, imp_fun_set, haz).impact(save_mat=True)
# imp.plot_hexbin_eai_exposure(ignore_zero=False, gridsize=40)

dmg_thresh = 1e4
imp_now=imp
imp_obs_now = imp_measured
xmin=1e2
#Impf type: Emprirically calibrated function in the form of Emanuel (2011)
impf = 'emp_emanuel'

#create impact dataframe with modelled and reported impacts per event (day)
#get all dates with modelled OR reported damages above xmin
ord_dates_nonZero = np.sort(np.unique(np.concatenate((imp_now.date[imp_now.at_event>xmin],
                                                      imp_obs_now.date[imp_obs_now.at_event>xmin]))))
imp_df = pd.DataFrame(index=ord_dates_nonZero,data={
    'date':[dt.datetime.fromordinal(d) for d in ord_dates_nonZero]})

imp_dfMod = pd.DataFrame(data={"imp_modelled":imp_now.at_event},index = imp_now.date)
imp_dfObs = pd.DataFrame(data={"imp_obs":imp_obs_now.at_event},index = imp_obs_now.date)

imp_df= imp_df.join(imp_dfMod,how='left') #join on index
imp_df= imp_df.join(imp_dfObs,how='left') #join on index
imp_df = imp_df.fillna(0) #fill NaN with zeros (for days with no reported or modelled damages)


#caculate skill scores
rmse,rmsf,rmsf_weighted,FAR,POD,p_within_OOM,n_ev = sc.E.calc_skill_scores(imp_df,dmg_thresh)

#create dictionary with evaluation metrics to pass to plotting function
eval_dict = {var_name: globals()[var_name] for var_name in ["rmse","rmsf","rmsf_weighted",
            "FAR","POD","p_within_OOM","n_ev","haz_var","exposure","impf"]}

#plot scatter plot
fig = sc.plot_funcs.scatter_from_imp_df(imp_df,imp_now.unit,xmin,dmg_thresh,eval_dict)
plt.show()

#Compare individual event (adjusted)
date = dt.datetime(2021,6,15).toordinal()
event_id = int(imp.event_id[imp.date == date])
exp_evt = imp._build_exp_event(event_id)
exp_evt.tag.description = (f"Modelled impact: "
                           f"{dt.datetime.fromordinal(date):%Y-%m-%d}")
ax = exp_evt.plot_hexbin(
        gridsize = 40,
        vmin      = 1)

event_id_obs = int(imp_measured.event_id[imp_measured.date == date])
exp_obs      = imp_measured._build_exp_event(event_id_obs)
exp_obs.tag.description = (f"Observed impact: "
                           f"{dt.datetime.fromordinal(date):%Y-%m-%d}")
ax = exp_obs.plot_hexbin(gridsize=40, vmin=1)   # no title kwarg

plt.show()


 

##########################################
# save the necessary data to csv in the out_dir path
from pathlib import Path

CONFIG.local_data.out_dir = "/Users/ray/Desktop/202309_hail_damage_model/out_files"
# redefine the CSV output folder
csv_dir = Path(CONFIG.local_data.out_dir)
csv_dir.mkdir(parents=True, exist_ok=True)

# Export the hazard grid to a table
#    (reshape the xarray DataArray into a long table of [event, lat, lon, intensity])
meshs = xr.open_dataset(f"{data_dir}/test_meshs.nc")
da    = meshs['MZC']                # e.g. the “MZC” or “MESHS” DataArray
df_haz = da.to_dataframe(name="intensity") \
           .reset_index()
df_haz.to_csv("out_files/hazard_intensity.csv", index=False)

# Export the exposures GeoDataFrame
#    drop the geometry column if you just want coordinates + values
exp.gdf.drop(columns="geometry") \
       .to_csv(csv_dir / "exposures.csv", index=False)

# observed impacts
imp_obs_df = pd.DataFrame({
    "event_id":   imp_measured.event_id,
    "date":       [dt.datetime.fromordinal(int(d)) for d in imp_measured.date],
    "damage_obs": imp_measured.at_event
})
imp_obs_df.to_csv(csv_dir / "observed_impacts.csv", index=False)

# 3) Modelled impacts
imp_mod_df = pd.DataFrame({
    "event_id":    imp.event_id,
    "date":        [dt.datetime.fromordinal(int(d)) for d in imp.date],
    "damage_mod":  imp.at_event
})
imp_mod_df.to_csv(csv_dir / "modelled_impacts.csv", index=False)

# 4) Empirical calibration tables
df_all.to_csv(csv_dir / "empirical_all.csv",    index=True)   # index = intensity
df_roll.to_csv(csv_dir / "empirical_rolling.csv", index=True)
df_impf.to_csv(csv_dir / "final_impact_function.csv", index=True)

# 5) Event‐level damage comparison
imp_df.to_csv(csv_dir / "impact_comparison_by_event.csv", index=False)

# 6) Bootstrap samples flatten to DataFrame
#    ds_boot has dims (b_sample, intensity, variable)
#    we'll assume variable name is 'MDR'
boot_df = ds_boot["MDR"].to_dataframe(name="MDR") \
                    .reset_index()            # brings b_sample & intensity into cols
boot_df.to_csv(csv_dir / "bootstrap_MDR.csv", index=False)

print(f"✅ Saved all CSVs to {csv_dir.resolve()}")
