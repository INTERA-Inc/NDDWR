import os
import sys
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0,os.path.join("..","..","dependencies"))
import pyemu


def prep_resp_mat_emul(org_m_d,new_t_d):
	if os.path.exists(new_t_d):
		shutil.rmtree(new_t_d)
	os.makedirs(new_t_d)

	files = ["respmat.bin","respmat_obs_info.csv","base_levels.csv"]
	for f in files:
		shutil.copy2(os.path.join(org_m_d,f),os.path.join(new_t_d,f))

	respmat = pyemu.Matrix.from_binary(os.path.join(org_m_d,"respmat.bin")).to_dataframe()
	dvnames = respmat.index.to_list()
	onames = respmat.columns.to_list()
	pst = pyemu.Pst.from_par_obs_names(par_names=dvnames,obs_names=onames)
	pst.try_parse_name_metadata()
	par = pst.parameter_data
	par["partrans"] = "none"
	par["parval1"] = 1
	par["parubnd"] = 1
	par["parlbnd"] = 0

	obs = pst.observation_data
	rmobsdf = pd.read_csv(os.path.join(m_d,"respmat_obs_info.csv"),index_col=0)
	obs["botm"] = rmobsdf.loc[obs.obsnme,"botm"]
	obs["botm"].to_csv(os.path.join(new_t_d,"botm.csv"))
	tol = 1.0
	obs["obsval"] = obs.botm + tol
	obs["obgnme"] = "greater_than"

	par["ones"] = 1.0
	par["zeros"] = 0.0
	par.loc[:,["ones","zeros"]].T.to_csv(os.path.join(os.path.join(new_t_d,"dvecs.csv")))
	df = run_respmat(new_t_d)

	print(df)

def run_respmat(model_ws="."):
	respmat = respmat = pyemu.Matrix.from_binary(os.path.join(model_ws,"respmat.bin")).to_dataframe().T
	
	#print(respmat.columns)
	#print(respmat.index)
	#exit()
	dvecs = pd.read_csv(os.path.join(model_ws,"dvecs.csv"),index_col=0).loc[:,respmat.columns]
	print(np.dot(respmat.values,dvecs.loc["ones",:].values).max())
	base_levels = pd.read_csv(os.path.join(model_ws,"base_levels.csv"),index_col=0).loc[respmat.index]
	botm_values = pd.read_csv(os.path.join(model_ws,"botm.csv"),index_col=0).loc[respmat.index]
	emul_levels = {}
	bvals = base_levels.values
	for idx in dvecs.index:
		print(np.dot(respmat.values,dvecs.loc[idx,:].values).flatten().min())
		emul_level = (bvals - np.dot(respmat.values,dvecs.loc[idx,:].values)[0]).flatten()
		print(idx,emul_level.max())
		emul_levels[idx] = emul_level
		print(idx,emul_level.shape)
	df= pd.DataFrame(emul_levels,index=base_levels.index)
	bvals = botm_values.values.flatten()
	for col in df.columns:
		vals = df.loc[:,col].values
		#print(vals.shape,bvals.shape)
		dry = np.zeros_like(vals)
		dry[np.where(vals < bvals)] = 1.0
		print(dry.sum())	





if __name__ == "__main__":
	m_d = "master_flow_08_highdim_restrict_bcs_flood_full_final_rch_respmatsweep_fullalloc"
	new_t_d = "respmat_template"
	prep_resp_mat_emul(m_d,new_t_d)