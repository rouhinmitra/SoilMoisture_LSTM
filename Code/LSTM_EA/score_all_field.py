"""Score the static-decomposition arms and the all-field model, seed-paired vs baseline."""
import sys, warnings, logging, importlib
from pathlib import Path
import numpy as np, pandas as pd, torch
warnings.filterwarnings("ignore"); sys.path.insert(0, str(Path(__file__).resolve().parent))
logging.disable(logging.INFO)
import run_sensitivity as rs
from run_cv import CVConfig, CV_FOLDS
from run_variant import VARIANTS
from src.config import ModelConfig
from src.models import get_model
from src.evaluator import Evaluator
from src.data_loader import RZSMDataset
from src import data_loader as dl
from torch.utils.data import DataLoader
import run_season_split_partA as A

BASE=Path(__file__).resolve().parent; MV=BASE/"outputs"/"model_variants"
FA="/Users/rouhinmitra/SM_work/Code/Data/field_avg"
PT_DIR="/Users/rouhinmitra/SM_work/Data/Base/"
FD_DIR="/Users/rouhinmitra/SM_work/Data/Base_field/"
RN_DIR="/Users/rouhinmitra/SM_work/Data/Base_field_renorm/"
PT_P="/Users/rouhinmitra/SM_work/Code/Data/s2_pixels/presto_embeddings_fused_interpolated.csv"
FD_P=f"{FA}/presto_embeddings_field_interpolated.csv"
SEEDS=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,42]
S2C=['ndvi','b2','b3','b4','b5','b6','b7','b8','b8a','b11','b12']
S1C=['VV','VH','VV_VH_diff']
SM={"ne1":"US-Ne1","ne2":"US-Ne2","ne3":"US-Ne3"}
_O=dl.DataProcessor.load_and_clean

def install(dyn_field):
    if not dyn_field:
        dl.DataProcessor.load_and_clean=_O; return
    s2=pd.read_csv(f"{FA}/s2_daily_field.csv",parse_dates=["Date"])
    s1=pd.read_csv(f"{FA}/s1_daily_field.csv",parse_dates=["Date"])
    def patched(self,fp):
        fc=self.feature_config; keep=list(fc.dynamic_cols)
        fc.dynamic_cols=[c for c in keep if c not in S1C]
        try: df=_O(self,fp)
        finally: fc.dynamic_cols=keep
        if df is None: return df
        site=SM[dl._site_from_filepath(str(fp))]
        d=pd.to_datetime(df["Date"]).dt.normalize()
        a=s2[s2.Site==site][["Date"]+S2C].rename(columns={c:c+"_new" for c in S2C})
        df=df.merge(a,left_on=d,right_on="Date",how="left",suffixes=("","_d"))
        for c in S2C: df[c]=df[c+"_new"]
        df=df.drop(columns=[c+"_new" for c in S2C]+[c for c in ("key_0","Date_d") if c in df.columns])
        d2=pd.to_datetime(df["Date"]).dt.normalize()
        df=df.merge(s1[s1.Site==site][["Date"]+S1C],left_on=d2,right_on="Date",how="left",suffixes=("","_d"))
        return df.drop(columns=[c for c in ("key_0","Date_d") if c in df.columns])
    dl.DataProcessor.load_and_clean=patched

def cfg(ddir,presto,add_s1):
    e=rs.build_experiments()["baseline_no_doy"]
    c=CVConfig(); c.seq_length=20
    c.dynamic_cols=list(e["dynamic_cols"])+(S1C if add_s1 else [])
    c.static_cols=e["static_cols"]; c.use_presto_static=e["use_presto_static"]; c.add_temporal=False
    for k,v in VARIANTS["paper_config_fixed"].items(): setattr(c,k,v)
    c.data_dir=ddir; c.presto_embeddings_path=presto
    return c

ARMS=[("baseline",  MV/"paper_config_fixed_sensitivity", PT_DIR, PT_P, False, False),
      ("ae_field",  MV/"statdec_ae",                     FD_DIR, PT_P, False, False),
      ("ae_renorm", MV/"statdec_ae_renorm",              RN_DIR, PT_P, False, False),
      ("presto_fld",MV/"statdec_presto",                 PT_DIR, FD_P, False, False),
      ("all_field", MV/"all_field",                      FD_DIR, FD_P, True,  True)]
rows=[]
for arm,ck,ddir,presto,dynf,adds1 in ARMS:
    install(dynf); importlib.reload(A)
    c=cfg(ddir,presto,adds1)
    for fold in CV_FOLDS:
        proc,d=A.make_proc(fold,c,(4,10))
        ld=DataLoader(RZSMDataset(d["X_d_test"],d["X_s_test"],d["y_test"]),batch_size=256)
        dts=pd.to_datetime(d["dates_test"])
        for seed in SEEDS:
            p=ck/f"seed{seed}"/fold["name"]/"checkpoint_best.pt"
            if not p.exists(): continue
            sd=torch.load(p,map_location="cpu",weights_only=False); sd=sd.get("model_state_dict",sd)
            m=get_model(ModelConfig(model_type="LSTM",hidden_dim=c.hidden_dim,
                                    dropout=c.dropout,num_layers=c.num_layers),
                        d["X_d_test"].shape[2],d["X_s_test"].shape[2])
            m.load_state_dict(sd)
            yp,yt=Evaluator(m,proc.scaler_y,str(BASE),device="cpu").predict(ld)
            yp,yt=np.asarray(yp).ravel(),np.asarray(yt).ravel()
            k=np.asarray((dts.month>=5)&(dts.month<=10)&(dts.year<=2023))
            for g,msk in [("ALL",k)]+[(y,k&np.asarray(dts.year==y)) for y in range(2017,2024)]:
                if msk.sum()<20: continue
                a_,b_=yt[msk],yp[msk]
                rows.append(dict(arm=arm,site=fold["test"],seed=seed,year=g,
                                 r2=1-((a_-b_)**2).sum()/((a_-a_.mean())**2).sum(),
                                 rmse=float(np.sqrt(((a_-b_)**2).mean())),
                                 offset=float((a_-b_).mean()),n=int(msk.sum())))
    print(f"  {arm}: done",flush=True)
install(False)
R=pd.DataFrame(rows); R.to_csv(MV/"all_field_scored.csv",index=False)
print(f"\nscored -> {MV/'all_field_scored.csv'} ({len(R)} rows)")
