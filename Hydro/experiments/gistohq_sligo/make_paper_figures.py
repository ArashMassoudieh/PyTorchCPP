#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv
from pathlib import Path
try:
    import matplotlib.pyplot as plt
except Exception as e:
    raise SystemExit(f"matplotlib is required for paper figures: {e}")
MODES=("ffn","ffn_pinn","lstm","lstm_pinn","pinn")
LABEL={"ffn":"FFN","ffn_pinn":"FFN + PINN","lstm":"LSTM","lstm_pinn":"LSTM + PINN","pinn":"PINN"}

def read(path):
    with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def seed42(stage4,mode):
    rows=[r for r in read(stage4/'batch_summary.csv') if r['mode']==mode]
    return next((r for r in rows if r.get('random_seed')=='42'),rows[0])
def pred(stage4,mode):
    r=seed42(stage4,mode); return read(stage4/mode/r['experiment_id']/'predictions.csv')
def plot_predictions(root,subdir,filename,title,split=None):
    stage4=root/subdir/'04_stage4_robustness'; first=pred(stage4,'ffn')
    if split: first=[r for r in first if r['split']==split]
    x=[float(r['x']) for r in first]; y=[float(r['observed']) for r in first]
    fig,ax=plt.subplots(figsize=(9,4.8)); ax.plot(x,y,label='Observed / truth',linewidth=2.2)
    for mode in MODES:
        rows=pred(stage4,mode)
        if split: rows=[r for r in rows if r['split']==split]
        ax.plot([float(r['x']) for r in rows],[float(r['predicted']) for r in rows],label=LABEL[mode],linewidth=1.3)
    ax.set_xlabel('Time'); ax.set_ylabel('Runoff'); ax.set_title(title); ax.legend(ncol=3,fontsize=8); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(root/filename,dpi=300); plt.close(fig)
def plot_rmse(root):
    rows=read(root/'02_sligo_hydro'/'paper_method_summary.csv'); labels=[LABEL[r['mode']] for r in rows]; means=[float(r['rmse_mean']) for r in rows]; errs=[float(r['rmse_std']) for r in rows]
    fig,ax=plt.subplots(figsize=(7.5,4.8)); ax.bar(labels,means,yerr=errs,capsize=4); ax.set_ylabel('RMSE'); ax.set_title('Sligo Creek five-seed robustness'); ax.tick_params(axis='x',rotation=20); ax.grid(axis='y',alpha=.2); fig.tight_layout(); fig.savefig(root/'paper_sligo_rmse_robustness.png',dpi=300); plt.close(fig)
def plot_k_recovery(root):
    s2=root/'01_synthetic_controlled'/'02_stage2_physics'; rows=[r for r in read(s2/'batch_summary.csv') if r['mode']=='pinn']; groups={}
    for r in rows:
        k=float(r['latent_recession_per_hour']); groups.setdefault(k,[]).append(float(r['validation_mse']))
    ks=sorted(groups); vals=[min(groups[k]) for k in ks]
    fig,ax=plt.subplots(figsize=(6.5,4.5)); ax.plot(ks,vals,marker='o'); ax.axvline(.08,linestyle='--',label='Synthetic truth k=0.08'); ax.set_xlabel('Candidate reservoir k'); ax.set_ylabel('Validation MSE'); ax.set_title('Standalone PINN known-parameter recovery'); ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(root/'paper_synthetic_k_recovery.png',dpi=300); plt.close(fig)
def main():
    p=argparse.ArgumentParser(); p.add_argument('paper_root',type=Path); a=p.parse_args(); root=a.paper_root.resolve()
    plot_predictions(root,'01_synthetic_controlled','paper_synthetic_known_truth_predictions.png','Controlled reduced-reservoir known-truth validation')
    plot_predictions(root,'02_sligo_hydro','paper_sligo_test_predictions.png','Sligo Creek held-out test predictions',split='test')
    plot_rmse(root); plot_k_recovery(root)
    print('[paper-figures] wrote 4 figures to',root)
if __name__=='__main__':main()
