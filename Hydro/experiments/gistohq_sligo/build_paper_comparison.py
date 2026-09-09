#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, math
from pathlib import Path

MODES=("ffn","ffn_pinn","lstm","lstm_pinn","pinn")
LABEL={"ffn":"FFN","ffn_pinn":"FFN + PINN","lstm":"LSTM","lstm_pinn":"LSTM + PINN","pinn":"PINN"}

def read(path):
    with Path(path).open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))

def by_mode(rows): return {r['mode']:r for r in rows}

def write(path, rows):
    keys=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen: keys.append(k); seen.add(k)
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

def fmt(x, nd=4):
    try:
        v=float(x)
        return f"{v:.{nd}f}" if math.isfinite(v) else "--"
    except: return "--"

def fmt_kge(value, status='', defined='', total=''):
    try:
        v=float(value)
        if math.isfinite(v):
            suffix=''
            try:
                d=int(float(defined)); n=int(float(total))
                if 0 < d < n: suffix=f" ({d}/{n} seeds)"
            except: pass
            return f"{v:.4f}{suffix}"
    except: pass
    if 'constant_prediction' in status:
        return "N/A (near-constant prediction)"
    return "N/A (undefined)"

def main():
    p=argparse.ArgumentParser(); p.add_argument('paper_root',type=Path); a=p.parse_args(); root=a.paper_root.resolve()
    syn=by_mode(read(root/'01_synthetic_controlled'/'paper_method_summary.csv'))
    hyd=by_mode(read(root/'02_sligo_hydro'/'paper_method_summary.csv'))
    known_path=root/'01_synthetic_controlled'/'paper_synthetic_known_truth_summary.csv'
    known=by_mode(read(known_path)) if known_path.exists() else {}
    rows=[]
    for mode in MODES:
        s=syn[mode]; h=hyd[mode]; k=known.get(mode,{})
        rows.append({
            'method':LABEL[mode],'mode':mode,
            'hidden_layers':h.get('hidden_layers',''),'activation':h.get('activation',''),
            'input_lags':h.get('input_lags',''),'lstm_sequence_length':h.get('lstm_sequence_length',''),
            'learning_rate':h.get('learning_rate',''),'batch_size':h.get('batch_size',''),
            'physics_weight':h.get('physics_weight',''),'reservoir_k':h.get('reservoir_k',''),
            'synthetic_whole_rmse_mean':k.get('whole_rmse_mean',''), 'synthetic_whole_rmse_std':k.get('whole_rmse_std',''),
            'synthetic_whole_nse_mean':k.get('whole_nse_mean',''), 'synthetic_whole_kge_mean':k.get('whole_kge_mean',''),
            'synthetic_whole_pbias_mean':k.get('whole_pbias_mean',''),
            'sligo_rmse_mean':h.get('rmse_mean',''),'sligo_rmse_std':h.get('rmse_std',''),
            'sligo_mae_mean':h.get('mae_mean',''),'sligo_nse_mean':h.get('nse_mean',''),
            'sligo_kge_mean':h.get('kge_mean',''),'sligo_kge_std':h.get('kge_std',''),
            'sligo_seed_count':h.get('seed_count',''),
            'sligo_kge_defined_seed_count':h.get('kge_defined_seed_count',''),
            'sligo_correlation_defined_seed_count':h.get('correlation_defined_seed_count',''),
            'sligo_kge_status':h.get('kge_status',''),
            'sligo_prediction_near_constant_seed_count':h.get('prediction_near_constant_seed_count',''),
            'sligo_prediction_test_std_mean':h.get('prediction_test_std_mean',''),
            'sligo_pbias_mean':h.get('pbias_mean',''),
            'sligo_physics_residual_rmse_mean':h.get('physics_residual_rmse_mean',''),
        })
    write(root/'paper_final_method_comparison.csv',rows)
    md=["# HydroPINN final paper tables","", "Selection is validation-based; Stage-4 values are five-seed robustness results.","",
        "## Controlled reduced-reservoir validation","",
        "| Method | Whole-domain RMSE | NSE | KGE | PBIAS (%) |","|---|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['method']} | {fmt(r['synthetic_whole_rmse_mean'])} ± {fmt(r['synthetic_whole_rmse_std'])} | {fmt(r['synthetic_whole_nse_mean'])} | {fmt(r['synthetic_whole_kge_mean'])} | {fmt(r['synthetic_whole_pbias_mean'],2)} |")
    md += ["","## Sligo Creek Hydro package","", "| Method | RMSE | MAE | NSE | KGE | PBIAS (%) |","|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['method']} | {fmt(r['sligo_rmse_mean'])} ± {fmt(r['sligo_rmse_std'])} | {fmt(r['sligo_mae_mean'])} | {fmt(r['sligo_nse_mean'])} | {fmt_kge(r['sligo_kge_mean'],r['sligo_kge_status'],r['sligo_kge_defined_seed_count'],r['sligo_seed_count'])} | {fmt(r['sligo_pbias_mean'],2)} |")
    md += ["","KGE is not imputed when Pearson correlation is undefined. Parenthetical seed counts indicate that the reported KGE mean is based only on seeds for which KGE is mathematically defined.","",
           "### Sligo metric-definition diagnostics","",
           "| Method | KGE defined | Near-constant prediction seeds | Mean test prediction SD | Status |","|---|---:|---:|---:|---|"]
    for r in rows:
        md.append(f"| {r['method']} | {r['sligo_kge_defined_seed_count']}/{r['sligo_seed_count']} | {r['sligo_prediction_near_constant_seed_count']}/{r['sligo_seed_count']} | {fmt(r['sligo_prediction_test_std_mean'],8)} | {r['sligo_kge_status']} |")
    md += ["","## Frozen hyperparameters","", "| Method | Hidden | Activation | Lags | Sequence | LR | Batch | Physics w | k |","|---|---|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['method']} | {r['hidden_layers']} | {r['activation']} | {r['input_lags']} | {r['lstm_sequence_length']} | {r['learning_rate']} | {r['batch_size']} | {r['physics_weight']} | {r['reservoir_k']} |")
    (root/'paper_final_tables.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(root/'paper_final_method_comparison.csv'); print(root/'paper_final_tables.md')
if __name__=='__main__': main()
