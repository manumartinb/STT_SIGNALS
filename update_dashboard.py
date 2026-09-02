"""
Build STT REGIME dashboard (v4: IV_CONV / IV ATM [eje vol, ex-VIX] / PUT SKEW NIVEL)
- 3 senales como percentil expanding 0-100 (ex-ante)
- Estudios: solas, 2-a-2, 3-a-3. Todas las tablas/charts dual mean+median.
- data.json + evidence/*.png
"""
import pandas as pd, numpy as np, sys, os, json
from bisect import bisect_right, insort
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding='utf-8')

OUTDIR = r'C:/Users/Administrator/Desktop/BULK OPTIONSTRAT/ESTRATEGIAS/Skew/dashboards/STT_REGIME_DASHBOARD'
EVDIR  = os.path.join(OUTDIR, 'evidence')
os.makedirs(EVDIR, exist_ok=True)

PATH = r'C:/Users/Administrator/Desktop/Backtests DATABASE/STT/STT_CLASSIC_V9_MERGED_T0_mediana.csv'
VIX_PATH = r'C:/Users/Administrator/Desktop/FINAL DATA/VIX_CLOSE_HISTORICAL_PRICES.csv'
PS_PATH  = r'C:/Users/Administrator/Desktop/BULK OPTIONSTRAT/ESTRATEGIAS/Skew/SKEW_PUT_ENRICHED.csv'
# Parquets 30MIN crudos (solo lectura) para sanar dias glitch r=0 (ver stt_heal.py)
PARQUET_DIR = r'C:/Users/Administrator/Desktop/FINAL DATA/HIST AND STREAMING DATA/UPDATED HISTORICAL DAYS PARQUET'
PS_REF_NPZ  = os.path.join(OUTDIR, 'PS_SKEW_REF_FROZEN.npz')  # referencia FIJA de PUT SKEW (bloque [3])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stt_heal

pnl_cols = [f'PnL_d{i:03d}' for i in range(1, 31)]
xs = list(range(1, 31))

def expanding_pct(series_values, warmup=30):
    acc=[]; out=np.full(len(series_values), np.nan)
    for i,v in enumerate(series_values):
        if not pd.isna(v):
            if len(acc)>=warmup: out[i]=100.0*bisect_right(acc,float(v))/len(acc)
            insort(acc,float(v))
    return out

def smooth3(s):
    # Mediana movil trailing 3d (hoy + 2 dias previos), ex-ante (sin lookahead).
    # Aplica al RAW de las senales day-level de mercado antes de percentilizar, para
    # matar outliers de 1 dia en la fuente (ej. 29-may: iv_50d cae 16% y revierte).
    # Validado por APR: r se mantiene, cola (PF) mejora. DEBE ser identica en daily_refresh.py.
    return pd.Series(s).rolling(3, min_periods=1).median().values

print('[1] Loading STT ...', flush=True)
# IV_CONVEXITY (trade-specific) ya NO se lee aqui: vive en la madre y en el LIVE.
df = pd.read_csv(PATH, usecols=['dia','SPX']+pnl_cols, low_memory=False)
df['dia'] = pd.to_datetime(df['dia']).dt.normalize()
df['year'] = df['dia'].dt.year
df = df.sort_values('dia').reset_index(drop=True)

# ============================================================================
# CONFIG 2026-09-02 (auditoria @AuditoriaManumb + @APR de ventanas). 3 cambios medidos:
#  1) IV_CONV pasa a la version GENERICA de mercado (iv_5d/iv_15d/iv_30d @dte160)
#     TAMBIEN en las tablas. Antes las tablas usaban la trade-specific
#     (iv_k1+iv_k3)/2-iv_k2 y el panel el proxy: el lector veia una LUZ y una
#     ESTADISTICA de variables distintas (Jaccard 0.68 en el umbral P80, y el
#     edge desplegable era +1.40 frente al +1.83 publicado).
#     La trade-specific NO se pierde: vive donde SI hay trade -- madre
#     STT_CLASSIC_V9_MERGED_T0_mediana.csv (col IV_CONVEXITY, 100% cobertura) y
#     el LIVE "STT V1 LIVE ... SQI V2" (compute_sqi_v2_live -> SQI_V2_PCTLE).
#     Medido: el generico rinde MEJOR en percentil que en raw
#     (bootstrap-by-day raw - pct = -0.0556, CI95 [-0.1015, -0.0094], SIG).
#  2) PUT SKEW pasa a REFERENCIA FIJA congelada (searchsorted) en vez de percentil
#     expanding. Es la convencion LIVE del proyecto y es una transformacion
#     MONOTONA del raw, asi que hereda su estadistica exacta: el @APR mostro que
#     el raw BATE al expanding (+0.0631, CI95 [+0.0418, +0.0864], SIG) y que el
#     expanding RESTA (partial(exp|raw) = -0.128; k-fold 5/5 vs 4/5).
#  3) NO se cablean ventanas rolling (504/252/126): las 12 comparaciones
#     rolling-vs-expanding pierden, 11 de ellas de forma significativa.
# Ver memory/analisis_predictabilidad_robustez_ventanas_stt_20260902.md
# ============================================================================
print('[2] Senales de mercado @dte160 (IV ATM, IV_CONV generico, PUT SKEW) ...', flush=True)
psf = pd.read_csv(PS_PATH, usecols=['trade_date','dte_target','skew_25d_vs50',
                                    'iv_5d','iv_15d','iv_30d','iv_50d'])
psf['trade_date'] = pd.to_datetime(psf['trade_date']).dt.normalize()
psf160 = psf[psf['dte_target']==160].copy().sort_values('trade_date').reset_index(drop=True)

# --- EJE VOL = ATM-IV (iv_50d). Clave JSON legacy 'vix_*' por contrato con index.html.
atm = psf160[['trade_date','iv_50d']].rename(columns={'trade_date':'dia','iv_50d':'VIX_Close'}).dropna().sort_values('dia').reset_index(drop=True)
atm['vix_pct'] = expanding_pct(smooth3(atm['VIX_Close'].values))
df = df.merge(atm[['dia','VIX_Close','vix_pct']], on='dia', how='left')

# --- IV_CONV GENERICO: el MISMO calculo que el panel (una variable, una definicion)
gen = psf160[['trade_date','iv_5d','iv_15d','iv_30d']].rename(columns={'trade_date':'dia'}).copy()
gen['ivc_gen_raw'] = smooth3(((gen['iv_5d']+gen['iv_30d'])/2 - gen['iv_15d']).values)
gen = gen.dropna(subset=['ivc_gen_raw']).sort_values('dia').reset_index(drop=True)
gen['ivc_pct'] = expanding_pct(gen['ivc_gen_raw'].values)
df = df.merge(gen[['dia','ivc_gen_raw','ivc_pct']], on='dia', how='left')

# --- PUT SKEW NIVEL: referencia FIJA congelada (searchsorted), NO expanding.
print('[3] PUT SKEW @dte160 contra referencia FIJA congelada ...', flush=True)
ps = psf160[['trade_date','skew_25d_vs50']].rename(columns={'trade_date':'dia'}).dropna().sort_values('dia').reset_index(drop=True)
ps['ps_raw'] = smooth3(ps['skew_25d_vs50'].values)
if os.path.isfile(PS_REF_NPZ):
    _z = np.load(PS_REF_NPZ, allow_pickle=True)
    PS_REF = _z['ref_sorted']
    print('    referencia FIJA cargada: n=%d (congelada el %s)' % (len(PS_REF), str(_z['built'])))
else:
    PS_REF = np.sort(ps['ps_raw'].dropna().values)
    np.savez(PS_REF_NPZ, ref_sorted=PS_REF, built=str(ps['dia'].max().date()),
             note='PUT SKEW skew_25d_vs50 @dte160 suavizado 3d. Referencia CONGELADA para searchsorted.')
    print('    referencia FIJA creada: n=%d (hasta %s)' % (len(PS_REF), ps['dia'].max().date()))
ps['ps_pct'] = np.searchsorted(PS_REF, ps['ps_raw'].values, side='right') / float(len(PS_REF)) * 100.0
ps.loc[ps['ps_raw'].isna(), 'ps_pct'] = np.nan
PS_THR = dict((k, float(np.percentile(PS_REF, k))) for k in (10, 20, 30, 70, 80, 90))
print('    umbrales RAW equivalentes: P20=%.5f  P80=%.5f  P90=%.5f'
      % (PS_THR[20], PS_THR[80], PS_THR[90]))
df = df.merge(ps[['dia','ps_raw','ps_pct']], on='dia', how='left')

# Universo canonico: las 3 senales presentes
sub = df.dropna(subset=['ivc_pct','vix_pct','ps_pct']).reset_index(drop=True)
print('    N=%d  dias=%d  ps_cov=%.1f%%' % (len(sub), sub['dia'].dt.date.nunique(),
                                            df['ps_pct'].notna().mean()*100))

RAW_AVG_MEAN = sub[pnl_cols].mean().mean()
RAW_AVG_MED  = sub[pnl_cols].median().mean()

SIGS = {'ivc': ('ivc_pct','IV_CONV'), 'vix': ('vix_pct','IV ATM'), 'ps': ('ps_pct','PUT SKEW')}

def banda(p):
    if p>=80: return 'FAVORABLE'
    if p<=20: return 'ADVERSO'
    return 'NEUTRAL'

def cohort(d, label):
    if len(d)==0: return None
    pos=d['PnL_d030'][d['PnL_d030']>0].sum(); neg=abs(d['PnL_d030'][d['PnL_d030']<0].sum())
    pf=pos/neg if neg>0 else float('inf')
    # by_year: reparto por anio de la cohorte. Sin esto, un N=27 con 26 trades de 2020
    # es indistinguible de un N=27 repartido -> el lector no puede juzgar la cohorte.
    by_year={str(int(y)):int(c) for y,c in d['dia'].dt.year.value_counts().sort_index().items()}
    return {'label':label,'n':int(len(d)),'days':int(d['dia'].dt.date.nunique()),'by_year':by_year,
        'd20_mean':round(d['PnL_d020'].mean(),2),'d20_med':round(d['PnL_d020'].median(),2),
        'd30_mean':round(d['PnL_d030'].mean(),2),'d30_med':round(d['PnL_d030'].median(),2),
        'avg_mean':round(d[pnl_cols].mean().mean(),2),'avg_med':round(d[pnl_cols].median().mean(),2),
        'wr30':round((d['PnL_d030']>0).mean()*100,1),
        'pf30':round(pf,2) if pf!=float('inf') else 999}

# ---------- daily series + latest ----------
daily = sub.groupby('dia').agg(ivc_pct=('ivc_pct','mean'), vix_pct=('vix_pct','mean'),
                               ps_pct=('ps_pct','mean'), vix=('VIX_Close','mean'),
                               iv_conv_raw=('ivc_gen_raw','mean')).reset_index().sort_values('dia')
latest = daily.iloc[-1]

data = {'latest': {'date': latest['dia'].strftime('%Y-%m-%d'),
                   'ivc_pct': float(latest['ivc_pct']), 'regime_ivc': banda(latest['ivc_pct']),
                   'vix_pct': float(latest['vix_pct']), 'regime_vix': banda(latest['vix_pct']),
                   'ps_pct': float(latest['ps_pct']),  'regime_ps': banda(latest['ps_pct']),
                   'vix_raw': float(latest['vix']), 'ivc_raw': float(latest['iv_conv_raw'])},
        'series': [{'t': r['dia'].strftime('%Y-%m-%d'),
                    'ivc': round(float(r['ivc_pct']),2), 'vix': round(float(r['vix_pct']),2),
                    'ps': round(float(r['ps_pct']),2), 'vraw': round(float(r['vix']),2)} for _,r in daily.iterrows()],
        'meta': {'dataset':'STT_CLASSIC_V9_MERGED_T0_mediana','ivc_source':'generico de mercado (iv_5d+iv_30d)/2-iv_15d @dte160','ps_method':'referencia FIJA congelada (searchsorted)','ps_thr_raw':{str(k):round(v,5) for k,v in PS_THR.items()},'n_trades':int(len(sub)),
                 'n_days':int(sub['dia'].dt.date.nunique()),'date_min':sub['dia'].min().strftime('%Y-%m-%d'),
                 'date_max':sub['dia'].max().strftime('%Y-%m-%d')},
        'baseline': {'n_trades':len(sub),'n_days':int(sub['dia'].dt.date.nunique()),
                     'mean_d001_d030':round(RAW_AVG_MEAN,3),'med_d001_d030':round(RAW_AVG_MED,3),
                     'mean_d020':round(sub['PnL_d020'].mean(),2),'med_d020':round(sub['PnL_d020'].median(),2),
                     'mean_d030':round(sub['PnL_d030'].mean(),2),'med_d030':round(sub['PnL_d030'].median(),2),
                     'wr_d030':round((sub['PnL_d030']>0).mean()*100,1)}}

# ---------- SOLO tables ----------
def solo_table(col, name):
    t=[cohort(sub,'RAW (universo)')]
    for thr,lbl in [(70,f'TOP30 ({name}>=P70)'),(80,f'TOP20 ({name}>=P80)'),(90,f'TOP10 ({name}>=P90)')]:
        t.append(cohort(sub[sub[col]>=thr], lbl))
    for thr,lbl in [(30,f'BOT30 ({name}<=P30)'),(20,f'BOT20 ({name}<=P20)'),(10,f'BOT10 ({name}<=P10)')]:
        t.append(cohort(sub[sub[col]<=thr], lbl))
    return t
data['tbl_ivc']=solo_table('ivc_pct','IVC')
data['tbl_vix']=solo_table('vix_pct','IVATM')
data['tbl_ps'] =solo_table('ps_pct','PS')

# ---------- 2x2 joint (terciles) ----------
TERC=[('bajo (P<=33)',0,33),('medio (33-66)',33,66),('alto (>=P66)',66,100)]
def joint_table(colA,colB,labA,labB):
    t=[]
    for la,alo,ahi in TERC:
        for lb,blo,bhi in TERC:
            ss=sub[(sub[colA]>=alo)&(sub[colA]<=ahi)&(sub[colB]>=blo)&(sub[colB]<=bhi)]
            if len(ss)>=20: t.append(cohort(ss,f'{labA} {la} & {labB} {lb}'))
    return t
data['tbl_joint_iv']=joint_table('ivc_pct','vix_pct','IVC','IV ATM')
data['tbl_joint_ip']=joint_table('ivc_pct','ps_pct','IVC','PS')
data['tbl_joint_vp']=joint_table('vix_pct','ps_pct','IV ATM','PS')

# ---------- 2x2 composite (gold + iron) ----------
def combo_table(colA,colB,labA,labB):
    tA=sub[sub[colA]>=80]; tB=sub[sub[colB]>=80]
    tAND=sub[(sub[colA]>=80)&(sub[colB]>=80)]; tOR=sub[(sub[colA]>=80)|(sub[colB]>=80)]
    bAND=sub[(sub[colA]<=20)&(sub[colB]<=20)]; bOR=sub[(sub[colA]<=20)|(sub[colB]<=20)]
    return [cohort(sub,'RAW universo'),
            cohort(tA,f'TOP20 {labA} solo'), cohort(tB,f'TOP20 {labB} solo'),
            cohort(tAND,f'AND TOP20 {labA}+{labB} (ORO)'), cohort(tOR,f'OR TOP20 {labA}|{labB}'),
            cohort(bAND,f'AND BOT20 {labA}+{labB} (HIERRO)'), cohort(bOR,f'OR BOT20 {labA}|{labB}')]
data['tbl_combo_iv']=combo_table('ivc_pct','vix_pct','IVC','IV ATM')
data['tbl_combo_ip']=combo_table('ivc_pct','ps_pct','IVC','PS')
data['tbl_combo_vp']=combo_table('vix_pct','ps_pct','IV ATM','PS')

# ---------- 3x3 triple ----------
top = {k:(sub[SIGS[k][0]]>=80) for k in SIGS}
bot = {k:(sub[SIGS[k][0]]<=20) for k in SIGS}
n_top = top['ivc'].astype(int)+top['vix'].astype(int)+top['ps'].astype(int)
n_bot = bot['ivc'].astype(int)+bot['vix'].astype(int)+bot['ps'].astype(int)
data['tbl_triple']=[
    cohort(sub,'RAW universo'),
    cohort(sub[top['ivc']],'TOP20 IVC solo'),
    cohort(sub[top['vix']],'TOP20 IV ATM solo'),
    cohort(sub[top['ps']], 'TOP20 PS solo'),
    cohort(sub[top['ivc']&top['vix']&top['ps']],'AND TOP20 las 3 (ORO TRIPLE)'),
    cohort(sub[n_top>=2],'al menos 2 de 3 TOP20'),
    cohort(sub[n_top>=1],'OR TOP20 cualquiera'),
    cohort(sub[bot['ivc']&bot['vix']&bot['ps']],'AND BOT20 las 3 (HIERRO TRIPLE)'),
]

# ---------- stats ----------
def sp(a,b):
    m=sub.dropna(subset=[a,b]); return round(float(m[a].rank().corr(m[b].rank())),4)
data['stats']={
    'corr_ivc_vix':sp('ivc_pct','vix_pct'),'corr_ivc_ps':sp('ivc_pct','ps_pct'),'corr_vix_ps':sp('vix_pct','ps_pct'),
    'r_ivc_pnl':sp('ivc_pct','PnL_d030'),'r_vix_pnl':sp('vix_pct','PnL_d030'),'r_ps_pnl':sp('ps_pct','PnL_d030'),
    'vix_method':'expanding','vix_gate_r_exp':sp('vix_pct','PnL_d030')}

# ===== GRAFICO/PANEL = REGIMEN DIARIO A HOY (decoupled del estudio backtest) =====
# Las tablas/stats de arriba quedan FIJAS (backtest STT, IV_CONV trade-specific).
# El grafico + panel 'latest' se actualizan a la ultima fecha de mercado disponible
# en SKEW_PUT_ENRICHED @dte160 (como el dashboard PUT_SKEW_NIVEL_BATMAN_LT):
#   IV ATM  = iv_50d                         -> smooth3 -> expanding_pct
#   PUT SKEW = skew_25d_vs50 (raw)           -> smooth3 -> expanding_pct (recalc local)
#   IV_CONV = proxy (iv_5d+iv_30d)/2 - iv_15d -> smooth3 -> expanding_pct  [rho +0.84 con trade-specific]
# Las 3 senales se suavizan 3d (ex-ante) antes de percentilizar. BLOQUE IDENTICO a daily_refresh.py.
print('[6] Serie diaria regimen-a-hoy (chart/panel) SUAVIZADO 3d + HEAL r=0 ...', flush=True)
dpsf = pd.read_csv(PS_PATH, usecols=['trade_date','dte_target','iv_5d','iv_15d','iv_25d','iv_30d','iv_50d',
                                     'skew_25d_vs50','underlying_price','strike_hit_50d','expiration_used'])
dpsf['dia'] = pd.to_datetime(dpsf['trade_date']).dt.normalize()
dpsf = dpsf[dpsf['dte_target']==160].sort_values('dia').reset_index(drop=True)
# SANEAR dias glitch r=0 (forward colapsado) desde parquet crudo con tasa real. No-op si no hay glitch.
dpsf = stt_heal.heal(dpsf, PARQUET_DIR, log=lambda m: print('    '+m, flush=True))
dpsf['ivc_proxy_raw'] = (dpsf['iv_5d']+dpsf['iv_30d'])/2 - dpsf['iv_15d']
dpsf['ivc_d'] = expanding_pct(smooth3(dpsf['ivc_proxy_raw'].values))
dpsf['atm_d'] = expanding_pct(smooth3(dpsf['iv_50d'].values))
# PUT SKEW: MISMA referencia FIJA que las tablas (searchsorted, no expanding).
# Asi el panel y las tablas usan exactamente la misma definicion.
_ps_raw_d = smooth3(dpsf['skew_25d_vs50'].values)
dpsf['ps_d'] = np.where(np.isnan(_ps_raw_d), np.nan,
                        np.searchsorted(PS_REF, _ps_raw_d, side='right') / float(len(PS_REF)) * 100.0)
# SQI_V2 PROXY diario (trade-level score aproximado con datos de mercado):
#   comp1 proxy = iv_25d @160 (rho +0.910 con THETA_OVER_SPOT diario)
#   comp3 proxy = ivc_proxy (rho +0.844). Proxy total: rho +0.848 con SQI_V2 trade,
#   r +0.425 vs PnL_d030 diario. Mismos pesos congelados 0.57/0.43.
dpsf['sqi_d'] = 0.57*expanding_pct(smooth3(dpsf['iv_25d'].values)) + 0.43*dpsf['ivc_d']
# SPX close (cotizacion) para el eje secundario del grafico
_spx = pd.read_csv(r'C:/Users/Administrator/Desktop/FINAL DATA/SP_SPX_CLOSE_HISTORICAL_PRICES.csv', usecols=['time','close'])
_spx['dia'] = pd.to_datetime(_spx['time']).dt.normalize()
_spx = _spx.rename(columns={'close':'spx_close'})[['dia','spx_close']]
dpsf = dpsf.merge(_spx, on='dia', how='left')
dser = dpsf.dropna(subset=['ivc_d','atm_d','ps_d','sqi_d']).reset_index(drop=True)
data['series'] = [{'t':r['dia'].strftime('%Y-%m-%d'),'ivc':round(float(r['ivc_d']),2),
                   'vix':round(float(r['atm_d']),2),'ps':round(float(r['ps_d']),2),
                   'sqi':round(float(r['sqi_d']),2),
                   'spx':(round(float(r['spx_close']),2) if pd.notna(r['spx_close']) else None)} for _,r in dser.iterrows()]
_last = dser.iloc[-1]

def _last_fav(frame, col):
    """Ultima fecha en que la senal marco FAVORABLE (>=80). Una luz que lleva anios
    sin encenderse es indistinguible de una que hoy esta en rojo: esto lo hace visible."""
    f = frame[frame[col] >= 80]
    return f.iloc[-1]['dia'].strftime('%Y-%m-%d') if len(f) else None

data['latest'] = {'date':_last['dia'].strftime('%Y-%m-%d'),
    'last_fav_ivc':_last_fav(dser,'ivc_d'), 'last_fav_vix':_last_fav(dser,'atm_d'),
    'last_fav_ps':_last_fav(dser,'ps_d'),   'last_fav_sqi':_last_fav(dser,'sqi_d'),
    'ivc_pct':float(_last['ivc_d']),'regime_ivc':banda(_last['ivc_d']),
    'vix_pct':float(_last['atm_d']),'regime_vix':banda(_last['atm_d']),
    'ps_pct':float(_last['ps_d']),'regime_ps':banda(_last['ps_d']),
    'sqi_pct':float(_last['sqi_d']),'regime_sqi':banda(_last['sqi_d']),
    'vix_raw':float(_last['iv_50d']),'ivc_raw':float(_last['ivc_proxy_raw'])}
data['meta']['chart_date_max'] = _last['dia'].strftime('%Y-%m-%d')
data['meta']['chart_n_days']  = int(len(dser))
print(f'    serie diaria: {len(dser)} dias, ultima {data["latest"]["date"]}')

with open(os.path.join(OUTDIR,'data.json'),'w') as f: json.dump(data,f,indent=2)
print('[4] data.json saved')

# ================== CHARTS ==================
plt.style.use('dark_background'); COL_BG='#0d1117'; COL_PAN='#161b22'
GREENS=['#3fb950','#58e078','#a3e635']; REDS=['#fcad6a','#f85149','#c0282f']

def traj_dual(col, name, fname):
    fig,axes=plt.subplots(1,2,figsize=(16,6.5),dpi=120,facecolor=COL_BG)
    cuts=[(90,f'TOP10 (>=P90)',GREENS[0]),(80,'TOP20 (>=P80)',GREENS[1]),(70,'TOP30 (>=P70)',GREENS[2]),
          (30,'BOT30 (<=P30)',REDS[0]),(20,'BOT20 (<=P20)',REDS[1]),(10,'BOT10 (<=P10)',REDS[2])]
    for ax,agg,ttl in [(axes[0],'mean','MEAN'),(axes[1],'median','MEDIAN')]:
        ax.set_facecolor(COL_PAN); aggf=(lambda s:s.mean()) if agg=='mean' else (lambda s:s.median())
        ax.plot(xs,[aggf(sub[f'PnL_d{i:03d}']) for i in xs],color='white',linewidth=2.0,label=f'RAW N={len(sub)}')
        for thr,lbl,c in cuts:
            s=sub[sub[col]>=thr] if 'TOP' in lbl else sub[sub[col]<=thr]
            ax.plot(xs,[aggf(s[f'PnL_d{i:03d}']) for i in xs],color=c,linewidth=1.3,label=f'{lbl} N={len(s)}')
        ax.axhline(0,color='gray',linewidth=0.5,alpha=0.5); ax.grid(alpha=0.2)
        ax.set_xlabel('dia (d001..d030)'); ax.set_ylabel(f'{ttl} PnL acum (pts)')
        ax.set_title(f'STT - PnL por corte {name} [{ttl}]'); ax.legend(loc='upper left',fontsize=7.5)
    fig.tight_layout(); fig.savefig(os.path.join(EVDIR,fname),facecolor=COL_BG); plt.close(fig)
    print(f'    [chart] {fname}')

def heat_dual(colA,colB,labA,labB,fname):
    fig,axes=plt.subplots(1,2,figsize=(16,6.6),dpi=120,facecolor=COL_BG)
    rng=[(0,33),(33,66),(66,100)]; lab=['bajo(P<=33)','medio','alto(>=P66)']
    for ax,agg,ttl in [(axes[0],'mean','MEAN'),(axes[1],'median','MEDIAN')]:
        ax.set_facecolor(COL_PAN); mat=np.full((3,3),np.nan); nmat=np.zeros((3,3),dtype=int)
        for i,(alo,ahi) in enumerate(rng):
            for j,(blo,bhi) in enumerate(rng):
                ss=sub[(sub[colA]>=alo)&(sub[colA]<=ahi)&(sub[colB]>=blo)&(sub[colB]<=bhi)]
                if len(ss)>=20:
                    mat[i,j]=ss['PnL_d030'].mean() if agg=='mean' else ss['PnL_d030'].median(); nmat[i,j]=len(ss)
        im=ax.imshow(mat,cmap='RdYlGn',vmin=-10,vmax=10,aspect='auto')
        for i in range(3):
            for j in range(3):
                if not np.isnan(mat[i,j]): ax.text(j,i,f'{mat[i,j]:+.1f}\nN={nmat[i,j]}',ha='center',va='center',fontsize=10,color='black',fontweight='bold')
        ax.set_xticks(range(3)); ax.set_xticklabels([f'{labB} {x}' for x in lab],fontsize=8)
        ax.set_yticks(range(3)); ax.set_yticklabels([f'{labA} {x}' for x in lab],fontsize=8)
        ax.set_title(f'STT - PnL_d030 {labA} x {labB} [{ttl}]'); plt.colorbar(im,ax=ax)
    fig.tight_layout(); fig.savefig(os.path.join(EVDIR,fname),facecolor=COL_BG); plt.close(fig)
    print(f'    [chart] {fname}')

def combo_dual(colA,colB,labA,labB,fname):
    fig,axes=plt.subplots(1,2,figsize=(16,6),dpi=120,facecolor=COL_BG)
    tA=sub[sub[colA]>=80]; tB=sub[sub[colB]>=80]
    tAND=sub[(sub[colA]>=80)&(sub[colB]>=80)]; tOR=sub[(sub[colA]>=80)|(sub[colB]>=80)]
    bAND=sub[(sub[colA]<=20)&(sub[colB]<=20)]
    for ax,agg,ttl in [(axes[0],'mean','MEAN'),(axes[1],'median','MEDIAN')]:
        ax.set_facecolor(COL_PAN); aggf=(lambda s:s.mean()) if agg=='mean' else (lambda s:s.median())
        def line(s,c,lbl,lw=1.4,ls='-'):
            if len(s)>0: ax.plot(xs,[aggf(s[f'PnL_d{i:03d}']) for i in xs],color=c,linewidth=lw,linestyle=ls,label=f'{lbl} N={len(s)}')
        line(sub,'white','RAW',2.0)
        line(tA,'#58a6ff',f'TOP20 {labA}'); line(tB,'#ff9f43',f'TOP20 {labB}')
        line(tAND,'#3fb950',f'AND (ORO)',2.0); line(tOR,'#bc8cff','OR',1.2,'--'); line(bAND,'#f85149','AND BOT (HIERRO)',1.6)
        ax.axhline(0,color='gray',linewidth=0.5,alpha=0.5); ax.grid(alpha=0.2)
        ax.set_xlabel('dia (d001..d030)'); ax.set_ylabel(f'{ttl} PnL acum (pts)')
        ax.set_title(f'STT - {labA} x {labB} [{ttl}]'); ax.legend(loc='upper left',fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(EVDIR,fname),facecolor=COL_BG); plt.close(fig)
    print(f'    [chart] {fname}')

def triple_dual(fname):
    fig,axes=plt.subplots(1,2,figsize=(16,6),dpi=120,facecolor=COL_BG)
    AND3=sub[top['ivc']&top['vix']&top['ps']]; OR3=sub[n_top>=1]; TWO=sub[n_top>=2]
    IRON3=sub[bot['ivc']&bot['vix']&bot['ps']]
    for ax,agg,ttl in [(axes[0],'mean','MEAN'),(axes[1],'median','MEDIAN')]:
        ax.set_facecolor(COL_PAN); aggf=(lambda s:s.mean()) if agg=='mean' else (lambda s:s.median())
        def line(s,c,lbl,lw=1.5,ls='-'):
            if len(s)>0: ax.plot(xs,[aggf(s[f'PnL_d{i:03d}']) for i in xs],color=c,linewidth=lw,linestyle=ls,label=f'{lbl} N={len(s)}')
        line(sub,'white','RAW',2.0)
        line(AND3,'#3fb950','AND 3 (ORO TRIPLE)',2.2); line(TWO,'#a3e635','>=2 de 3'); line(OR3,'#bc8cff','OR cualquiera',1.2,'--')
        line(IRON3,'#f85149','AND BOT 3 (HIERRO)',1.6)
        ax.axhline(0,color='gray',linewidth=0.5,alpha=0.5); ax.grid(alpha=0.2)
        ax.set_xlabel('dia (d001..d030)'); ax.set_ylabel(f'{ttl} PnL acum (pts)')
        ax.set_title(f'STT - Triple IVC x IV ATM x PS [{ttl}]'); ax.legend(loc='upper left',fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(EVDIR,fname),facecolor=COL_BG); plt.close(fig)
    print(f'    [chart] {fname}')

print('[5] charts ...', flush=True)
traj_dual('ivc_pct','IV_CONV','traj_ivc.png')
traj_dual('vix_pct','IV ATM','traj_vix.png')
traj_dual('ps_pct','PUT SKEW','traj_ps.png')
heat_dual('ivc_pct','vix_pct','IVC','IV ATM','heat_iv.png')
heat_dual('ivc_pct','ps_pct','IVC','PS','heat_ip.png')
heat_dual('vix_pct','ps_pct','IV ATM','PS','heat_vp.png')
combo_dual('ivc_pct','vix_pct','IVC','IV ATM','comp_iv.png')
combo_dual('ivc_pct','ps_pct','IVC','PS','comp_ip.png')
combo_dual('vix_pct','ps_pct','IV ATM','PS','comp_vp.png')
triple_dual('comp_triple.png')

print('\n=== Summary ===')
print(f'Latest: {data["latest"]}')
print(f'Stats: {data["stats"]}')
print(f'Triple ORO (AND 3): {[c for c in data["tbl_triple"] if "ORO TRIPLE" in c["label"]]}')
