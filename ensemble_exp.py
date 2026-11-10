#!/usr/bin/env python3
"""Exp G: single (beta, seed), x0=0.01490, 100pA signal. Seed-parameterized for later ensemble."""
import sys, numpy as np, jax, jax.numpy as jnp, jax.lax as lax, networkx as nx
from jax import jit, vmap

jax.config.update("jax_enable_x64", False)

BETA = float(sys.argv[1])  # scale of shifted exponential
SEED = int(sys.argv[2])    # random seed (42 for scouting)

N=50; K_WS=4; BETA_WS=0.3; A0=1.0; EPSILON=0.03; TAU=0.3
G_X0=0.005; G_CLIP_LO=0.005; G_CLIP_HI=0.05
DT=0.01; T_MAX=50000.0; T_START=5000.0; NS=int(T_MAX/DT); SS=int(T_START/DT); H=5

hh=np.load("hh_noise100_scaled.npz"); V_HH=hh["V_scaled"]

G_ws=nx.watts_strogatz_graph(N, K_WS, BETA_WS, seed=SEED)
W=nx.to_numpy_array(G_ws, dtype=np.float32)
SRC, TGT = np.where(W>0); E=len(SRC)
DEG=np.maximum(W.sum(axis=1), 1.0).astype(np.float32)
md=int(round(TAU/DT))+2
DL=jnp.array(np.full(E, int(round(TAU/DT)), np.int32))
SR=jnp.array(SRC, jnp.int32); TG=jnp.array(TGT, jnp.int32); NU=jnp.int32(N)
xs=jnp.array(V_HH, dtype=jnp.float32); sd=jnp.sqrt(jnp.float32(DT))
MU_S=jnp.float32(np.mean(V_HH[SS:])); S_post_zm=jnp.array(V_HH[SS:]-MU_S, dtype=jnp.float32)

raw = G_X0 + np.random.default_rng(SEED).exponential(scale=BETA, size=E).astype(np.float32)
raw = np.clip(raw, G_CLIP_LO, G_CLIP_HI)
EG = jnp.array((raw/DEG[SRC]).astype(np.float32))

v0=jnp.array(np.random.default_rng(SEED).uniform(-1,1,N), jnp.float32)
w0=jnp.zeros(N, jnp.float32)

# C1
D_vals=np.arange(0.0, 0.065, 0.002)
C1=np.zeros(len(D_vals), np.float32)

@jit
def _c1(sigma, k):
    ns=sigma*sd; buf=jnp.zeros((md,N),jnp.float32)
    c=(v0, w0, buf, jnp.int32(0), k, jnp.int32(0))
    def step(c, x):
        v,w,b,bi,k,si=c; dt_=x
        b=b.at[bi].set(v); ri=(bi-DL)%md; vd=b[ri,TG]; vs=v[SR]
        Ic=jax.ops.segment_sum(EG*(vd-vs), SR, NU)
        k,kc,ki=jax.random.split(k,3); xi=jax.random.normal(ki,(N,))*ns
        vn=v+((v-v**3/3-w+dt_+Ic)/EPSILON)*DT+xi; wn=w+(v+A0)*DT
        spike=(v<0.0)&(vn>=0.0); R_raw=jnp.mean(jnp.where(spike,1.0,0.0))
        return (vn,wn,b,(bi+1)%md,k,si+1), R_raw
    _, R_all=lax.scan(step, c, xs); R_post=R_all[SS:]
    return jnp.mean(S_post_zm*R_post)/(jnp.std(S_post_zm)*jnp.std(R_post)+1e-10)

@jit
def run_c1(D, k):
    ks=jax.random.split(k, D.shape[0])
    return vmap(_c1)(D, ks)

for start in range(0, len(D_vals), 3):
    end=min(start+3, len(D_vals))
    ck=jax.random.fold_in(jax.random.PRNGKey(SEED), start)
    C1[start:end]=np.asarray(run_c1(jnp.array(D_vals[start:end], jnp.float32), ck))

# RMSE_t
D_vals_r=np.concatenate([[0.0005,0.001,0.0015], np.arange(0.0,0.065,0.002)])
D_vals_r=np.sort(np.unique(D_vals_r))
RMSE_t=np.zeros(len(D_vals_r))

@jit
def run_sim(D_val, key):
    ns=D_val*sd
    buf=jnp.zeros((md,N),jnp.float32); vt=jnp.zeros((NS,N),jnp.float32)
    c=(v0, w0, buf, jnp.int32(0), key, jnp.int32(0), vt)
    def step(c, x):
        v,w,b,bi,k,si,vt=c; dt_=x
        b=b.at[bi].set(v); ri=(bi-DL)%md; vd=b[ri,TG]; vs=v[SR]
        Ic=jax.ops.segment_sum(EG*(vd-vs), SR, NU)
        k,kc,ki=jax.random.split(k,3); xi=jax.random.normal(ki,(N,))*ns
        vn=v+((v-v**3/3-w+dt_+Ic)/EPSILON)*DT+xi; wn=w+(v+A0)*DT
        vt=vt.at[si].set(vn)
        return (vn,wn,b,(bi+1)%md,k,si+1,vt), None
    (_,_,_,_,_,_,vtf),_=lax.scan(step, c, xs)
    return vtf

Ut=V_HH[SS+H:].astype(np.float64)
sp=int(0.75*(NS-SS-H))
for i, D in enumerate(D_vals_r):
    vtf=run_sim(jnp.float32(D), jax.random.PRNGKey(SEED))
    Rf=np.array(vtf).astype(np.float64)
    Rt=Rf[SS:-H]
    Rt_t=np.copy(Rt); Rt_t[(Rt>-1.2)&(Rt<-0.8)]=0
    Wr=np.linalg.solve(Rt_t[:sp].T@Rt_t[:sp]+1e-4*np.eye(50), Rt_t[:sp].T@Ut[:sp])
    scl=np.std(Ut[:sp])/np.std(Rt_t[:sp]@Wr)
    pv=((Rt_t[sp:]@Wr)-np.mean(Rt_t[sp:]@Wr))*scl+np.mean(Ut[sp:])
    RMSE_t[i]=np.sqrt(np.mean((Ut[sp:]-pv)**2))

np.savez(f"exp_g_x0005_beta{BETA:.4f}_seed{SEED}.npz",
         beta=BETA, x0=G_X0, seed=SEED, D=D_vals, C1=C1, D_rmse=D_vals_r, RMSE_t=RMSE_t)
print(f"beta={BETA:.4f} x0={G_X0} seed={SEED}: C1max={C1.max():.4f}, RMSE_t_min={np.nanmin(RMSE_t):.6f}", flush=True)
