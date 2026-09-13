"""Analytical tracking-time-constant rule for back-calculation anti-windup.
Reproduces all figures of the manuscript. Version 1.0 (2026-09-12)."""
import numpy as np, sys
from scipy.optimize import brentq

# ---------- full discrete-time simulation (Algorithm 1 of Caparroz et al.) ----------
def sim(x, Rs, DdT, alpha, LT=1/6, T=3.0, K=1.0, Ts=0.01, d0=1.0, t0=1.0):
    L=LT*T; Kp=T/(K*(x*T+L)); Ti=T; Ki=Kp/Ti; Tt=max(alpha,Ts/Ti)*Ti
    umin=-d0*(1-Rs); Dd=DdT*T; tend=Dd+25*T
    N=int(tend/Ts); nL=int(round(L/Ts))
    vh=np.zeros(N+nL+1); y=ui=esat=0.0; iae=0.0
    yh=np.empty(N); uh=np.empty(N)
    for k in range(N):
        t=k*Ts; e=-y
        ui=ui+Ki*Ts*e+Ts/Tt*esat
        uc=Kp*e+ui; us=max(uc,umin); esat=us-uc
        d=d0 if t0<=t<t0+Dd else 0.0
        vh[k+nL]=us+d
        y=y+Ts/T*(-y+K*vh[k]); yh[k]=y; uh[k]=us; iae+=abs(e)*Ts
    return iae, yh, uh, np.arange(N)*Ts

def alpha_star_sim(x,Rs,DdT,LT=1/6,Ts=0.01,grid=None):
    if grid is None: grid=np.concatenate([[Ts/3.0],np.linspace(0.02,1.0,50)])
    iae=[sim(x,Rs,DdT,a,LT,Ts=Ts)[0] for a in grid]
    a=grid[int(np.argmin(iae))]
    return 0.0 if a<=Ts/3.0+1e-12 else a

# ---------- reduced normalized recovery-phase model ----------
h=1e-3
def dphase(y0,ui0,vpre,g,ell,tend=60.0):
    nL=int(round(ell/h)); N=int(tend/h)
    vh=np.full(N+nL+1,vpre); y=y0; ui=ui0; ys=np.empty(N)
    for k in range(N):
        u=-g*y+ui; vh[k+nL]=u; ui-=h*g*y; y+=h*(-y+vh[k]); ys[k]=y
    return ys
def median_exact(x,LT):
    g=1/(x+LT); hi=dphase(0.0,1.0,0.0,g,LT); c=np.cumsum(hi)*h
    tm=np.argmax(c>=0.5*c[-1])*h
    B=dphase(0.0,-1.0,-1.0,g,LT); A=dphase(1.0,g,0.0,g,LT); km=int(tm/h)
    return tm, -B[km]/A[km], hi, A, B

# ---------- closed-form pieces ----------
def y0star_explicit(x,LT):
    g=1/(x+LT); tm=1/g+np.log(2)
    return tm if abs(1-g)<1e-9 else (np.exp((1-g)*tm)-1)/(1-g)
def y0star_nodelay(x,LT):
    g=1/(x+LT)
    f=lambda t: np.exp(-t)-np.exp(-g*t)/g-(g-1)/(2*g)
    ts=np.linspace(0.01,12,4000); v=f(ts); i=np.where(np.diff(np.sign(v))!=0)[0][-1]
    tm=brentq(f,ts[i],ts[i+1])
    return tm if abs(1-g)<1e-9 else (np.exp((1-g)*tm)-1)/(1-g)
def Rs_lim(y0s): return y0s/(1+y0s)
def alpha_long(Rs,Rsl):
    return 0.0 if Rs<=Rsl else 1-np.log(1-Rsl)/np.log(1-Rs)
def alpha_short(x,Rs,DdT,LT,y0s):
    g=1/(x+LT); y0=Rs/(1-Rs); ss=(1-Rs)/g; Delta=DdT-ss-LT
    yD=y0+(1-(1+y0)*np.exp(-(ss+LT)))*np.exp(-Delta)
    if yD<=y0s: return 0.0
    te_star=np.log((1+yD)/(1+y0s))
    te=lambda a: a*np.log(1+y0+(1-(1+y0)*np.exp(-LT/a))*np.exp(-Delta/a))
    return brentq(lambda a: te(a)-te_star,1e-4,10.0)

if __name__=="__main__":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":9,"axes.labelsize":9,"legend.fontsize":8,"figure.dpi":150})
    LT=1/6; T=3.0
    # ----- Fig 1: time responses -----
    x,Rs,DdT=0.2,0.5,10
    tm,y0s,_,_,_=median_exact(x,LT); Rsl=Rs_lim(y0s); a_star=alpha_long(Rs,Rsl)
    fig,ax=plt.subplots(3,1,figsize=(5.2,5.6),sharex=True)
    for a,lab,ls in [(1.0,r"$T_t=T_i$","--"),(0.0,"instantaneous ($T_t\\to0$)",":"),(a_star,r"$T_t=T_t^*$ (rule)","-")]:
        iae,yh,uh,t=sim(x,Rs,DdT,a,LT)
        Kp=T/(x*T+LT*T); umin=-(1-Rs)
        ax[0].plot(t,yh,ls,label=f"{lab}, IAE={iae:.2f}"); ax[1].plot(t,uh,ls)
        # z = 1 + ui - g y in normalized units: reconstruct from uc? use uh only in saturation; plot ui via uc = Kp e + ui -> not stored; skip
    ax[0].axvspan(1,1+DdT*T,color="0.9",label="disturbance on"); ax[0].set_ylabel("process output $y$"); ax[0].legend(loc="upper right")
    ax[1].set_ylabel("applied input $u_{sat}$"); ax[1].axhline(-(1-Rs),color="k",lw=0.6,ls="-.")
    # z panel: normalized tracking variable computed from reduced model for the three alphas
    g=1/(x+LT); y0=Rs/(1-Rs)
    sig=np.linspace(-2,3,500)
    for a,ls in [(1.0,"--"),(0.02,":"),(a_star,"-")]:
        z=np.where(sig<0,-a*g*y0*np.ones_like(sig),-a*g*y0*np.exp(-sig/a)+a*g*(1-np.exp(-sig/a)))
        te=a*np.log(1+y0); z=np.where(sig>te,np.nan,z)
        ax[2].plot(1+DdT*T+LT*T+sig*T,z,ls)
        ax[2].plot(1+DdT*T+LT*T+te*T,0,"ko",ms=3)
    ax[2].axhline(0,color="k",lw=0.6); ax[2].set_ylabel(r"tracking variable $z$"); ax[2].set_xlabel("time [s]")
    ax[2].set_xlim(20,45)
    fig.tight_layout(); fig.savefig("fig1_time_response.pdf"); plt.close(fig)
    # ----- Fig 2: h_i median and critical response -----
    fig,ax=plt.subplots(1,2,figsize=(6.4,2.6))
    sig=np.arange(int(60/h))*h
    for x_ ,c in [(0.2,"C0"),(1.0,"C3")]:
        tm,y0s,hi,A,B=median_exact(x_,LT); g=1/(x_+LT)
        ax[0].plot(sig,hi*g,c,label=f"$x={x_}$"); ax[0].axvline(tm,color=c,ls=":")
        yn=y0s*A+B; ax[1].plot(sig,yn/y0s,c,label=f"$x={x_}$, $\\hat y_0^*={y0s:.2f}$"); ax[1].axvline(tm,color=c,ls=":")
    ax[0].set_xlim(0,8); ax[0].set_xlabel(r"$\sigma=t/T$"); ax[0].set_ylabel(r"$g\,h_i(\sigma)$"); ax[0].legend()
    ax[1].set_xlim(0,8); ax[1].set_ylim(-0.06,0.6); ax[1].axhline(0,color="k",lw=0.6); ax[1].set_xlabel(r"$\sigma=t/T$"); ax[1].set_ylabel(r"$\hat y(\sigma)/\hat y_0^*$"); ax[1].legend()
    fig.tight_layout(); fig.savefig("fig2_median.pdf"); plt.close(fig)
    # ----- Fig 3: Rs_lim(x, L/T) -----
    xs=np.linspace(0.2,1.5,14); fig,ax=plt.subplots(figsize=(5.2,3.2))
    for LT_,c in [(0.05,"C0"),(1/6,"C1"),(1/3,"C2"),(0.5,"C3")]:
        ex=[Rs_lim(median_exact(x_,LT_)[1]) for x_ in xs]
        ap=[Rs_lim(y0star_explicit(x_,LT_)) for x_ in xs]
        ax.plot(xs,ex,c,label=f"$L/T={LT_:.3g}$ exact"); ax.plot(xs,ap,c+"--")
    simpts=[(x_,alpha_star_sim) for x_ in [0.2,0.4,0.6,0.8,1.0]]
    Rsl_sim=[]
    for x_ in [0.2,0.4,0.6,0.8,1.0]:
        for Rs_ in np.arange(0.2,0.8,0.01):
            if alpha_star_sim(x_,Rs_,10,LT,Ts=0.02)>0: Rsl_sim.append((x_,Rs_)); break
    ax.plot(*zip(*Rsl_sim),"k^",ms=5,label="full simulation, $L/T=1/6$")
    ax.plot([0.2,1.0],[0.3412,0.69],"ks",mfc="none",ms=6,label="Caparroz et al. (2026), Fig. 10a")
    ax.set_xlabel(r"controller aggressiveness $x=\lambda/T$"); ax.set_ylabel(r"$R_{S,\mathrm{lim}}$"); ax.set_ylim(0,1); ax.legend(fontsize=7,ncol=2)
    fig.tight_layout(); fig.savefig("fig3_rslim.pdf"); plt.close(fig)
    np.save("rsl_sim.npy",np.array(Rsl_sim))
    # ----- Fig 4: alpha* vs Rs (long disturbance) -----
    fig,axs=plt.subplots(1,2,figsize=(6.6,2.9)); ax=axs[0]; Rss=np.linspace(0.05,0.97,200)
    for x_,c in [(0.2,"C0"),(0.5,"C1"),(1.0,"C3")]:
        Rsl=Rs_lim(median_exact(x_,LT)[1]); ax.plot(Rss,[alpha_long(r,Rsl) for r in Rss],c,label=f"$x={x_}$ rule")
        Rsl_e=Rs_lim(y0star_explicit(x_,LT)); ax.plot(Rss,[alpha_long(r,Rsl_e) for r in Rss],c+"--",lw=0.8)
        pts=np.arange(0.1,0.96,0.05); ax.plot(pts,[alpha_star_sim(x_,r,10,LT,Ts=0.02) for r in pts],c+"o",ms=3)
    ax.plot([0.95,0.95],[0.866,0.61],"ks",mfc="none",ms=6,label="Caparroz et al. $\\alpha_{max}$")
    ax.set_xlabel(r"saturation ratio $R_S$"); ax.set_ylabel(r"$\alpha^*=T_t^*/T_i$"); ax.set_ylim(0,1); ax.legend(fontsize=7)
    ax.set_title("(a) long disturbance",fontsize=9)
    ax=axs[1]; Dds=np.linspace(0.3,10,120)
    for (x_,Rs_),c in [((0.2,0.6),"C0"),((0.2,0.8),"C1"),((0.2,0.9),"C2"),((1.0,0.9),"C3")]:
        y0s=median_exact(x_,LT)[1]
        ax.plot(Dds,[alpha_short(x_,Rs_,d,LT,y0s) for d in Dds],c,label=f"$x={x_}$, $R_S={Rs_}$")
        pts=[1/3,2/3,1,2,4,10]; ax.plot(pts,[alpha_star_sim(x_,Rs_,d,LT,Ts=0.02) for d in pts],c+"o",ms=3)
    ax.set_xscale("log"); ax.set_xlabel(r"disturbance duration $D_d/T$"); ax.set_ylabel(r"$\alpha^*$"); ax.set_ylim(0,1); ax.legend(fontsize=7)
    ax.set_title("(b) dependence on duration",fontsize=9)
    fig.tight_layout(); fig.savefig("fig4_alpha.pdf"); plt.close(fig)
    print("done", Rsl_sim)
