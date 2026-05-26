// bitches (a dice game) — COMPETITIVE N-player strategy ("lowest score wins").
//
// Everything in exact_dp.cpp minimizes the EXPECTED score (single agent). The
// actual game is won by having the LOWEST score at the table, which is a
// different objective: you don't care how low you go, only about beating the
// field. That rewards variance management (gamble when behind, play safe when
// ahead) and, for symmetric players, becomes a game-theoretic equilibrium.
//
// This program implements the full plan from TODO.md item 1:
//
//  (A) SCORE-DISTRIBUTION DP. Extend the value DP to carry the full distribution
//      of final scores per state (not just the mean). For the expected-optimal
//      policy this gives mean/variance/percentiles/P(perfect) for free.
//
//  (B) WIN-VALUE W(s). Given (N-1) i.i.d. opponents each with score distribution
//      D, a player who finishes at score s earns expected win-share
//        W(s) = ( P(opp>=s)^N - P(opp>s)^N ) / ( N * P(opp=s) )      (=ge^(N-1) if P(opp=s)=0)
//      i.e. outright win if all opponents are higher, split N-ways on ties.
//      This is the terminal payoff (higher is better; W is non-increasing in s).
//
//  (C) WIN-PROBABILITY DP U(S,g). State = (hand-state S, points banked so far g).
//        U(S,g) = E_roll [ max over kept K<S  U(K, g + banked(roll,K)) ]
//        U(empty,g) = W(g)
//      U(full,0) is the player's expected win-share. The maximizing action
//      depends on g — that's the variance management the mean-optimal policy lacks.
//
//  (D) BEST RESPONSE to "everyone plays expected-optimal": field D = expected-
//      optimal score distribution. a(N)=U(full,0) > 1/N measures the exploitation.
//
//  (E) SYMMETRIC EQUILIBRIUM via fictitious play: iterate D <- (score dist of the
//      best response to (N-1) opponents drawn from D) until D is a fixed point.
//
//  (F) VALIDATION: multi-player Monte-Carlo tournament confirms the DP win-shares.
//
// Score support: a d6 face f has penalty 6-f (<=5), so 12 d6 <= 60; d8<=7, d10<=9,
// d12<=11. Max total score = 87. Bound g per state by reachability (see GMAXof).
//
// Build: c++ -O3 -mcpu=native -o competitive competitive.cpp   (-march=native on x86)

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>

static inline int idx(int a6,int a8,int a10,int a12){ return ((a6*2+a8)*2+a10)*2+a12; }

static const int MAXS = 87;     // maximum possible final score
static const int SZ   = 88;     // score-index size (0..87)

static double V[104];           // expected-optimal value function
static double fact[13], pw6[13];
static int    Mof[104];         // max remaining penalty from a state
static int    GMAXof[104];      // max banked-so-far g that can reach this state = 87 - M(S)
static int    NTH = 1;          // worker threads for the (S,g) DP passes

static inline int maxpen(int a6,int a8,int a10,int a12){ return 5*a6+7*a8+9*a10+11*a12; }

static void initTables(){
    fact[0]=1; for(int i=1;i<13;++i) fact[i]=fact[i-1]*i;
    for(int a=0;a<13;++a) pw6[a]=pow(6.0,a);
    for(int a6=0;a6<=12;++a6)for(int a8=0;a8<=1;++a8)for(int a10=0;a10<=1;++a10)for(int a12=0;a12<=1;++a12){
        int m=maxpen(a6,a8,a10,a12); int id=idx(a6,a8,a10,a12);
        Mof[id]=m; GMAXof[id]=MAXS-m;
    }
}

// ---- (shared) expand a roll into the sorted-prefix penalty sums for the d6 dice.
// Given face counts cp[0..5] (penalty 5,4,3,2,1,0), S6[k] = sum of k highest d6 pen.
static inline void prefix6(const int cp[6], double S6[13]){
    S6[0]=0; int k=0;
    for(int pen=5,c=0;pen>=0;--pen,++c) for(int j=0;j<cp[c];++j){ S6[k+1]=S6[k]+pen; ++k; }
}

// =====================================================================
// (A0) Expected-optimal value function V (same DP as exact_dp.cpp).
// =====================================================================
static void solveV(){
    V[idx(0,0,0,0)]=0;
    for(int total=1; total<=15; ++total)
     for(int a6=0;a6<=12;++a6)for(int a8=0;a8<=1;++a8)for(int a10=0;a10<=1;++a10)for(int a12=0;a12<=1;++a12){
      if(a6+a8+a10+a12!=total) continue; double Vacc=0;
      for(int n1=0;n1<=a6;++n1)for(int n2=0;n2<=a6-n1;++n2)for(int n3=0;n3<=a6-n1-n2;++n3)
      for(int n4=0;n4<=a6-n1-n2-n3;++n4)for(int n5=0;n5<=a6-n1-n2-n3-n4;++n5){
        int n6=a6-n1-n2-n3-n4-n5;
        double w6=(fact[a6]/(fact[n1]*fact[n2]*fact[n3]*fact[n4]*fact[n5]*fact[n6]))/pw6[a6];
        int cp[6]={n1,n2,n3,n4,n5,n6}; double S6[13]; prefix6(cp,S6);
        double totp6=S6[a6]; double w8=a8?1.0/8:1,w10=a10?1.0/10:1,w12=a12?1.0/12:1;
        for(int p8=0;p8<=(a8?7:0);++p8)for(int p10=0;p10<=(a10?9:0);++p10)for(int p12=0;p12<=(a12?11:0);++p12){
          double S8[2]={0,(double)p8},S10[2]={0,(double)p10},S12[2]={0,(double)p12};
          double totpen=totp6+(a8?p8:0)+(a10?p10:0)+(a12?p12:0); double best=-1e300;
          for(int k6=0;k6<=a6;++k6)for(int k8=0;k8<=a8;++k8)for(int k10=0;k10<=a10;++k10)for(int k12=0;k12<=a12;++k12){
            if(k6==a6&&k8==a8&&k10==a10&&k12==a12) continue;
            double val=S6[k6]+S8[k8]+S10[k10]+S12[k12]-V[idx(k6,k8,k10,k12)]; if(val>best)best=val;
          }
          Vacc += w6*w8*w10*w12*(totpen-best);
        }
      }
      V[idx(a6,a8,a10,a12)]=Vacc;
     }
}

// expected-optimal kept composition for a roll (canonical tie-break: first found
// = fewer kept = bank more, matching the MC simulators).
static inline void optKV(int a6,int a8,int a10,int a12,const double*S6,int p8,int p10,int p12,
                         int&bk6,int&bk8,int&bk10,int&bk12){
    double S8[2]={0,(double)p8},S10[2]={0,(double)p10},S12[2]={0,(double)p12};
    double best=-1e300; bk6=bk8=bk10=bk12=0;
    for(int k6=0;k6<=a6;++k6)for(int k8=0;k8<=a8;++k8)for(int k10=0;k10<=a10;++k10)for(int k12=0;k12<=a12;++k12){
        if(k6==a6&&k8==a8&&k10==a10&&k12==a12) continue;
        double val=S6[k6]+S8[k8]+S10[k10]+S12[k12]-V[idx(k6,k8,k10,k12)];
        if(val>best){ best=val; bk6=k6;bk8=k8;bk10=k10;bk12=k12; }
    }
}

// =====================================================================
// (A) Score distribution of the EXPECTED-OPTIMAL policy.
//     Dopt[S][v] = P(additional penalty from S = v) under optimal play.
// =====================================================================
static std::vector<double> Dopt[104];

static void scoreDistOptimal(){
    for(int s=0;s<104;++s){ Dopt[s].assign(SZ,0.0); }
    Dopt[idx(0,0,0,0)][0]=1.0;
    for(int total=1; total<=15; ++total)
     for(int a6=0;a6<=12;++a6)for(int a8=0;a8<=1;++a8)for(int a10=0;a10<=1;++a10)for(int a12=0;a12<=1;++a12){
      if(a6+a8+a10+a12!=total) continue;
      int sid=idx(a6,a8,a10,a12); std::vector<double>&D=Dopt[sid];
      for(int n1=0;n1<=a6;++n1)for(int n2=0;n2<=a6-n1;++n2)for(int n3=0;n3<=a6-n1-n2;++n3)
      for(int n4=0;n4<=a6-n1-n2-n3;++n4)for(int n5=0;n5<=a6-n1-n2-n3-n4;++n5){
        int n6=a6-n1-n2-n3-n4-n5;
        double w6=(fact[a6]/(fact[n1]*fact[n2]*fact[n3]*fact[n4]*fact[n5]*fact[n6]))/pw6[a6];
        int cp[6]={n1,n2,n3,n4,n5,n6}; double S6[13]; prefix6(cp,S6);
        double w8=a8?1.0/8:1,w10=a10?1.0/10:1,w12=a12?1.0/12:1;
        for(int p8=0;p8<=(a8?7:0);++p8)for(int p10=0;p10<=(a10?9:0);++p10)for(int p12=0;p12<=(a12?11:0);++p12){
          int bk6,bk8,bk10,bk12; optKV(a6,a8,a10,a12,S6,p8,p10,p12,bk6,bk8,bk10,bk12);
          int totpen=(int)S6[a6]+(a8?p8:0)+(a10?p10:0)+(a12?p12:0);
          int keptpen=(int)S6[bk6]+(bk8?p8:0)+(bk10?p10:0)+(bk12?p12:0);
          int b=totpen-keptpen;                  // banked this roll
          double w=w6*w8*w10*w12;
          const std::vector<double>&DK=Dopt[idx(bk6,bk8,bk10,bk12)];
          int mk=Mof[idx(bk6,bk8,bk10,bk12)];
          for(int v=0;v<=mk;++v) if(DK[v]>0) D[b+v]+=w*DK[v];
        }
      }
     }
}

// =====================================================================
// (B) Win-value W(s): expected win-share finishing at score s, against
//     (N-1) i.i.d. opponents with score distribution D, fair tie-splitting.
// =====================================================================
static void computeW(const double*D,int N,double*W /*size SZ*/){
    // suffix sums: ge(s)=P(opp>s), eq(s)=P(opp=s)
    double ge=0.0;
    for(int s=MAXS;s>=0;--s){
        double eq=D[s];
        double geq=ge+eq;                         // P(opp>=s)
        if(eq<=0){
            W[s]=pow(ge,N-1);                     // no ties possible at s
        }else{
            W[s]=(pow(geq,N)-pow(ge,N))/(N*eq);
        }
        ge=geq;
    }
}

// =====================================================================
// (C) Win-probability DP U(S,g).  Uhi[sid] points at SZ doubles.
//     Only g in [0, GMAXof[sid]] are meaningful (others unreachable).
// =====================================================================
static double Ugrid[104][SZ];

// one worker: processes the d6 face-counts whose first count is n1 (grabbed via
// the shared atomic), accumulating  w * max_K U(K,g+banked)  into private acc[].
static void uSlab(int a6,int a8,int a10,int a12,int gmax,std::atomic<int>*n1ctr,double*acc){
    int n1;
    while((n1=n1ctr->fetch_add(1))<=a6){
      for(int n2=0;n2<=a6-n1;++n2)for(int n3=0;n3<=a6-n1-n2;++n3)
      for(int n4=0;n4<=a6-n1-n2-n3;++n4)for(int n5=0;n5<=a6-n1-n2-n3-n4;++n5){
        int n6=a6-n1-n2-n3-n4-n5;
        double w6=(fact[a6]/(fact[n1]*fact[n2]*fact[n3]*fact[n4]*fact[n5]*fact[n6]))/pw6[a6];
        int cp[6]={n1,n2,n3,n4,n5,n6}; double S6[13]; prefix6(cp,S6);
        double w8=a8?1.0/8:1,w10=a10?1.0/10:1,w12=a12?1.0/12:1;
        for(int p8=0;p8<=(a8?7:0);++p8)for(int p10=0;p10<=(a10?9:0);++p10)for(int p12=0;p12<=(a12?11:0);++p12){
          int totpen=(int)S6[a6]+(a8?p8:0)+(a10?p10:0)+(a12?p12:0);
          double w=w6*w8*w10*w12;
          int cKidx[104], cBank[104], nc=0;
          for(int k6=0;k6<=a6;++k6)for(int k8=0;k8<=a8;++k8)for(int k10=0;k10<=a10;++k10)for(int k12=0;k12<=a12;++k12){
            if(k6==a6&&k8==a8&&k10==a10&&k12==a12) continue;
            int kept=(int)S6[k6]+(k8?p8:0)+(k10?p10:0)+(k12?p12:0);
            cKidx[nc]=idx(k6,k8,k10,k12); cBank[nc]=totpen-kept; ++nc;
          }
          for(int g=0; g<=gmax; ++g){
            double best=-1e300;
            for(int c=0;c<nc;++c){ double v=Ugrid[cKidx[c]][g+cBank[c]]; if(v>best)best=v; }
            acc[g]+=w*best;
          }
        }
      }
    }
}

static void solveU(const double*W){
    int e=idx(0,0,0,0);
    for(int g=0; g<=GMAXof[e]; ++g) Ugrid[e][g]=W[g];   // U(empty,g)=W(g)
    for(int total=1; total<=15; ++total)
     for(int a6=0;a6<=12;++a6)for(int a8=0;a8<=1;++a8)for(int a10=0;a10<=1;++a10)for(int a12=0;a12<=1;++a12){
      if(a6+a8+a10+a12!=total) continue;
      int sid=idx(a6,a8,a10,a12); int gmax=GMAXof[sid];
      int nth = (a6>=4)? NTH : 1;                       // small states: not worth threading
      std::atomic<int> n1ctr(0);
      std::vector<std::vector<double>> accs(nth, std::vector<double>(gmax+1,0.0));
      if(nth==1){ uSlab(a6,a8,a10,a12,gmax,&n1ctr,accs[0].data()); }
      else{
        std::vector<std::thread> th;
        for(int t=0;t<nth;++t) th.emplace_back(uSlab,a6,a8,a10,a12,gmax,&n1ctr,accs[t].data());
        for(auto&x:th) x.join();
      }
      for(int g=0; g<=gmax; ++g){ double s=0; for(int t=0;t<nth;++t) s+=accs[t][g]; Ugrid[sid][g]=s; }
     }
}

// best-response action at (S,g) for a given roll (uses current Ugrid).
static inline void brAction(int a6,int a8,int a10,int a12,const double*S6,int p8,int p10,int p12,int g,
                            int&bk6,int&bk8,int&bk10,int&bk12,int&banked){
    int totpen=(int)S6[a6]+(a8?p8:0)+(a10?p10:0)+(a12?p12:0);
    double best=-1e300; bk6=bk8=bk10=bk12=0; banked=0;
    for(int k6=0;k6<=a6;++k6)for(int k8=0;k8<=a8;++k8)for(int k10=0;k10<=a10;++k10)for(int k12=0;k12<=a12;++k12){
        if(k6==a6&&k8==a8&&k10==a10&&k12==a12) continue;
        int kept=(int)S6[k6]+(k8?p8:0)+(k10?p10:0)+(k12?p12:0);
        int b=totpen-kept; double v=Ugrid[idx(k6,k8,k10,k12)][g+b];
        if(v>best){ best=v; bk6=k6;bk8=k8;bk10=k10;bk12=k12; banked=b; }
    }
}

// =====================================================================
// (E-helper) Score distribution of the best-response policy currently in Ugrid.
//   Forward push of probability mass over (S,g) states, top-down by dice count.
//   Returns the final-score distribution (size SZ) into out[].
// =====================================================================
// worker: pushes mass from state S (rows of msrc) into a private delta grid.
static void brSlab(int a6,int a8,int a10,int a12,int gmax,const double*msrc,
                   std::atomic<int>*n1ctr,double*delta /*104*SZ*/){
    int n1;
    while((n1=n1ctr->fetch_add(1))<=a6){
      for(int n2=0;n2<=a6-n1;++n2)for(int n3=0;n3<=a6-n1-n2;++n3)
      for(int n4=0;n4<=a6-n1-n2-n3;++n4)for(int n5=0;n5<=a6-n1-n2-n3-n4;++n5){
        int n6=a6-n1-n2-n3-n4-n5;
        double w6=(fact[a6]/(fact[n1]*fact[n2]*fact[n3]*fact[n4]*fact[n5]*fact[n6]))/pw6[a6];
        int cp[6]={n1,n2,n3,n4,n5,n6}; double S6[13]; prefix6(cp,S6);
        double w8=a8?1.0/8:1,w10=a10?1.0/10:1,w12=a12?1.0/12:1; double w=w6*w8*w10*w12;
        for(int p8=0;p8<=(a8?7:0);++p8)for(int p10=0;p10<=(a10?9:0);++p10)for(int p12=0;p12<=(a12?11:0);++p12){
          for(int g=0; g<=gmax; ++g){
            double mass=msrc[g]; if(mass<=0) continue;
            int bk6,bk8,bk10,bk12,b; brAction(a6,a8,a10,a12,S6,p8,p10,p12,g,bk6,bk8,bk10,bk12,b);
            delta[idx(bk6,bk8,bk10,bk12)*SZ + (g+b)] += mass*w;
          }
        }
      }
    }
}

static void brScoreDist(double*out /*size SZ*/){
    static double m[104][SZ];
    for(int s=0;s<104;++s) for(int g=0;g<SZ;++g) m[s][g]=0.0;
    m[idx(12,1,1,1)][0]=1.0;

    for(int total=15; total>=1; --total)
     for(int a6=0;a6<=12;++a6)for(int a8=0;a8<=1;++a8)for(int a10=0;a10<=1;++a10)for(int a12=0;a12<=1;++a12){
      if(a6+a8+a10+a12!=total) continue;
      int sid=idx(a6,a8,a10,a12); int gmax=GMAXof[sid];
      bool any=false; for(int g=0;g<=gmax;++g) if(m[sid][g]>0){any=true;break;}
      if(!any) continue;
      int nth = (a6>=4)? NTH : 1;
      std::atomic<int> n1ctr(0);
      std::vector<std::vector<double>> deltas(nth, std::vector<double>(104*SZ,0.0));
      if(nth==1){ brSlab(a6,a8,a10,a12,gmax,m[sid],&n1ctr,deltas[0].data()); }
      else{
        std::vector<std::thread> th;
        for(int t=0;t<nth;++t) th.emplace_back(brSlab,a6,a8,a10,a12,gmax,m[sid],&n1ctr,deltas[t].data());
        for(auto&x:th) x.join();
      }
      for(int t=0;t<nth;++t){ const double*d=deltas[t].data();
        for(int s=0;s<104;++s) for(int g=0;g<SZ;++g) m[s][g]+=d[s*SZ+g]; }
     }
    int e=idx(0,0,0,0);
    for(int g=0;g<SZ;++g) out[g]=m[e][g];
}

// dot product of a finishing distribution with the win-value -> expected win-share.
static double winShare(const double*D,const double*W){
    double s=0; for(int g=0;g<=MAXS;++g) s+=D[g]*W[g]; return s;
}

// =====================================================================
// CSV artifacts: persist the strategy tables to disk.
//   A policy here is a VALUE table; the move is derived greedily from it.
// =====================================================================
static void dumpV(const char*path){
    FILE*f=fopen(path,"w"); if(!f){ perror(path); return; }
    fprintf(f,"# bitches: expected-optimal (mean-minimizing) value function V.\n");
    fprintf(f,"# state = dice still in hand (a6 d6 in 0..12; a8/a10/a12 each 0/1). V = optimal expected remaining score.\n");
    fprintf(f,"# OPTIMAL MOVE for a roll: keep the subset K of your dice maximizing (sum of kept penalties) - V[K].\n");
    fprintf(f,"a6,a8,a10,a12,V\n");
    for(int a6=0;a6<=12;++a6)for(int a8=0;a8<=1;++a8)for(int a10=0;a10<=1;++a10)for(int a12=0;a12<=1;++a12)
        fprintf(f,"%d,%d,%d,%d,%.10g\n",a6,a8,a10,a12,V[idx(a6,a8,a10,a12)]);
    fclose(f); printf("  wrote %s (104 states)\n",path);
}

// dump the win-probability table currently in Ugrid (defines a g-dependent policy).
static void dumpUgrid(const char*path,const char*desc){
    FILE*f=fopen(path,"w"); if(!f){ perror(path); return; }
    fprintf(f,"# bitches competitive policy: %s\n",desc);
    fprintf(f,"# state = (a6,a8,a10,a12) dice in hand; g = penalty points already banked this game.\n");
    fprintf(f,"# U = expected win-share (probability of having the lowest score) from (state,g) under this policy.\n");
    fprintf(f,"# OPTIMAL MOVE at (state,g) for a roll: keep the subset K maximizing U[K][g + banked_penalty(K)].\n");
    fprintf(f,"# only reachable g (0..87-maxRemaining) are listed.\n");
    fprintf(f,"a6,a8,a10,a12,g,U\n");
    long rows=0;
    for(int a6=0;a6<=12;++a6)for(int a8=0;a8<=1;++a8)for(int a10=0;a10<=1;++a10)for(int a12=0;a12<=1;++a12){
        int sid=idx(a6,a8,a10,a12);
        for(int g=0;g<=GMAXof[sid];++g){ fprintf(f,"%d,%d,%d,%d,%d,%.10g\n",a6,a8,a10,a12,g,Ugrid[sid][g]); ++rows; }
    }
    fclose(f); printf("        wrote %s (%ld rows)\n",path,rows);
}

// =====================================================================
// Monte-Carlo tournament validation.
// =====================================================================
static inline uint64_t xr(uint64_t&s0,uint64_t&s1){ uint64_t x=s0,y=s1; s0=y; x^=x<<23; s1=x^y^(x>>17)^(y>>26); return s1+y; }
static inline int rollD(uint64_t&s0,uint64_t&s1,int sides){ return (int)(((xr(s0,s1)>>32)*(uint64_t)sides)>>32); }

// Play one full game with policy: comp==false -> expected-optimal (uses V);
// comp==true -> competitive best response in Ugrid (g-dependent). Returns score.
static int playGame(uint64_t&s0,uint64_t&s1,bool comp){
    int a6=12,a8=1,a10=1,a12=1,sc=0;
    while(a6+a8+a10+a12>0){
        int h[6]={0,0,0,0,0,0}; for(int j=0;j<a6;++j) h[rollD(s0,s1,6)]++;
        double S6[13]; { int cp[6]={h[5],h[4],h[3],h[2],h[1],h[0]}; prefix6(cp,S6); }
        int p8=a8?rollD(s0,s1,8):0, p10=a10?rollD(s0,s1,10):0, p12=a12?rollD(s0,s1,12):0;
        int bk6,bk8,bk10,bk12;
        if(comp){ int b; brAction(a6,a8,a10,a12,S6,p8,p10,p12,sc,bk6,bk8,bk10,bk12,b); }
        else      optKV(a6,a8,a10,a12,S6,p8,p10,p12,bk6,bk8,bk10,bk12);
        int totpen=(int)S6[a6]+(a8?p8:0)+(a10?p10:0)+(a12?p12:0);
        int keptpen=(int)S6[bk6]+(bk8?p8:0)+(bk10?p10:0)+(bk12?p12:0);
        sc+=totpen-keptpen; a6=bk6;a8=bk8;a10=bk10;a12=bk12;
    }
    return sc;
}

// N seats: seat 0 = hero (heroComp), seats 1..N-1 = opponents (oppComp).
// NOTE: hero and opponents may share Ugrid only if they use the same field; if
// they differ, caller must serialize (this program runs one config at a time).
static double tournament(int N,bool heroComp,bool oppComp,long games,uint64_t seed){
    uint64_t s0=seed*0x9E3779B97F4A7C15ULL+1, s1=seed*0xD1B54A32D192ED03ULL+0x9E3779B9ULL;
    for(int w=0;w<8;++w) xr(s0,s1);
    double heroShare=0;
    std::vector<int> sc(N);
    for(long t=0;t<games;++t){
        sc[0]=playGame(s0,s1,heroComp);
        for(int i=1;i<N;++i) sc[i]=playGame(s0,s1,oppComp);
        int best=sc[0]; for(int i=1;i<N;++i) if(sc[i]<best)best=sc[i];
        int cnt=0; for(int i=0;i<N;++i) if(sc[i]==best)cnt++;
        if(sc[0]==best) heroShare += 1.0/cnt;
    }
    return heroShare/games;
}

// =====================================================================
int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    NTH = (argc>2)?atoi(argv[2]):(int)std::thread::hardware_concurrency(); if(NTH<1) NTH=1;
    int eqMaxN = (argc>3)?atoi(argv[3]):4;   // largest N to solve the equilibrium for (each ~min on 16 cores)
    printf("(threads: %d ; equilibrium up to N=%d)\n", NTH, eqMaxN);
    auto t0=std::chrono::steady_clock::now();
    initTables();
    solveV();
    scoreDistOptimal();

    const double* Dfull = Dopt[idx(12,1,1,1)].data();   // expected-optimal score distribution

    printf("===== strategy artifacts (CSV) =====\n");
    dumpV("optimal_policy.csv");                          // single-agent expected-optimal value table
    const int DUMP_MAXN = eqMaxN;                         // record competitive tables for N=2..maxN (the CLI arg)


    // ---- (A) report the expected-optimal score distribution ----
    double mean=0,m2=0; for(int s=0;s<=MAXS;++s){ mean+=s*Dfull[s]; }
    for(int s=0;s<=MAXS;++s){ m2+=(s-mean)*(s-mean)*Dfull[s]; }
    double sd=sqrt(m2);
    printf("===== (A) Expected-optimal policy: full score distribution =====\n");
    printf("  V(full)        = %.5f   (DP optimal expected score)\n", V[idx(12,1,1,1)]);
    printf("  mean of dist   = %.5f   (must match V)\n", mean);
    printf("  std deviation  = %.5f   variance = %.4f\n", sd, m2);
    printf("  P(score=0)     = %.4e  (1 in %.0f)\n", Dfull[0], 1.0/Dfull[0]);
    { // percentiles
      int qs[]={1,5,10,25,50,75,90,95,99}; double cum=0; int qi=0; int pct[9];
      for(int s=0;s<=MAXS&&qi<9;++s){ cum+=Dfull[s]; while(qi<9 && cum*100.0>=qs[qi]){ pct[qi]=s; ++qi; } }
      printf("  percentiles    :");
      for(int i=0;i<9;++i) printf(" p%d=%d", qs[i], pct[i]);
      printf("\n");
    }
    // E[min of N i.i.d. optimal scores] -> Fetterman Fig.13-style baseline
    printf("  E[min score] for N players (all expected-optimal):\n     ");
    for(int N=2;N<=8;++N){
        double ge=0,Emin=0; // ge=P(>s); P(min=s)=P(>=s)^N - P(>s)^N
        // iterate s ascending: need P(>=s),P(>s)
        double tail=1.0; // P(score>=s) starting s=0 is 1
        double prevtail=1.0;
        for(int s=0;s<=MAXS;++s){
            double pge=tail-Dfull[s];        // P(>s)
            double pmin=pow(tail,N)-pow(pge,N);
            Emin+=s*pmin; tail=pge; (void)prevtail; (void)ge;
        }
        printf(" N=%d:%.3f", N, Emin);
    }
    printf("\n\n");

    // ---- (D) best response to the field of expected-optimal players ----
    printf("===== (D) Best response vs (N-1) expected-optimal opponents =====\n");
    printf("   (win-share 1/N is the symmetric baseline; >1/N means exploitation)\n");
    static double W[SZ];
    for(int N=2;N<=8;++N){
        computeW(Dfull,N,W);
        solveU(W);
        double br=Ugrid[idx(12,1,1,1)][0];          // best-response win-share
        double opt=winShare(Dfull,W);               // expected-optimal's own win-share vs same field
        printf("  N=%d  baseline 1/N=%.4f   expected-optimal=%.5f   best-response=%.5f  (+%.2f%% rel)\n",
               N, 1.0/N, opt, br, 100.0*(br-1.0/N)/(1.0/N));
        if(N<=DUMP_MAXN){ char p[64]; snprintf(p,sizeof p,"competitive_bestresponse_N%d.csv",N);
            char d[96]; snprintf(d,sizeof d,"best response vs %d expected-optimal opponents",N-1);
            dumpUgrid(p,d); }
    }
    printf("\n");

    // ---- (E) symmetric equilibrium via fictitious play ----
    printf("===== (E) Symmetric equilibrium (fictitious play with damping) =====\n");
    int eqNsAll[]={2,3,4,6,8};
    static double Deq[SZ], Dnew[SZ];
    for(int ni=0; ni<5; ++ni){
        int N=eqNsAll[ni];
        if(N>eqMaxN) break;
        for(int g=0;g<SZ;++g) Deq[g]=Dfull[g];      // init field = expected-optimal
        printf("  N=%d  fictitious play:\n", N);
        double alpha=0.5; double delta=0; int it=0; const int MAXIT=120;
        for(it=0; it<MAXIT; ++it){
            auto ti=std::chrono::steady_clock::now();
            computeW(Deq,N,W);
            solveU(W);
            brScoreDist(Dnew);                       // score dist of best response to Deq
            delta=0; for(int g=0;g<SZ;++g) delta+=fabs(Dnew[g]-Deq[g]);
            for(int g=0;g<SZ;++g) Deq[g]=(1-alpha)*Deq[g]+alpha*Dnew[g];
            double share=Ugrid[idx(12,1,1,1)][0];
            double its=std::chrono::duration<double>(std::chrono::steady_clock::now()-ti).count();
            printf("      iter %2d: L1 delta=%.2e  BR win-share=%.5f  (%.2fs)\n", it, delta, share, its);
            if(delta<1e-7) break;
        }
        // consistency: at the fixed point each symmetric player's share must be 1/N.
        computeW(Deq,N,W); solveU(W);
        double selfshare=Ugrid[idx(12,1,1,1)][0];    // BR vs (N-1) equilibrium opponents
        double naiveVsEq=winShare(Dfull,W);          // expected-optimal vs (N-1) equilibrium opponents
        double eqmean=0; for(int s=0;s<=MAXS;++s) eqmean+=s*Deq[s];
        double eqsd=0;   for(int s=0;s<=MAXS;++s) eqsd+=(s-eqmean)*(s-eqmean)*Deq[s]; eqsd=sqrt(eqsd);
        printf("  N=%d  conv in %d iters (L1 delta %.1e)\n", N, it, delta);
        printf("        equilibrium score dist: mean=%.4f sd=%.4f  (vs optimal mean=%.4f sd=%.4f)\n",
               eqmean, eqsd, mean, sd);
        printf("        equilibrium self win-share=%.5f (should be 1/N=%.5f)\n", selfshare, 1.0/N);
        printf("        expected-optimal player vs (N-1) equilibrium opponents = %.5f  (%.2f%% rel vs 1/N)\n",
               naiveVsEq, 100.0*(naiveVsEq-1.0/N)/(1.0/N));
        if(N<=DUMP_MAXN){ char p[64]; snprintf(p,sizeof p,"competitive_equilibrium_N%d.csv",N);
            char d[96]; snprintf(d,sizeof d,"symmetric Nash equilibrium policy for %d players",N);
            dumpUgrid(p,d); }
    }
    printf("\n");

    auto t1=std::chrono::steady_clock::now();
    printf("(DP+equilibrium compute time %.2f s)\n\n", std::chrono::duration<double>(t1-t0).count());

    // ---- (F) Monte-Carlo tournament validation ----
    long G = (argc>1)?atol(argv[1]):4000000;
    printf("===== (F) Monte-Carlo tournament validation (%ld games/config) =====\n", G);
    for(int N : {2,3,4}){
        // hero = best response to expected-optimal field; opponents = expected-optimal.
        computeW(Dfull,N,W); solveU(W);
        double dpBR=Ugrid[idx(12,1,1,1)][0];
        double mcBR=tournament(N,/*heroComp=*/true,/*oppComp=*/false,G,0xC0FFEEull + N);
        // sanity: all expected-optimal -> hero share ~ 1/N
        double mcAll=tournament(N,false,false,G,0xBEEFull + N);
        printf("  N=%d  best-response vs optimal:  DP=%.5f  MC=%.5f  (diff %+.5f)\n", N, dpBR, mcBR, mcBR-dpBR);
        printf("        all expected-optimal:       1/N=%.5f  MC=%.5f\n", 1.0/N, mcAll);
    }
    printf("\n(total time %.2f s)\n", std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count());
    return 0;
}
