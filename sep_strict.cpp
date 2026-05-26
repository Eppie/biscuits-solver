// STRICT separability test for the optimal policy (tie-immune).
// For each state, type t and penalty p, look across all rolls at:
//   vKeep = best achievable value while KEEPING a pen-p die of type t
//   vBank = best achievable value while BANKING a pen-p die
// (value = sum kept penalties - V(kept)). If in some roll vBank > vKeep
// (strictly want to bank) and in another vKeep > vBank (strictly want to keep)
// for the SAME (t,p), then no fixed per-die cutoff is optimal -> non-separable.
// Ties (vBank==vKeep, i.e. either choice is optimal) never trigger a flag.
//
// Build: c++ -O3 -mcpu=native -o sep_strict sep_strict.cpp

#include <cstdio>
#include <cmath>

static inline int idx(int a6,int a8,int a10,int a12){ return ((a6*2+a8)*2+a10)*2+a12; }
static double V[104]; static double fact[13],pw6[13];
static const int SZ[4]={6,8,10,12};

int main(){
    fact[0]=1; for(int i=1;i<13;++i) fact[i]=fact[i-1]*i;
    for(int a=0;a<13;++a) pw6[a]=pow(6.0,a);
    // ---- optimal V ----
    V[idx(0,0,0,0)]=0;
    for(int total=1; total<=15; ++total)
     for(int a6=0;a6<=12;++a6)for(int a8=0;a8<=1;++a8)for(int a10=0;a10<=1;++a10)for(int a12=0;a12<=1;++a12){
      if(a6+a8+a10+a12!=total) continue; double Vacc=0;
      for(int n1=0;n1<=a6;++n1)for(int n2=0;n2<=a6-n1;++n2)for(int n3=0;n3<=a6-n1-n2;++n3)
      for(int n4=0;n4<=a6-n1-n2-n3;++n4)for(int n5=0;n5<=a6-n1-n2-n3-n4;++n5){
        int n6=a6-n1-n2-n3-n4-n5;
        double w6=(fact[a6]/(fact[n1]*fact[n2]*fact[n3]*fact[n4]*fact[n5]*fact[n6]))/pw6[a6];
        double S6[13]; S6[0]=0; { int k=0,cp[6]={n1,n2,n3,n4,n5,n6};
          for(int pen=5,c=0;pen>=0;--pen,++c) for(int j=0;j<cp[c];++j){ S6[k+1]=S6[k]+pen; ++k; } }
        double tp6=S6[a6]; double w8=a8?1.0/8:1,w10=a10?1.0/10:1,w12=a12?1.0/12:1;
        for(int p8=0;p8<=(a8?7:0);++p8)for(int p10=0;p10<=(a10?9:0);++p10)for(int p12=0;p12<=(a12?11:0);++p12){
          double S8[2]={0,(double)p8},S10[2]={0,(double)p10},S12[2]={0,(double)p12};
          double tp=tp6+(a8?p8:0)+(a10?p10:0)+(a12?p12:0); double best=-1e300;
          for(int k6=0;k6<=a6;++k6)for(int k8=0;k8<=a8;++k8)for(int k10=0;k10<=a10;++k10)for(int k12=0;k12<=a12;++k12){
            if(k6==a6&&k8==a8&&k10==a10&&k12==a12) continue;
            double v=S6[k6]+S8[k8]+S10[k10]+S12[k12]-V[idx(k6,k8,k10,k12)]; if(v>best)best=v;
          }
          Vacc += w6*w8*w10*w12*(tp-best);
        }
      }
      V[idx(a6,a8,a10,a12)]=Vacc;
     }
    printf("OPTIMAL V(12,1,1,1) = %.6f\n\n", V[idx(12,1,1,1)]);

    // ---- strict separability + cutoff per state ----
    int nNon=0;
    int cutFace[104][4]; bool sep[104]; for(int s=0;s<104;++s){sep[s]=true; for(int t=0;t<4;++t)cutFace[s][t]=99;}

    for(int a6=0;a6<=12;++a6)for(int a8=0;a8<=1;++a8)for(int a10=0;a10<=1;++a10)for(int a12=0;a12<=1;++a12){
      if(a6+a8+a10+a12==0) continue; int S=idx(a6,a8,a10,a12);
      bool strictBank[4][12]={}, strictKeep[4][12]={};
      const double EPS=1e-7;

      for(int n1=0;n1<=a6;++n1)for(int n2=0;n2<=a6-n1;++n2)for(int n3=0;n3<=a6-n1-n2;++n3)
      for(int n4=0;n4<=a6-n1-n2-n3;++n4)for(int n5=0;n5<=a6-n1-n2-n3-n4;++n5){
        int n6=a6-n1-n2-n3-n4-n5; int cp[6]={n1,n2,n3,n4,n5,n6};
        double S6[13]; S6[0]=0; { int k=0; for(int pen=5,c=0;pen>=0;--pen,++c) for(int j=0;j<cp[c];++j){ S6[k+1]=S6[k]+pen; ++k; } }
        for(int p8=0;p8<=(a8?7:0);++p8)for(int p10=0;p10<=(a10?9:0);++p10)for(int p12=0;p12<=(a12?11:0);++p12){
          double S8[2]={0,(double)p8},S10[2]={0,(double)p10},S12[2]={0,(double)p12};
          // bestByK[t][k] = max value over kept-compositions with k_t == k
          double bestByK[4][13]; for(int t=0;t<4;++t)for(int k=0;k<13;++k) bestByK[t][k]=-1e300;
          for(int k6=0;k6<=a6;++k6)for(int k8=0;k8<=a8;++k8)for(int k10=0;k10<=a10;++k10)for(int k12=0;k12<=a12;++k12){
            if(k6==a6&&k8==a8&&k10==a10&&k12==a12) continue;
            double v=S6[k6]+S8[k8]+S10[k10]+S12[k12]-V[idx(k6,k8,k10,k12)];
            if(v>bestByK[0][k6])bestByK[0][k6]=v;
            if(v>bestByK[1][k8])bestByK[1][k8]=v;
            if(v>bestByK[2][k10])bestByK[2][k10]=v;
            if(v>bestByK[3][k12])bestByK[3][k12]=v;
          }
          // d6: for each penalty p present, hi=#pen>p, eq=#pen==p
          if(a6) for(int p=0;p<=5;++p){ int eq=cp[5-p]; if(!eq) continue;
            int hi=0; for(int pp=p+1;pp<=5;++pp) hi+=cp[5-pp];
            double vKeep=-1e300; for(int k=hi+1;k<=a6;++k) if(bestByK[0][k]>vKeep)vKeep=bestByK[0][k];     // keeps a pen-p
            double vBank=-1e300; for(int k=0;k<=hi+eq-1;++k) if(bestByK[0][k]>vBank)vBank=bestByK[0][k];    // banks a pen-p
            if(vBank>vKeep+EPS) strictBank[0][p]=true;
            if(vKeep>vBank+EPS) strictKeep[0][p]=true;
          }
          int pbig[4]={0,p8,p10,p12},pres[4]={a6,a8,a10,a12};
          for(int t=1;t<4;++t){ if(!pres[t]) continue; int p=pbig[t]; // hi=0,eq=1
            double vKeep=bestByK[t][1];   // keep the die
            double vBank=bestByK[t][0];   // bank the die
            if(vBank>vKeep+EPS) strictBank[t][p]=true;
            if(vKeep>vBank+EPS) strictKeep[t][p]=true;
          }
        }
      }
      int pres[4]={a6,a8,a10,a12};
      for(int t=0;t<4;++t){ if(!pres[t]) continue; int sz=SZ[t];
        bool bad=false; for(int p=0;p<sz;++p) if(strictBank[t][p]&&strictKeep[t][p]) bad=true;
        // cutoff: bank iff penalty <= T; T = highest p that is ever strict-bank.
        int T=-1; for(int p=0;p<sz;++p) if(strictBank[t][p]) T=p;
        cutFace[S][t] = sz - T;        // bank iff face >= this (T=-1 -> sz+1 = never)
        if(T<0) cutFace[S][t]=sz+1;
        if(bad) sep[S]=false;
      }
      if(!sep[S]) ++nNon;
    }

    printf("STRICT separability: %d / 104 states have a genuinely NON-separable optimal action.\n\n", nNon);
    printf("Optimal bank cutoffs (bank iff face >= X; '-' absent; '*' = type non-separable here):\n");
    printf("state(a6,a8,a10,a12) | d6  d8  d10 d12 | V(S)\n");
    for(int a12=0;a12<=1;++a12)for(int a10=0;a10<=1;++a10)for(int a8=0;a8<=1;++a8)for(int a6=12;a6>=0;--a6){
      if(a6+a8+a10+a12==0) continue; int S=idx(a6,a8,a10,a12); int pres[4]={a6,a8,a10,a12};
      bool sb[4][12]; // recompute per-type non-sep flag for display via strict arrays not stored; approximate with sep[S]
      (void)sb;
      char b[4][6];
      for(int t=0;t<4;++t){ if(!pres[t]){snprintf(b[t],6," - ");}
        else { int c=cutFace[S][t]; if(c>SZ[t]) snprintf(b[t],6," x ");   // never voluntarily bank
               else snprintf(b[t],6,"%2d ",c); } }
      printf("    (%2d,%d,%d,%d)        %s %s %s %s  %s %.3f\n",a6,a8,a10,a12,b[0],b[1],b[2],b[3],
             sep[S]?"  ":"NS",V[S]);
    }
    return 0;
}
