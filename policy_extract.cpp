// Extract the PROVABLY OPTIMAL policy for bitches and test separability.
//
// 1. DP over 104 states -> optimal value V[S] (lower = better).
// 2. For each state S, enumerate every roll exactly. For each roll the optimal
//    action keeps a set K ⊊ S maximizing  (sum of kept penalties) - V(K).
//    Within a type you keep the highest-penalty dice, so an action is a per-type
//    kept-count. We track, over all optimal actions of each roll, whether a die
//    of (type t, penalty p) is *forced banked* (banked in every optimal action),
//    *forced kept*, or free. A per-die threshold "bank iff penalty <= T_t" is
//    OPTIMAL iff no penalty p is forced-banked in one roll and forced-kept in
//    another. We derive T_t and flag non-separable states.
// 3. Verify: replay the extracted thresholds (forced = optimal single removal)
//    with optimal continuation V; if the one-step value equals V[S] for every
//    state, the threshold policy is provably globally optimal.
//
// Build: c++ -O3 -mcpu=native -o policy_extract policy_extract.cpp

#include <cstdio>
#include <cmath>
#include <algorithm>

static inline int idx(int a6,int a8,int a10,int a12){ return ((a6*2+a8)*2+a10)*2+a12; }
static double V[104];
static double fact[13], pw6[13];
static const int SZ[4]={6,8,10,12};

// penalty-cutoff per state per type (bank iff penalty<=Tt); -1 = never bank
static int Tt[104][4];
static bool nonsep[104];
static double oneStepGap[104];

int main(){
    fact[0]=1; for(int i=1;i<13;++i) fact[i]=fact[i-1]*i;
    for(int a=0;a<13;++a) pw6[a]=pow(6.0,a);

    // ---------- pass A: optimal DP ----------
    V[idx(0,0,0,0)]=0;
    for(int total=1; total<=15; ++total)
     for(int a6=0;a6<=12;++a6)for(int a8=0;a8<=1;++a8)for(int a10=0;a10<=1;++a10)for(int a12=0;a12<=1;++a12){
      if(a6+a8+a10+a12!=total) continue;
      double Vacc=0;
      for(int n1=0;n1<=a6;++n1)for(int n2=0;n2<=a6-n1;++n2)for(int n3=0;n3<=a6-n1-n2;++n3)
      for(int n4=0;n4<=a6-n1-n2-n3;++n4)for(int n5=0;n5<=a6-n1-n2-n3-n4;++n5){
        int n6=a6-n1-n2-n3-n4-n5;
        double w6=(fact[a6]/(fact[n1]*fact[n2]*fact[n3]*fact[n4]*fact[n5]*fact[n6]))/pw6[a6];
        double S6[13]; S6[0]=0; { int k=0,cp[6]={n1,n2,n3,n4,n5,n6};
          for(int pen=5,c=0;pen>=0;--pen,++c) for(int j=0;j<cp[c];++j){ S6[k+1]=S6[k]+pen; ++k; } }
        double totpen6=S6[a6];
        double w8=a8?1.0/8:1,w10=a10?1.0/10:1,w12=a12?1.0/12:1;
        for(int p8=0;p8<=(a8?7:0);++p8)for(int p10=0;p10<=(a10?9:0);++p10)for(int p12=0;p12<=(a12?11:0);++p12){
          double S8[2]={0,(double)p8},S10[2]={0,(double)p10},S12[2]={0,(double)p12};
          double totpen=totpen6+(a8?p8:0)+(a10?p10:0)+(a12?p12:0);
          double best=-1e300;
          for(int k6=0;k6<=a6;++k6)for(int k8=0;k8<=a8;++k8)for(int k10=0;k10<=a10;++k10)for(int k12=0;k12<=a12;++k12){
            if(k6==a6&&k8==a8&&k10==a10&&k12==a12) continue;
            double val=S6[k6]+S8[k8]+S10[k10]+S12[k12]-V[idx(k6,k8,k10,k12)];
            if(val>best) best=val;
          }
          Vacc += w6*w8*w10*w12*(totpen-best);
        }
      }
      V[idx(a6,a8,a10,a12)]=Vacc;
     }
    printf("OPTIMAL expected score V(12,1,1,1) = %.6f\n\n", V[idx(12,1,1,1)]);

    // ---------- pass B: per-state separability + threshold extraction ----------
    for(int s=0;s<104;++s){ nonsep[s]=false; for(int t=0;t<4;++t) Tt[s][t]=-1; }

    for(int a6=0;a6<=12;++a6)for(int a8=0;a8<=1;++a8)for(int a10=0;a10<=1;++a10)for(int a12=0;a12<=1;++a12){
      int total=a6+a8+a10+a12; if(total==0) continue;
      int S=idx(a6,a8,a10,a12);
      // canBank[t][p]/canKeep[t][p]: across VOLUNTARY rolls, is a (type t, penalty
      // p) die banked in some optimal action / kept in some optimal action? If a
      // penalty is BOTH banked-somewhere and kept-somewhere, no fixed per-die
      // threshold can reproduce the optimum -> non-separable.
      bool canBank[4][12]={}, canKeep[4][12]={};

      for(int n1=0;n1<=a6;++n1)for(int n2=0;n2<=a6-n1;++n2)for(int n3=0;n3<=a6-n1-n2;++n3)
      for(int n4=0;n4<=a6-n1-n2-n3;++n4)for(int n5=0;n5<=a6-n1-n2-n3-n4;++n5){
        int n6=a6-n1-n2-n3-n4-n5; int cp[6]={n1,n2,n3,n4,n5,n6}; // counts of pen 5,4,3,2,1,0
        double S6[13]; S6[0]=0; { int k=0; for(int pen=5,c=0;pen>=0;--pen,++c) for(int j=0;j<cp[c];++j){ S6[k+1]=S6[k]+pen; ++k; } }
        for(int p8=0;p8<=(a8?7:0);++p8)for(int p10=0;p10<=(a10?9:0);++p10)for(int p12=0;p12<=(a12?11:0);++p12){
          double S8[2]={0,(double)p8},S10[2]={0,(double)p10},S12[2]={0,(double)p12};
          // find optimal value M and the min/max kept-count per type over optimal actions
          double M=-1e300;
          for(int k6=0;k6<=a6;++k6)for(int k8=0;k8<=a8;++k8)for(int k10=0;k10<=a10;++k10)for(int k12=0;k12<=a12;++k12){
            if(k6==a6&&k8==a8&&k10==a10&&k12==a12) continue;
            double val=S6[k6]+S8[k8]+S10[k10]+S12[k12]-V[idx(k6,k8,k10,k12)];
            if(val>M) M=val;
          }
          // If keeping EVERYTHING would be best (but is illegal), this roll is a
          // FORCED bank — banking here is involuntary, so it tells us nothing
          // about the voluntary per-die threshold. Skip it for extraction.
          double totpen = S6[a6] + (a8?p8:0) + (a10?p10:0) + (a12?p12:0);
          double gFull  = totpen - V[S];
          if(gFull >= M - 1e-9) continue;
          int kmn[4]={99,99,99,99}, kmx[4]={-1,-1,-1,-1};
          for(int k6=0;k6<=a6;++k6)for(int k8=0;k8<=a8;++k8)for(int k10=0;k10<=a10;++k10)for(int k12=0;k12<=a12;++k12){
            if(k6==a6&&k8==a8&&k10==a10&&k12==a12) continue;
            double val=S6[k6]+S8[k8]+S10[k10]+S12[k12]-V[idx(k6,k8,k10,k12)];
            if(val>M-1e-9){ int kk[4]={k6,k8,k10,k12};
              for(int t=0;t<4;++t){ if(kk[t]<kmn[t])kmn[t]=kk[t]; if(kk[t]>kmx[t])kmx[t]=kk[t]; } }
          }
          // d6: for each penalty value p present, hi_p = #d6 with pen>p, eq = #d6 with pen==p
          if(a6){ for(int p=0;p<=5;++p){ int eq=cp[5-p]; if(!eq) continue;
              int hi=0; for(int pp=p+1;pp<=5;++pp) hi+=cp[5-pp];
              if(kmn[0] < hi+eq) canBank[0][p]=true;   // some optimal banks a pen-p d6
              if(kmx[0] > hi)    canKeep[0][p]=true;   // some optimal keeps a pen-p d6
          }}
          // big dice (single die each): hi=0, eq=1
          int pbig[4]={0,p8,p10,p12}; int present[4]={a6,a8,a10,a12};
          for(int t=1;t<4;++t){ if(!present[t]) continue; int p=pbig[t];
              if(kmn[t] < 1) canBank[t][p]=true;
              if(kmx[t] > 0) canKeep[t][p]=true;
          }
        }
      }
      // classify each present type
      int pres[4]={a6,a8,a10,a12};
      for(int t=0;t<4;++t){ if(!pres[t]) continue; int sz=SZ[t];
        bool bad=false;
        for(int p=0;p<sz;++p) if(canBank[t][p]&&canKeep[t][p]) bad=true; // banked & kept somewhere
        // separable threshold: bank iff penalty < (lowest penalty ever kept).
        int minkeep=999; for(int p=0;p<sz;++p) if(canKeep[t][p]&&p<minkeep) minkeep=p;
        Tt[S][t] = (minkeep==999) ? sz-1 : minkeep-1;   // never kept => bank all
        if(bad) nonsep[S]=true;
      }
    }

    // ---------- pass C: verify extracted thresholds are one-step optimal ----------
    int worst=-1; double worstgap=0;
    for(int a6=0;a6<=12;++a6)for(int a8=0;a8<=1;++a8)for(int a10=0;a10<=1;++a10)for(int a12=0;a12<=1;++a12){
      int total=a6+a8+a10+a12; if(total==0) continue; int S=idx(a6,a8,a10,a12);
      int T6=Tt[S][0],T8=Tt[S][1],T10=Tt[S][2],T12=Tt[S][3];
      double E=0;
      for(int n1=0;n1<=a6;++n1)for(int n2=0;n2<=a6-n1;++n2)for(int n3=0;n3<=a6-n1-n2;++n3)
      for(int n4=0;n4<=a6-n1-n2-n3;++n4)for(int n5=0;n5<=a6-n1-n2-n3-n4;++n5){
        int n6=a6-n1-n2-n3-n4-n5; int cp[6]={n1,n2,n3,n4,n5,n6};
        double w6=(fact[a6]/(fact[n1]*fact[n2]*fact[n3]*fact[n4]*fact[n5]*fact[n6]))/pw6[a6];
        // d6 kept (pen>T6) and banked penalty
        int kept6=0; double keptpen6=0; int lowpen6=99;
        for(int c=0,pen=5;c<6;++c,--pen){ if(cp[c]>0&&pen<lowpen6)lowpen6=pen; if(pen>T6){kept6+=cp[c];keptpen6+=(double)pen*cp[c];} }
        double totpen6=0; for(int c=0,pen=5;c<6;++c,--pen) totpen6+=(double)pen*cp[c];
        double w8=a8?1.0/8:1,w10=a10?1.0/10:1,w12=a12?1.0/12:1;
        for(int p8=0;p8<=(a8?7:0);++p8)for(int p10=0;p10<=(a10?9:0);++p10)for(int p12=0;p12<=(a12?11:0);++p12){
          int k8=(a8&&p8>T8),k10=(a10&&p10>T10),k12=(a12&&p12>T12);
          int K6=kept6,K8=k8,K10=k10,K12=k12;
          double keptpen=keptpen6+(k8?p8:0)+(k10?p10:0)+(k12?p12:0);
          double tot=totpen6+(a8?p8:0)+(a10?p10:0)+(a12?p12:0);
          double banked=tot-keptpen; int u6=K6,u8=K8,u10=K10,u12=K12;
          if(K6==a6&&K8==a8&&K10==a10&&K12==a12){ // forced: optimal single removal
            double bestv=1e300;
            if(a6){ double v=lowpen6+V[idx(a6-1,a8,a10,a12)]; if(v<bestv){bestv=v;u6=a6-1;u8=a8;u10=a10;u12=a12;} }
            if(a8){ double v=p8+V[idx(a6,a8-1,a10,a12)]; if(v<bestv){bestv=v;u6=a6;u8=a8-1;u10=a10;u12=a12;} }
            if(a10){double v=p10+V[idx(a6,a8,a10-1,a12)];if(v<bestv){bestv=v;u6=a6;u8=a8;u10=a10-1;u12=a12;} }
            if(a12){double v=p12+V[idx(a6,a8,a10,a12-1)];if(v<bestv){bestv=v;u6=a6;u8=a8;u10=a10;u12=a12-1;} }
            E += w6*w8*w10*w12*bestv; continue;
          }
          E += w6*w8*w10*w12*(banked+V[idx(u6,u8,u10,u12)]);
        }
      }
      double gap=E-V[S]; oneStepGap[S]=gap;
      if(gap>worstgap){worstgap=gap;worst=S;}
    }

    // ---------- pass D: full-game value of the extracted threshold policy ----------
    // (recursive: forced bank = optimal single removal, policy's own continuation)
    static double Vpol[104]; Vpol[idx(0,0,0,0)]=0;
    for(int total=1; total<=15; ++total)
     for(int a6=0;a6<=12;++a6)for(int a8=0;a8<=1;++a8)for(int a10=0;a10<=1;++a10)for(int a12=0;a12<=1;++a12){
      if(a6+a8+a10+a12!=total) continue; int S=idx(a6,a8,a10,a12);
      int T6=Tt[S][0],T8=Tt[S][1],T10=Tt[S][2],T12=Tt[S][3]; double E=0;
      for(int n1=0;n1<=a6;++n1)for(int n2=0;n2<=a6-n1;++n2)for(int n3=0;n3<=a6-n1-n2;++n3)
      for(int n4=0;n4<=a6-n1-n2-n3;++n4)for(int n5=0;n5<=a6-n1-n2-n3-n4;++n5){
        int n6=a6-n1-n2-n3-n4-n5; int cp[6]={n1,n2,n3,n4,n5,n6};
        double w6=(fact[a6]/(fact[n1]*fact[n2]*fact[n3]*fact[n4]*fact[n5]*fact[n6]))/pw6[a6];
        int kept6=0; double keptpen6=0; int lowpen6=99;
        for(int c=0,pen=5;c<6;++c,--pen){ if(cp[c]>0&&pen<lowpen6)lowpen6=pen; if(pen>T6){kept6+=cp[c];keptpen6+=(double)pen*cp[c];} }
        double totpen6=0; for(int c=0,pen=5;c<6;++c,--pen) totpen6+=(double)pen*cp[c];
        double w8=a8?1.0/8:1,w10=a10?1.0/10:1,w12=a12?1.0/12:1;
        for(int p8=0;p8<=(a8?7:0);++p8)for(int p10=0;p10<=(a10?9:0);++p10)for(int p12=0;p12<=(a12?11:0);++p12){
          int k8=(a8&&p8>T8),k10=(a10&&p10>T10),k12=(a12&&p12>T12);
          int K6=kept6,K8=k8,K10=k10,K12=k12; double keptpen=keptpen6+(k8?p8:0)+(k10?p10:0)+(k12?p12:0);
          double tot=totpen6+(a8?p8:0)+(a10?p10:0)+(a12?p12:0); double banked=tot-keptpen; int u6=K6,u8=K8,u10=K10,u12=K12;
          if(K6==a6&&K8==a8&&K10==a10&&K12==a12){ double bv=1e300;
            if(a6){double v=lowpen6+Vpol[idx(a6-1,a8,a10,a12)];if(v<bv){bv=v;u6=a6-1;u8=a8;u10=a10;u12=a12;}}
            if(a8){double v=p8+Vpol[idx(a6,a8-1,a10,a12)];if(v<bv){bv=v;u6=a6;u8=a8-1;u10=a10;u12=a12;}}
            if(a10){double v=p10+Vpol[idx(a6,a8,a10-1,a12)];if(v<bv){bv=v;u6=a6;u8=a8;u10=a10-1;u12=a12;}}
            if(a12){double v=p12+Vpol[idx(a6,a8,a10,a12-1)];if(v<bv){bv=v;u6=a6;u8=a8;u10=a10;u12=a12-1;}}
            E += w6*w8*w10*w12*bv; continue; }
          E += w6*w8*w10*w12*(banked+Vpol[idx(u6,u8,u10,u12)]);
        }
      }
      Vpol[S]=E;
     }

    // ---------- report ----------
    int nNon=0; for(int s=0;s<104;++s) if(nonsep[s]) ++nNon;
    printf("Separability: %d / 104 states have a NON-separable optimal action.\n", nNon);
    printf("Verification: extracted-threshold policy worst one-step gap vs optimal = %.2e\n", worstgap);
    printf("Full-game value of the extracted cutoff policy = %.4f  (optimal = %.4f, loss %.4f)\n",
           Vpol[idx(12,1,1,1)], V[idx(12,1,1,1)], Vpol[idx(12,1,1,1)]-V[idx(12,1,1,1)]);
    { int a6=worst/8, r=worst%8; (void)r;
      printf("worst one-step state idx=%d, gap=%.4f\n", worst, worstgap); (void)a6; }
    printf("\n");

    printf("Optimal bank cutoffs  (bank a die iff face >= cutoff;  '-' = type absent):\n");
    printf("state(a6,a8,a10,a12) | d6  d8  d10 d12 | sep? | V(S)\n");
    for(int a12=0;a12<=1;++a12)for(int a10=0;a10<=1;++a10)for(int a8=0;a8<=1;++a8)for(int a6=12;a6>=0;--a6){
      int total=a6+a8+a10+a12; if(total==0) continue; int S=idx(a6,a8,a10,a12);
      int pres[4]={a6,a8,a10,a12};
      char buf[4][6];
      for(int t=0;t<4;++t){ if(!pres[t]){ snprintf(buf[t],6," - "); }
        else { int cut=SZ[t]-Tt[S][t]; if(Tt[S][t]<0) cut=SZ[t]+1; // never bank
               snprintf(buf[t],6,"%2d ",cut); } }
      printf("    (%2d,%d,%d,%d)        %s %s %s %s   %s    %.3f\n",
             a6,a8,a10,a12,buf[0],buf[1],buf[2],buf[3], nonsep[S]?"NO ":"yes", V[S]);
    }
    return 0;
}
