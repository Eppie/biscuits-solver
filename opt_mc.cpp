// Monte-Carlo validation of the PROVABLY OPTIMAL policy.
// Compute optimal V (DP), then play games choosing, each roll, the kept-set K
// that maximizes (sum of kept penalties) - V(K). The average score must
// converge to V(12,1,1,1) = 8.0879 if both the DP value and the greedy-on-V
// action are correct.
//
// Build: c++ -O3 -mcpu=native -o opt_mc opt_mc.cpp

#include <cstdio>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>

static inline int idx(int a6,int a8,int a10,int a12){ return ((a6*2+a8)*2+a10)*2+a12; }
static double V[104];
static double fact[13], pw6[13];

static void solve(){
    fact[0]=1; for(int i=1;i<13;++i) fact[i]=fact[i-1]*i;
    for(int a=0;a<13;++a) pw6[a]=pow(6.0,a);
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

int main(int argc,char**argv){
    long N = (argc>1)?atol(argv[1]):20000000;
    solve();
    printf("optimal V(12,1,1,1) = %.6f  (target)\n", V[idx(12,1,1,1)]);

    std::mt19937_64 g(0xBADC0FFEE);
    double sum=0,sumsq=0; long hist[64]={0}; double rolls=0;
    auto t0=std::chrono::steady_clock::now();
    for(long t=0;t<N;++t){
        int a6=12,a8=1,a10=1,a12=1; int sc=0;
        while(a6+a8+a10+a12>0){
            rolls++;
            // roll
            int pe[12]; for(int j=0;j<a6;++j) pe[j]=g()%6;
            std::sort(pe,pe+a6,std::greater<int>());        // penalties desc
            int S6[13]; S6[0]=0; for(int k=0;k<a6;++k) S6[k+1]=S6[k]+pe[k];
            int p8=a8?g()%8:0, p10=a10?g()%10:0, p12=a12?g()%12:0;
            int totpen = S6[a6] + (a8?p8:0)+(a10?p10:0)+(a12?p12:0);

            // greedy on V: pick kept composition maximizing kept_pen - V(K)
            double best=-1e300; int bk6=0,bk8=0,bk10=0,bk12=0;
            for(int k6=0;k6<=a6;++k6)for(int k8=0;k8<=a8;++k8)for(int k10=0;k10<=a10;++k10)for(int k12=0;k12<=a12;++k12){
                if(k6==a6&&k8==a8&&k10==a10&&k12==a12) continue;
                double val = S6[k6] + (k8?p8:0)+(k10?p10:0)+(k12?p12:0) - V[idx(k6,k8,k10,k12)];
                if(val>best){ best=val; bk6=k6;bk8=k8;bk10=k10;bk12=k12; }
            }
            int keptpen = S6[bk6] + (bk8?p8:0)+(bk10?p10:0)+(bk12?p12:0);
            sc += totpen - keptpen;                          // banked penalty
            a6=bk6; a8=bk8; a10=bk10; a12=bk12;
        }
        sum+=sc; sumsq+=(double)sc*sc; if(sc<64) hist[sc]++;
    }
    double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    double mean=sum/N, sd=sqrt(sumsq/N-mean*mean);
    printf("MC loop (excl. DP solve): %.2f s  ->  %.3f M games/s,  %.1f M moves/s  (%.2f rolls/game)\n",
           secs, N/secs/1e6, rolls/secs/1e6, rolls/(double)N);
    printf("MC optimal-policy mean = %.5f  (sd %.4f, %ld games, stderr %.5f)\n",
           mean, sd, N, sd/sqrt((double)N));
    printf("difference from exact optimum = %+.5f\n", mean - V[idx(12,1,1,1)]);
    return 0;
}
