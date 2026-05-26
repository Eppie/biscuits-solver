// Monte-Carlo validation of the PROVABLY OPTIMAL policy.
// Compute optimal V (DP), then play games choosing, each roll, the kept-set K
// that maximizes (sum of kept penalties) - V(K). The average score must
// converge to V(12,1,1,1) = 8.0879 if both the DP value and the greedy-on-V
// action are correct.
//
// Build: c++ -O3 -mcpu=native -o opt_mc opt_mc.cpp

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <chrono>
#include <thread>
#include <vector>

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

static inline uint64_t xr(uint64_t&s0,uint64_t&s1){ uint64_t x=s0,y=s1; s0=y; x^=x<<23; s1=x^y^(x>>17)^(y>>26); return s1+y; }
static inline int roll(uint64_t&s0,uint64_t&s1,int sides){ return (int)(((xr(s0,s1)>>32)*(uint64_t)sides)>>32); }
struct Stat{ double sum=0,sumsq=0,rolls=0; long hist[64]={0}; };


static void worker(long games, uint64_t seed, Stat& st){
    uint64_t s0=seed*0x9E3779B97F4A7C15ULL+1, s1=seed*0xD1B54A32D192ED03ULL+0x9E3779B9ULL;
    for(int w=0;w<4;++w) xr(s0,s1);
    for(long t=0;t<games;++t){
        int a6=12,a8=1,a10=1,a12=1; int sc=0;
        while(a6+a8+a10+a12>0){
            st.rolls++;
            int h[6]={0,0,0,0,0,0}; for(int j=0;j<a6;++j) h[roll(s0,s1,6)]++;
            int S6[13]; S6[0]=0; { int k=0; for(int pen=5;pen>=0;--pen) for(int c=0;c<h[pen];++c){ S6[k+1]=S6[k]+pen; ++k; } }
            int p8=a8?roll(s0,s1,8):0, p10=a10?roll(s0,s1,10):0, p12=a12?roll(s0,s1,12):0;
            int totpen = S6[a6] + (a8?p8:0)+(a10?p10:0)+(a12?p12:0);
            float best=-1e30f; int bestc=0;
            for(int k8=0;k8<=a8;++k8)for(int k10=0;k10<=a10;++k10)for(int k12=0;k12<=a12;++k12){
                int bigidx=k8*4+k10*2+k12;
                float bigkept=(float)((k8?p8:0)+(k10?p10:0)+(k12?p12:0));
                const double* Vp=V+bigidx;
                int bigFull=(k8==a8&&k10==a10&&k12==a12);
                for(int k6=0;k6<=a6;++k6){
                    float v=(float)S6[k6]+bigkept-(float)Vp[k6*8];
                    v = (bigFull&(k6==a6)) ? -1e30f : v;
                    int comp=k6*8+bigidx; int better=v>best; best=better?v:best; bestc=better?comp:bestc;
                }
            }
            int bk6=bestc>>3, bidx=bestc&7, bk8=bidx>>2, bk10=(bidx>>1)&1, bk12=bidx&1;
            int keptpen = S6[bk6] + (bk8?p8:0)+(bk10?p10:0)+(bk12?p12:0);
            sc += totpen - keptpen; a6=bk6; a8=bk8; a10=bk10; a12=bk12;
        }
        st.sum+=sc; st.sumsq+=(double)sc*sc; if(sc<64) st.hist[sc]++;
    }
}
int main(int argc,char**argv){
    long N = (argc>1)?atol(argv[1]):20000000;
    solve();
    printf("optimal V(12,1,1,1) = %.6f  (target)\n", V[idx(12,1,1,1)]);

    unsigned nth = (argc>2)?(unsigned)atoi(argv[2]):std::thread::hardware_concurrency(); if(!nth)nth=1;
    std::vector<Stat> sts(nth); std::vector<std::thread> th;
    auto t0=std::chrono::steady_clock::now();
    long per=N/nth;
    for(unsigned i=0;i<nth;++i){ long g=(i==nth-1)?(N-per*(nth-1)):per; th.emplace_back(worker,g,(uint64_t)i+1,std::ref(sts[i])); }
    for(auto&x:th)x.join();
    double sum=0,sumsq=0,rolls=0; long hist[64]={0};
    for(auto&x:sts){ sum+=x.sum; sumsq+=x.sumsq; rolls+=x.rolls; for(int i=0;i<64;++i)hist[i]+=x.hist[i]; }
    double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    double mean=sum/N, sd=sqrt(sumsq/N-mean*mean);
    printf("MC loop (excl. DP solve): %.2f s  ->  %.3f M games/s,  %.1f M moves/s  (%.2f rolls/game)\n",
           secs, N/secs/1e6, rolls/secs/1e6, rolls/(double)N);
    printf("threads: %u\n", nth);
    printf("MC optimal-policy mean = %.5f  (sd %.4f, %ld games, stderr %.5f)\n",
           mean, sd, N, sd/sqrt((double)N));
    printf("difference from exact optimum = %+.5f\n", mean - V[idx(12,1,1,1)]);
    return 0;
}
