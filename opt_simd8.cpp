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

static uint64_t RS0=0x243F6A8885A308D3ULL, RS1=0x13198A2E03707344ULL;
static inline uint64_t xr(){ uint64_t x=RS0,y=RS1; RS0=y; x^=x<<23; RS1=x^y^(x>>17)^(y>>26); return RS1+y; }
static inline int roll(int sides){ return (int)(((xr()>>32)*(uint64_t)sides)>>32); } // penalty 0..sides-1

int main(int argc,char**argv){
    long N = (argc>1)?atol(argv[1]):20000000;
    solve();
    printf("optimal V(12,1,1,1) = %.6f  (target)\n", V[idx(12,1,1,1)]);
    static float Vf[104]; for(int i=0;i<104;++i) Vf[i]=(float)V[i];

    // fast RNG (xorshift128+) seeded above
    double sum=0,sumsq=0; long hist[64]={0}; double rolls=0;
    auto t0=std::chrono::steady_clock::now();
    for(long t=0;t<N;++t){
        int a6=12,a8=1,a10=1,a12=1; int sc=0;
        while(a6+a8+a10+a12>0){
            rolls++;
            // roll d6 -> histogram of penalties (0..5), build top-k prefix sums (no sort)
            int h[6]={0,0,0,0,0,0}; for(int j=0;j<a6;++j) h[roll(6)]++;
            int S6[13]; S6[0]=0; { int k=0; for(int pen=5;pen>=0;--pen) for(int c=0;c<h[pen];++c){ S6[k+1]=S6[k]+pen; ++k; } }
            int p8=a8?roll(8):0, p10=a10?roll(10):0, p12=a12?roll(12):0;
            int totpen = S6[a6] + (a8?p8:0)+(a10?p10:0)+(a12?p12:0);

            // 8-WIDE over big-configs: V[k6*8 + b] for b=0..7 is contiguous, so
            // run 8 INDEPENDENT max-chains over k6 (elementwise, vectorizable),
            // then combine. bigkept/valid precomputed per roll.
            float bigkept[8]; int valid[8];
            for(int b=0;b<8;++b){ int k8=(b>>2)&1,k10=(b>>1)&1,k12=b&1;
                valid[b] = (k8<=a8)&(k10<=a10)&(k12<=a12);
                bigkept[b]=(float)((k8?p8:0)+(k10?p10:0)+(k12?p12:0)); }
            int bfull=a8*4+a10*2+a12;
            float mv[8]; int mk[8];
            for(int b=0;b<8;++b){ mv[b]=-1e30f; mk[b]=0; }
            for(int k6=0;k6<=a6;++k6){
                float base=(float)S6[k6]; const float* vr=Vf+k6*8;
                int isfull=(k6==a6);
                for(int b=0;b<8;++b){                       // vectorizable, 8 indep chains
                    float v=base+bigkept[b]-vr[b];
                    int ok = valid[b] & ~(isfull & (b==bfull));
                    v = ok ? v : -1e30f;
                    int better=v>mv[b]; mv[b]=better?v:mv[b]; mk[b]=better?k6:mk[b];
                }
            }
            float best=-1e30f; int bk6=0,bidx=0;
            for(int b=0;b<8;++b){ int better=mv[b]>best; best=better?mv[b]:best; bk6=better?mk[b]:bk6; bidx=better?b:bidx; }
            int bk8=bidx>>2, bk10=(bidx>>1)&1, bk12=bidx&1;
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
