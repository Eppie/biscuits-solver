// Optimal-policy MC, restructured for SIMD-across-games via STATE BUCKETING.
// All games in a bucket share the same state, so the per-move argmax is uniform
// and vectorizes across games with no wasted lanes. Games flow strictly downward
// in total dice, so we process states by decreasing total and never revisit one.
//
// Build: c++ -O3 -mcpu=native -o opt_bucket opt_bucket.cpp

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <chrono>

static inline int idx(int a6,int a8,int a10,int a12){ return ((a6*2+a8)*2+a10)*2+a12; }
static double V[104]; static float Vf[104]; static double fact[13],pw6[13];

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
        double tp6=S6[a6]; double w8=a8?1.0/8:1,w10=a10?1.0/10:1,w12=a12?1.0/12:1;
        for(int p8=0;p8<=(a8?7:0);++p8)for(int p10=0;p10<=(a10?9:0);++p10)for(int p12=0;p12<=(a12?11:0);++p12){
          double S8[2]={0,(double)p8},S10[2]={0,(double)p10},S12[2]={0,(double)p12};
          double totpen=tp6+(a8?p8:0)+(a10?p10:0)+(a12?p12:0); double best=-1e300;
          for(int k6=0;k6<=a6;++k6)for(int k8=0;k8<=a8;++k8)for(int k10=0;k10<=a10;++k10)for(int k12=0;k12<=a12;++k12){
            if(k6==a6&&k8==a8&&k10==a10&&k12==a12) continue;
            double v=S6[k6]+S8[k8]+S10[k10]+S12[k12]-V[idx(k6,k8,k10,k12)]; if(v>best)best=v;
          }
          Vacc += w6*w8*w10*w12*(totpen-best);
        }
      }
      V[idx(a6,a8,a10,a12)]=Vacc;
     }
    for(int i=0;i<104;++i) Vf[i]=(float)V[i];
}

static inline uint64_t xs(uint64_t&s){ s^=s<<13; s^=s>>7; s^=s<<17; return s; }
static inline int draw(uint64_t&s,int sides){ return (int)(((xs(s)>>32)*(uint64_t)sides)>>32); }

static constexpr int CH = 1024;   // chunk: S6arr (13*CH*4 = 52KB) fits L1

// per-state game pools (SoA): rng state + accumulated score
static std::vector<uint64_t> brng[104];
static std::vector<int>      bscore[104];

int main(int argc,char**argv){
    long N = (argc>1)?atol(argv[1]):20000000;
    solve();
    printf("optimal V(12,1,1,1) = %.6f\n", V[idx(12,1,1,1)]);

    // seed the start bucket
    int START=idx(12,1,1,1);
    brng[START].resize(N); bscore[START].assign(N,0);
    for(long i=0;i<N;++i){ uint64_t s=(uint64_t)(i+1)*0x9E3779B97F4A7C15ULL; s^=s>>31; if(!s)s=1; brng[START][i]=s; }

    double sum=0,sumsq=0; long done=0, rolls=0;
    alignas(64) static float  S6arr[13*CH];
    alignas(64) static int    p8a[CH],p10a[CH],p12a[CH];
    alignas(64) static float  best[CH]; alignas(64) static int bestc[CH];

    auto t0=std::chrono::steady_clock::now();

    for(int total=15; total>=1; --total)
     for(int a6=0;a6<=12;++a6)for(int a8=0;a8<=1;++a8)for(int a10=0;a10<=1;++a10)for(int a12=0;a12<=1;++a12){
      if(a6+a8+a10+a12!=total) continue;
      int S=idx(a6,a8,a10,a12); long m=(long)brng[S].size(); if(!m) continue;
      uint64_t* RNG=brng[S].data(); int* SC=bscore[S].data();
      int bfull = a8*4+a10*2+a12;

      for(long off=0; off<m; off+=CH){
        int n=(int)((m-off<CH)?(m-off):CH);
        // --- roll + per-game sorted prefix sums S6arr[k6*CH + i] ---
        for(int i=0;i<n;++i){
          uint64_t s=RNG[off+i];
          int h[6]={0,0,0,0,0,0}; for(int j=0;j<a6;++j) h[draw(s,6)]++;
          int k=0; S6arr[0*CH+i]=0;
          for(int pen=5;pen>=0;--pen) for(int c=0;c<h[pen];++c){ S6arr[(k+1)*CH+i]=S6arr[k*CH+i]+pen; ++k; }
          p8a[i] =a8 ?draw(s,8) :0;
          p10a[i]=a10?draw(s,10):0;
          p12a[i]=a12?draw(s,12):0;
          RNG[off+i]=s;
        }
        rolls += n;
        // --- argmax over comps, SIMD across games ---
        #pragma omp simd
        for(int i=0;i<n;++i){ best[i]=-1e30f; bestc[i]=0; }
        for(int k8=0;k8<=a8;++k8)for(int k10=0;k10<=a10;++k10)for(int k12=0;k12<=a12;++k12){
          int bigidx=k8*4+k10*2+k12; int bigFull=(k8==a8&&k10==a10&&k12==a12);
          for(int k6=0;k6<=a6;++k6){
            if(bigFull&&k6==a6) continue;            // skip illegal keep-everything
            float Vc=Vf[k6*8+bigidx]; int comp=k6*8+bigidx;
            const float* s6=&S6arr[k6*CH];
            #pragma omp simd
            for(int i=0;i<n;++i){
              float v=s6[i] + (float)((k8?p8a[i]:0)+(k10?p10a[i]:0)+(k12?p12a[i]:0)) - Vc;
              int better=v>best[i]; best[i]=better?v:best[i]; bestc[i]=better?comp:bestc[i];
            }
          }
        }
        // --- transition: bank, score, scatter to new bucket ---
        const float* s6top=&S6arr[a6*CH];
        for(int i=0;i<n;++i){
          int c=bestc[i], bk6=c>>3, bidx=c&7, bk8=bidx>>2, bk10=(bidx>>1)&1, bk12=bidx&1;
          int banked=(int)(s6top[i]-S6arr[bk6*CH+i]) + (bk8?0:p8a[i]) + (bk10?0:p10a[i]) + (bk12?0:p12a[i]);
          int ns=SC[off+i]+banked;
          int nstate=idx(bk6,bk8,bk10,bk12);
          if(nstate==0){ sum+=ns; sumsq+=(double)ns*ns; ++done; }
          else { brng[nstate].push_back(RNG[off+i]); bscore[nstate].push_back(ns); }
        }
      }
      std::vector<uint64_t>().swap(brng[S]); std::vector<int>().swap(bscore[S]); // free
     }

    double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    double mean=sum/done, sd=sqrt(sumsq/done-mean*mean);
    printf("BUCKETED-SIMD: %.2f s -> %.3f M games/s, %.1f M moves/s (%.2f rolls/game)\n",
           secs, done/secs/1e6, rolls/secs/1e6, (double)rolls/done);
    printf("mean = %.5f  (optimal=%.5f, gap=%+.5f), games=%ld\n",
           mean, V[idx(12,1,1,1)], mean-V[idx(12,1,1,1)], done);
    return 0;
}
