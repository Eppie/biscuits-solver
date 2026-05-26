// "Decision card" variant: precompute per-big-config rank thresholds
// marg[b][j] = V(keep j d6 | big-config b) - V(keep j-1 d6 | b), then each roll
// keep the j-th-worst d6 while its penalty > marg[b][j] (stop at first that isn't),
// choosing the big-config by resulting value. Tests whether this is (a) faster
// than the argmax and (b) still optimal.
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <chrono>
static inline int idx(int a6,int a8,int a10,int a12){return ((a6*2+a8)*2+a10)*2+a12;}
static double V[104]; static double fact[13],pw6[13]; static float marg[8][13];
static void solve(){
    fact[0]=1;for(int i=1;i<13;++i)fact[i]=fact[i-1]*i; for(int a=0;a<13;++a)pw6[a]=pow(6.0,a);
    V[0]=0;
    for(int total=1;total<=15;++total)for(int a6=0;a6<=12;++a6)for(int a8=0;a8<=1;++a8)for(int a10=0;a10<=1;++a10)for(int a12=0;a12<=1;++a12){
      if(a6+a8+a10+a12!=total)continue;double Vacc=0;
      for(int n1=0;n1<=a6;++n1)for(int n2=0;n2<=a6-n1;++n2)for(int n3=0;n3<=a6-n1-n2;++n3)for(int n4=0;n4<=a6-n1-n2-n3;++n4)for(int n5=0;n5<=a6-n1-n2-n3-n4;++n5){
        int n6=a6-n1-n2-n3-n4-n5;double w6=(fact[a6]/(fact[n1]*fact[n2]*fact[n3]*fact[n4]*fact[n5]*fact[n6]))/pw6[a6];
        double S6[13];S6[0]=0;{int k=0,cp[6]={n1,n2,n3,n4,n5,n6};for(int pen=5,c=0;pen>=0;--pen,++c)for(int j=0;j<cp[c];++j){S6[k+1]=S6[k]+pen;++k;}}
        double w8=a8?1.0/8:1,w10=a10?1.0/10:1,w12=a12?1.0/12:1;
        for(int p8=0;p8<=(a8?7:0);++p8)for(int p10=0;p10<=(a10?9:0);++p10)for(int p12=0;p12<=(a12?11:0);++p12){
          double S8[2]={0,(double)p8},S10[2]={0,(double)p10},S12[2]={0,(double)p12};
          double tp=S6[a6]+(a8?p8:0)+(a10?p10:0)+(a12?p12:0);double best=-1e300;
          for(int k6=0;k6<=a6;++k6)for(int k8=0;k8<=a8;++k8)for(int k10=0;k10<=a10;++k10)for(int k12=0;k12<=a12;++k12){
            if(k6==a6&&k8==a8&&k10==a10&&k12==a12)continue;
            double v=S6[k6]+S8[k8]+S10[k10]+S12[k12]-V[idx(k6,k8,k10,k12)];if(v>best)best=v;}
          Vacc+=w6*w8*w10*w12*(tp-best);}
      }
      V[idx(a6,a8,a10,a12)]=Vacc;}
    for(int b=0;b<8;++b)for(int j=1;j<=12;++j) marg[b][j]=(float)(V[j*8+b]-V[(j-1)*8+b]);
}
static uint64_t S0=0x243F6A8885A308D3ULL,S1=0x13198A2E03707344ULL;
static inline uint64_t xr(){uint64_t x=S0,y=S1;S0=y;x^=x<<23;S1=x^y^(x>>17)^(y>>26);return S1+y;}
static inline int roll(int s){return (int)(((xr()>>32)*(uint64_t)s)>>32);}
int main(int argc,char**argv){
    long N=(argc>1)?atol(argv[1]):20000000; solve();
    printf("optimal V(12,1,1,1)=%.6f\n",V[idx(12,1,1,1)]);
    double sum=0,rolls=0; auto t0=std::chrono::steady_clock::now();
    for(long t=0;t<N;++t){int a6=12,a8=1,a10=1,a12=1;int sc=0;
      while(a6+a8+a10+a12>0){ rolls++;
        int h[6]={0,0,0,0,0,0}; for(int j=0;j<a6;++j) h[roll(6)]++;
        int pe[13]; {int k=0; for(int pen=5;pen>=0;--pen) for(int c=0;c<h[pen];++c) pe[++k]=pen;}  // pe[1..a6] desc
        int S6[13]; S6[0]=0; for(int k=1;k<=a6;++k) S6[k]=S6[k-1]+pe[k];
        int p8=a8?roll(8):0,p10=a10?roll(10):0,p12=a12?roll(12):0;
        int totpen=S6[a6]+(a8?p8:0)+(a10?p10:0)+(a12?p12:0);
        // CARD: for each big-config, k6 = keep-while penalty>marg (stop at first), pick best by value
        float best=-1e30f; int bk6=0,bk8=0,bk10=0,bk12=0;
        for(int k8=0;k8<=a8;++k8)for(int k10=0;k10<=a10;++k10)for(int k12=0;k12<=a12;++k12){
          int b=k8*4+k10*2+k12; float bigkept=(float)((k8?p8:0)+(k10?p10:0)+(k12?p12:0));
          int k6=0; while(k6<a6 && (float)pe[k6+1] > marg[b][k6+1]) k6++;   // threshold scan
          int bigFull=(k8==a8&&k10==a10&&k12==a12);
          if(bigFull && k6==a6) k6--;                                       // can't keep everything
          if(k6<0) k6=0;
          float v=(float)S6[k6]+bigkept-(float)V[k6*8+b];
          if(v>best){best=v;bk6=k6;bk8=k8;bk10=k10;bk12=k12;}
        }
        int keptpen=S6[bk6]+(bk8?p8:0)+(bk10?p10:0)+(bk12?p12:0);
        sc+=totpen-keptpen; a6=bk6;a8=bk8;a10=bk10;a12=bk12;
      }
      sum+=sc;}
    double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    double mean=sum/N;
    printf("CARD: %.3f M games/s, %.1f M moves/s  | mean=%.5f (optimal=%.5f, gap=%+.5f)\n",
      N/secs/1e6, rolls/secs/1e6, mean, V[idx(12,1,1,1)], mean-V[idx(12,1,1,1)]);
    return 0;
}
