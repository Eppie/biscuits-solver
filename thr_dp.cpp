// EXACT expected score of the THRESHOLD (Fetterman Blackjack-table) strategy,
// computed by the same 104-state DP but with the FIXED policy instead of the
// optimal max. Zero RNG -> if this equals our Monte Carlo (8.485), the sim is
// unbiased; if it equals Fetterman's 8.53, our policy/RNG has a skew.
//
// Policy: with `total` dice in hand, bank every die whose penalty <= THR[size][total];
// if none qualify, bank the single die that "differs from its cutoff by the least"
// (min penalty - THR). THR[s][D] = floor(W[s][D-1]) matches his Fig 7 exactly.
//
// Build: c++ -O3 -mcpu=native -o thr_dp thr_dp.cpp

#include <cstdio>
#include <cmath>
#include <chrono>

static inline int idx(int a6,int a8,int a10,int a12){ return ((a6*2+a8)*2+a10)*2+a12; }
static double V[104];
static double fact[13];
static int THR[13][16];

int main(){
    // threshold table: bank if penalty <= THR[s][D], D = dice left
    for(int s:{6,8,10,12}){ double W[16]; W[1]=(s-1)/2.0;
        for(int m=2;m<=15;++m){double c=W[m-1],a=0;for(int p=0;p<s;++p)a+=(p<c?p:c);W[m]=a/s;}
        for(int D=0;D<=15;++D) THR[s][D]=(D<=1)?s:(int)W[D-1]; }

    fact[0]=1; for(int i=1;i<13;++i) fact[i]=fact[i-1]*i;
    double pw6[13]; for(int a=0;a<13;++a) pw6[a]=pow(6.0,a);

    auto t0=std::chrono::steady_clock::now();
    V[idx(0,0,0,0)]=0.0;

    for(int total=1; total<=15; ++total){
      for(int a6=0;a6<=12;++a6) for(int a8=0;a8<=1;++a8)
      for(int a10=0;a10<=1;++a10) for(int a12=0;a12<=1;++a12){
        if(a6+a8+a10+a12!=total) continue;
        const int T6=THR[6][total],T8=THR[8][total],T10=THR[10][total],T12=THR[12][total];
        double Vacc=0.0;

        for(int n1=0;n1<=a6;++n1)
        for(int n2=0;n2<=a6-n1;++n2)
        for(int n3=0;n3<=a6-n1-n2;++n3)
        for(int n4=0;n4<=a6-n1-n2-n3;++n4)
        for(int n5=0;n5<=a6-n1-n2-n3-n4;++n5){
            int n6=a6-n1-n2-n3-n4-n5;
            double w6=(fact[a6]/(fact[n1]*fact[n2]*fact[n3]*fact[n4]*fact[n5]*fact[n6]))/pw6[a6];
            int cntp[6]={n1,n2,n3,n4,n5,n6};   // counts of d6 penalty 5,4,3,2,1,0

            // d6 banked under threshold: those with penalty <= T6 (low-pen dice).
            int kept6=0; double keptpen6=0;    // kept = high-penalty d6 (pen > T6)
            int lowpen6=99;                    // smallest d6 penalty present (for forced)
            for(int c=0,pen=5; c<6; ++c,--pen){
                if(cntp[c]>0 && pen<lowpen6) lowpen6=pen;
                if(pen>T6){ kept6+=cntp[c]; keptpen6+=(double)pen*cntp[c]; }
            }

            double w8=a8?1.0/8:1.0, w10=a10?1.0/10:1.0, w12=a12?1.0/12:1.0;
            for(int p8=0;p8<=(a8?7:0);++p8)
            for(int p10=0;p10<=(a10?9:0);++p10)
            for(int p12=0;p12<=(a12?11:0);++p12){
                // big dice kept iff penalty > threshold
                int k8 =(a8 &&p8 >T8 )?1:0;
                int k10=(a10&&p10>T10)?1:0;
                int k12=(a12&&p12>T12)?1:0;
                int kept6e=kept6; double keptpen=keptpen6;
                if(k8){keptpen+=p8;} if(k10){keptpen+=p10;} if(k12){keptpen+=p12;}

                double totpen = keptpen6 + (a8?0:0); // placeholder, recompute below
                // total penalty over all dice this roll:
                double tot = 0;
                for(int c=0,pen=5;c<6;++c,--pen) tot += (double)pen*cntp[c];
                if(a8) tot+=p8; if(a10) tot+=p10; if(a12) tot+=p12;
                (void)totpen;

                int K6=kept6e,K8=k8,K10=k10,K12=k12;
                double banked = tot - keptpen;     // banked = low-penalty dice

                // Forced: nothing qualified to bank (kept everything) -> bank the
                // single die with min (penalty - its threshold).
                if(K6==a6 && K8==a8 && K10==a10 && K12==a12){
                    double bestdiff=1e300; int rm=0; double rmpen=0; // rm: 6/8/10/12
                    if(a6){ double d=lowpen6-T6; if(d<bestdiff){bestdiff=d;rm=6;rmpen=lowpen6;} }
                    if(a8){ double d=p8-T8;   if(d<bestdiff){bestdiff=d;rm=8;rmpen=p8;} }
                    if(a10){double d=p10-T10; if(d<bestdiff){bestdiff=d;rm=10;rmpen=p10;} }
                    if(a12){double d=p12-T12; if(d<bestdiff){bestdiff=d;rm=12;rmpen=p12;} }
                    if(rm==6)K6--; else if(rm==8)K8--; else if(rm==10)K10--; else K12--;
                    banked = rmpen;
                }

                double minbanked = banked + V[idx(K6,K8,K10,K12)];
                Vacc += w6*w8*w10*w12*minbanked;
            }
        }
        V[idx(a6,a8,a10,a12)]=Vacc;
      }
    }
    auto t1=std::chrono::steady_clock::now();
    printf("EXACT threshold-strategy expected score V(12,1,1,1) = %.6f\n", V[idx(12,1,1,1)]);
    printf("(our Monte Carlo was 8.485; Fetterman reported 8.53; compute %.2f s)\n",
           std::chrono::duration<double>(t1-t0).count());
    return 0;
}
