// bitches (a dice game) — EXACT optimal policy via dynamic programming.
//
// State = multiset of dice still in hand: (a6, a8, a10, a12) with a6 in [0,12],
// a8/a10/a12 in {0,1}. 13*2*2*2 = 104 states. V(S) = optimal expected remaining
// score (lower is better). V(0,0,0,0)=0.
//
//   V(S) = sum over all roll outcomes  prob(outcome) * min over kept-sets K ⊊ S
//              [ (penalties of banked dice) + V(K) ]
//
// Within one roll, for a fixed kept-COUNT per die type you keep the highest-
// penalty dice (bank the good ones, reroll the bad ones), since V(K) depends
// only on the composition of K, not which specific dice. So the per-roll
// decision is a max over <=104 kept-compositions of  (kept top-penalty sum) -
// V(K), and banked = (total penalty) - (kept top sum). We enumerate every d6
// face-composition with its exact multinomial weight (no Monte Carlo), so V is
// exact and the induced policy is provably optimal.
//
// Build: c++ -O3 -march=native -o exact_dp exact_dp.cpp   (or -mcpu=native on ARM)

#include <cstdio>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>

static inline int idx(int a6,int a8,int a10,int a12){ return ((a6*2+a8)*2+a10)*2+a12; }

static double V[104];
static double fact[13];

int main(){
    fact[0]=1; for(int i=1;i<13;++i) fact[i]=fact[i-1]*i;
    double pw6[13]; for(int a=0;a<13;++a) pw6[a]=pow(6.0,a); // 6^a

    auto t0=std::chrono::steady_clock::now();
    V[idx(0,0,0,0)]=0.0;

    // Process states in increasing total dice count (every kept-set K is smaller).
    for(int total=1; total<=15; ++total){
      for(int a6=0;a6<=12;++a6) for(int a8=0;a8<=1;++a8)
      for(int a10=0;a10<=1;++a10) for(int a12=0;a12<=1;++a12){
        if(a6+a8+a10+a12!=total) continue;
        double Vacc=0.0;

        // Enumerate d6 face-composition n1..n6 (counts of faces 1..6), sum=a6.
        // penalty(face f) = 6-f, so face1->pen5 ... face6->pen0.
        for(int n1=0;n1<=a6;++n1)
        for(int n2=0;n2<=a6-n1;++n2)
        for(int n3=0;n3<=a6-n1-n2;++n3)
        for(int n4=0;n4<=a6-n1-n2-n3;++n4)
        for(int n5=0;n5<=a6-n1-n2-n3-n4;++n5){
            int n6=a6-n1-n2-n3-n4-n5;
            double w6=(fact[a6]/(fact[n1]*fact[n2]*fact[n3]*fact[n4]*fact[n5]*fact[n6]))/pw6[a6];

            // S6[k] = sum of the k highest d6 penalties (5's then 4's ...).
            double S6[13]; S6[0]=0; int k=0;
            int cntp[6]={n1,n2,n3,n4,n5,n6};     // counts of penalty 5,4,3,2,1,0
            for(int pen=5,c=0;pen>=0;--pen,++c)
                for(int j=0;j<cntp[c];++j){ S6[k+1]=S6[k]+pen; ++k; }
            double totpen6=S6[a6];

            // Big dice (d8,d10,d12): enumerate present ones' penalties uniformly.
            int p8lo=0,p8hi=a8?7:0;  double w8=a8?1.0/8:1.0;
            int p10lo=0,p10hi=a10?9:0; double w10=a10?1.0/10:1.0;
            int p12lo=0,p12hi=a12?11:0; double w12=a12?1.0/12:1.0;

            for(int p8=p8lo;p8<=p8hi;++p8)
            for(int p10=p10lo;p10<=p10hi;++p10)
            for(int p12=p12lo;p12<=p12hi;++p12){
                double S8[2]={0,(double)p8}, S10[2]={0,(double)p10}, S12[2]={0,(double)p12};
                double totpen = totpen6 + (a8?p8:0) + (a10?p10:0) + (a12?p12:0);

                // max over kept compositions K ⊊ S of (kept top sum) - V(K).
                double best=-1e300;
                for(int k6=0;k6<=a6;++k6)
                for(int k8=0;k8<=a8;++k8)
                for(int k10=0;k10<=a10;++k10)
                for(int k12=0;k12<=a12;++k12){
                    if(k6==a6&&k8==a8&&k10==a10&&k12==a12) continue; // must bank >=1
                    double val=S6[k6]+S8[k8]+S10[k10]+S12[k12]-V[idx(k6,k8,k10,k12)];
                    if(val>best) best=val;
                }
                double minbanked = totpen - best;
                Vacc += w6*w8*w10*w12*minbanked;
            }
        }
        V[idx(a6,a8,a10,a12)]=Vacc;
      }
    }

    auto t1=std::chrono::steady_clock::now();
    double secs=std::chrono::duration<double>(t1-t0).count();

    printf("EXACT optimal expected score V(12,1,1,1) = %.6f\n", V[idx(12,1,1,1)]);
    printf("(compute time %.2f s)\n\n", secs);

    printf("Optimal V for selected states (a6,a8,a10,a12):\n");
    int show[][4]={{12,1,1,1},{12,0,0,0},{6,1,1,1},{6,0,0,0},{3,0,0,0},{1,0,0,0},
                   {0,0,0,1},{0,0,1,0},{0,1,0,0},{1,1,1,1},{2,1,1,1}};
    for(auto&s:show) printf("  V(%2d,%d,%d,%d) = %.4f\n",s[0],s[1],s[2],s[3],V[idx(s[0],s[1],s[2],s[3])]);
    return 0;
}
