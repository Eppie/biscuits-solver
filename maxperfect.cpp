// Policy that MAXIMIZES P(perfect game), and its probability.
// To finish at 0 every banked die must be a max face (penalty 0). So you only
// ever bank max-faced dice; the choice each roll is WHICH nonempty subset of the
// max-faced dice to bank (bank fewer => keep more dice in hand to generate maxes
// next roll, but more rolls to survive). If no die shows its max you are forced
// to bank a non-max => run ends imperfect.
//
//   Pmax(S) = E_Z[ Z=empty ? 0 : max over nonempty B<=Z of Pmax(S - B) ]
//
// where Z = multiset of dice showing their max this roll (independent per die).
// Build: c++ -O3 -mcpu=native -o maxperfect maxperfect.cpp

#include <cstdio>
#include <cmath>
#include <random>
#include <algorithm>

static inline int idx(int a6,int a8,int a10,int a12){ return ((a6*2+a8)*2+a10)*2+a12; }
static double Pm[104];
static double C[13][13];
static const double Q[4]={1.0/6,1.0/8,1.0/10,1.0/12}; // P(die shows its max)

static inline double binom(int n,int k,double q){ return C[n][k]*pow(q,k)*pow(1-q,n-k); }

int main(int argc,char**argv){
    long N=(argc>1)?atol(argv[1]):50000000;
    for(int n=0;n<13;++n){ C[n][0]=1; for(int k=1;k<=n;++k) C[n][k]=C[n-1][k-1]+C[n-1][k]; }

    Pm[idx(0,0,0,0)]=1.0;
    bool alwaysBankAll=true;
    for(int total=1; total<=15; ++total)
     for(int a6=0;a6<=12;++a6)for(int a8=0;a8<=1;++a8)for(int a10=0;a10<=1;++a10)for(int a12=0;a12<=1;++a12){
      if(a6+a8+a10+a12!=total) continue; double acc=0;
      // sum over Z = (z6,z8,z10,z12) dice showing max
      for(int z6=0;z6<=a6;++z6)for(int z8=0;z8<=a8;++z8)for(int z10=0;z10<=a10;++z10)for(int z12=0;z12<=a12;++z12){
        if(z6+z8+z10+z12==0) continue;   // no max shown -> forced imperfect (0)
        double pz=binom(a6,z6,Q[0])*binom(a8,z8,Q[1])*binom(a10,z10,Q[2])*binom(a12,z12,Q[3]);
        // choose nonempty B<=Z maximizing Pm[S-B]
        double bestc=-1; int bb6=0,bb8=0,bb10=0,bb12=0;
        for(int b6=0;b6<=z6;++b6)for(int b8=0;b8<=z8;++b8)for(int b10=0;b10<=z10;++b10)for(int b12=0;b12<=z12;++b12){
          if(b6+b8+b10+b12==0) continue;
          double c=Pm[idx(a6-b6,a8-b8,a10-b10,a12-b12)];
          if(c>bestc){ bestc=c; bb6=b6;bb8=b8;bb10=b10;bb12=b12; }
        }
        if(!(bb6==z6&&bb8==z8&&bb10==z10&&bb12==z12)) alwaysBankAll=false; // kept a maxed die
        acc += pz*bestc;
      }
      Pm[idx(a6,a8,a10,a12)]=acc;
     }

    double p=Pm[idx(12,1,1,1)];
    printf("MAX P(perfect) = %.6e  = 1 in %.0f  (%.4f%%)\n", p, 1.0/p, 100*p);
    printf("optimal max-perfect policy is 'bank EVERY max-faced die each roll': %s\n",
           alwaysBankAll? "YES" : "NO (sometimes keep a max-faced die)");
    printf("(for reference, under expected-optimal play it was 1.613e-3 = 1 in 620)\n");

    // MC of the ACTUAL optimal max-perfect policy (choose bank-subset via Pm)
    std::mt19937_64 g(0xFEED);
    long zeros=0;
    for(long t=0;t<N;++t){
        int a6=12,a8=1,a10=1,a12=1; bool perfect=true;
        while(a6+a8+a10+a12>0){
            int z6=0; for(int j=0;j<a6;++j) if(g()%6==5) z6++;
            int z8 =a8 ?(g()%8==7):0, z10=a10?(g()%10==9):0, z12=a12?(g()%12==11):0;
            if(z6+z8+z10+z12==0){ perfect=false; break; }
            double bc=-1; int bb6=0,bb8=0,bb10=0,bb12=0;
            for(int b6=0;b6<=z6;++b6)for(int b8=0;b8<=z8;++b8)for(int b10=0;b10<=z10;++b10)for(int b12=0;b12<=z12;++b12){
              if(b6+b8+b10+b12==0) continue;
              double c=Pm[idx(a6-b6,a8-b8,a10-b10,a12-b12)];
              if(c>bc){bc=c;bb6=b6;bb8=b8;bb10=b10;bb12=b12;}
            }
            a6-=bb6;a8-=bb8;a10-=bb10;a12-=bb12;
        }
        if(perfect) zeros++;
    }
    printf("MC (optimal max-perfect policy) P(perfect) = %.6e  (%ld/%ld)\n",(double)zeros/N,zeros,N);
    return 0;
}
