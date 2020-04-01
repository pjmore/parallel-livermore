/************************************************************************
 * cc  lloops.c cpuidc.c -lm -O3 -o lloopsIL
 * ~/toolchain/raspbian-toolchain-gcc-4.7.2-linux32/bin/arm-linux-gnueabihf-gcc  lloops.c cpuidc.c -lm -O3 -march=armv6 -mfloat-abi=hard -mfpu=vfp -o liverloopsPiA6
 * 
 * L. L. N. L.  " C "  K E R N E L S:  M F L O P S  P C  V E R S I O N  *
 *
 *  #define Version not used
 *
 *  Different compilers can produce different floating point numeric
 *  results, probably due to compiling instructions in a different
 *  sequence. As the program checks these, they may need to be changed.
 *  The log file indicates non-standard results and these values can
 *  be copied and pasted into this program. See // COMPILER
 *  #define codes and function checkOut(). Some values are for
 *  optimised compiling and non-optimised results might be different.
 *
 *  Change #define options for print of optimisation level and

 * gcc  lloops2.c cpuidc.c -lm -lrt -O3 -mcpu=cortex-a7 -mfloat-abi=hard -mfpu=neon-vfpv4 -o liverloopsPiA7
 */
 
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <time.h> 
#include <unistd.h>
#include <sys/time.h>
#include <omp.h>

#include "cpuidh.h"

// COMPILER for numeric results see function checkOut()

// #define WATCOM
// #define VISUALC
#define GCCINTEL64
// #define GCCINTEL32
// #define CCINTEL32
// #define GCCARMDP
//#define GCCARMPI

typedef int   Boolean;
#define TRUE  1
#define FALSE 0
// ______________________ My functions and types for reductions and the like ___________________


#define FP_ERROR_THRESHOLD 0.01 

typedef struct{
    long index;
    double val;
}min_index;


// Reductions over value and also contains location data

//#define MIN_INDEX( value ) { 0, value }

int imin(int i, int j){
    if (i < j){
        return i;
    }
    return j;
}

int imax(int i, int j){
    if(i>j){
        return i;
    }
    return j;
}

min_index MIN_INDEX(){
    min_index new_min_index = {0, __DBL_MAX__};
    return new_min_index;
}

min_index reduce_min_index(min_index i, min_index j){
    if(i.val < j.val){
        return i;
    }
    return j;
}


// Reductions over the relevant variables in kernel 16. Since the kernel typically breaks out of loop as soon as certain conditions are met
// Reduce over values keeping the ones from the thread that triggered the break

typedef struct{
    long _k2;
    long _k3; 
    unsigned int tid;
    unsigned int tid_of_breaking;
}k16_data;



k16_data reduce_kernel16_data(k16_data i, k16_data j){
    k16_data ret_val;
    ret_val._k3 = 0;
    if(i.tid == i.tid_of_breaking){
        ret_val._k2 = i._k2;
        ret_val.tid = i.tid;
    }else if(j.tid == j.tid_of_breaking){
        ret_val._k2 = j._k2;
        ret_val.tid = j.tid; 
    }else{
        if(i.tid < j.tid){
            ret_val.tid = i.tid;
        }else{
            ret_val.tid=j.tid;
        }
    }
    if(i.tid <= i.tid_of_breaking){
        ret_val._k3 += i._k3;
    }
    if(j.tid <= j.tid_of_breaking){
        ret_val._k3 += j._k3;
    }
    return ret_val;
}



double dmax(double v, double c){
    if(v > c){
        return v;
    }
    return c;
}

Boolean isDenormal(double v){
    return (isnormal(v) || isnan(v)) == FALSE;
}

double relative_error(double true_val,double estimation){
    if(true_val == 0.0 || estimation == 0.0){
        return 100.0;
    }
    if(isDenormal(true_val) && isDenormal(estimation)){
        #pragma omp critical
        printf("Used the denormilization route\n");
        return 0.0;
    }
    if(isinf(true_val) && isinf(estimation)){
        printf("Used the infinte route\n");
        return 0.0;
    }
    double rel_err = fabs((estimation - true_val)/true_val)*100.0;
    return rel_err;
}

Boolean withinErrorThreshold(double true_val, double estimation, double thresh){
    if(relative_error(true_val,estimation) > thresh){
        return FALSE;
    }
    return TRUE;
}

void taskAtomicAddToVectorMarkDone(double constant, double * vec, int lower, int upper){
    printf("Adding %f to x from %d to %d\n", constant, lower, upper);
    fflush(stdout);
    sleep(1);
    for(int i = lower; i < upper; i ++){
        #pragma omp atomic
        vec[i] += constant;
    }
}






// _______________________ end of my functions _______________________________________
   struct Arrays
   {
     double U[1001];
     double V[1001];
     double W[1001];
     double X[1001];
     double Y[1001];
     double Z[1001];
     double G[1001];
     double Du1[101];
     double Du2[101];
     double Du3[101];
     double Grd[1001];
     double Dex[1001];
     double Xi[1001];
     double Ex[1001];
     double Ex1[1001];
     double Dex1[1001];
     double Vx[1001];
     double Xx[1001];
     double Rx[1001];
     double Rh[2048];
     double Vsp[101];
     double Vstp[101];
     double Vxne[101];
     double Vxnd[101];
     double Ve3[101];
     double Vlr[101];
     double Vlin[101];
     double B5[101];
     double Plan[300];
     double D[300];
     double Sa[101];
     double Sb[101];     
     double P[512][4];
     double Px[101][25];
     double Cx[101][25];
     double Vy[25][101];
     double Vh[7][101];
     double Vf[7][101];
     double Vg[7][101];
     double Vs[7][101];
     double Za[7][101];
     double Zp[7][101];
     double Zq[7][101];
     double Zr[7][101];
     double Zm[7][101];
     double Zb[7][101];
     double Zu[7][101];
     double Zv[7][101];
     double Zz[7][101];               
     double B[64][64];
     double C[64][64];
     double H[64][64];     
     double U1[2][101][5];
     double U2[2][101][5];
     double U3[2][101][5];
     double Xtra[40];     
     long   E[96];
     long   F[96];
     long   Ix[1001];
     long   Ir[1001];
     long   Zone[301];
     double X0[1001];
     double W0[1001];
     double Px0[101][25];
     double P0[512][4];
     double H0[64][64];
     double Rh0[2048];
     double Vxne0[101];
     double Zr0[7][101];
     double Zu0[7][101];
     double Zv0[7][101];
     double Zz0[7][101];
     double Za0[101][101];  // was 101 25
     double Stb50;               
     double Xx0;

  }as1;

#define  refresh_vars \
\
u        = as1.U;           /*U     [1001]*/  \
v        = as1.V;           /*V     [1001]*/  \
w        = as1.W;           /*W     [1001]*/  \
x        = as1.X;           /*X     [1001]*/  \
y        = as1.Y;           /*Y     [1001]*/  \
z        = as1.Z;           /*Z     [1001]*/  \
g        = as1.G;           /*G     [1001]*/  \
du1      = as1.Du1;         /*Du1[101]*/    \
du2      = as1.Du2;         /*Du2[101]*/    \
du3      = as1.Du3;         /*Du3[101]*/    \
grd      = as1.Grd;         /*Grd[1001]*/    \
dex      = as1.Dex;         /*Dex[1001]*/    \
xi       = as1.Xi;          /*Xi[1001]*/ \
ex       = as1.Ex;          /*Ex[1001]*/ \
ex1      = as1.Ex1;         /*Ex1[1001]*/    \
dex1     = as1.Dex1;        /*Dex1[1001]*/   \
vx       = as1.Vx;          /*Vx[1001]*/ \
xx       = as1.Xx;          /*Xx[1001]*/ \
rx       = as1.Rx;          /*Rx[1001]*/ \
rh       = as1.Rh;          /*Rh[2048]*/ \
vsp      = as1.Vsp;         /*Vsp[101]*/    \
vstp     = as1.Vstp;        /*Vstp[101]*/   \
vxne     = as1.Vxne;        /*Vxne[101]*/   \
vxnd     = as1.Vxnd;        /*Vxnd[101]*/   \
ve3      = as1.Ve3;         /*Ve3[101]*/    \
vlr      = as1.Vlr;         /*Vlr[101]*/    \
vlin     = as1.Vlin;        /*Vlin[101]*/   \
b5       = as1.B5;          /*B5[101]*/ \
plan     = as1.Plan;        /*Plan[300]*/   \
d        = as1.D;           /*D[300]*/  \
sa       = as1.Sa;          /*Sa[101]*/ \
sb       = as1.Sb;          /*Sb[101]*/      \
p       = as1.P;            /*P [512][4];*/  \
px       = as1.Px;          /*Px[101][25];*/ \
cx       = as1.Cx;          /*Cx[101][25];*/ \
vy       = as1.Vy;         /*Vy[25][101];*/ \
vh       = as1.Vh;         /*Vh[7][101];*/ \
vf      = as1.Vf;         /*Vf[7][101];*/ \
vg       = as1.Vg;         /*Vg[7][101];*/ \
vs       = as1.Vs;         /*Vs[7][101];*/ \
za       = as1.Za;         /*Za[7][101];*/ \
zb       = as1.Zb;         /*Zp[7][101];*/ \
zp       = as1.Zp;         /*Zq[7][101];*/ \
zq       = as1.Zq;         /*Zr[7][101];*/ \
zr       = as1.Zr;         /*Zm[7][101];*/ \
zm       = as1.Zm;         /*Zb[7][101];*/ \
zz       = as1.Zz;         /*Zu[7][101];*/ \
zu       = as1.Zu;         /*Zv[7][101];*/ \
zv       = as1.Zv;         /*Zz[7][101];*/   \
b        = as1.B;           /*B [64][64];*/  \
c        = as1.C;           /*C [64][64];*/  \
h        = as1.H;           /*H [64][64];*/       \
u1     = as1.U1;      /*U1[2][101][5];*/ \
u2    = as1.U2;      /*U2[2][101][5];*/ \
u3     = as1.U3;      /*U3[2][101][5];*/ \
xtra     = as1.Xtra;              /*Xtra[40];     */ \
a11      = as1.Xtra[1];      \
a12      = as1.Xtra[2];      \
a13      = as1.Xtra[3];      \
a21      = as1.Xtra[4];      \
a22      = as1.Xtra[5];      \
a23      = as1.Xtra[6];      \
a31      = as1.Xtra[7];      \
a32      = as1.Xtra[8];      \
a33      = as1.Xtra[9];      \
c0       = as1.Xtra[12];     \
dk       = as1.Xtra[15];     \
dm22     = as1.Xtra[16];     \
dm23     = as1.Xtra[17];     \
dm24     = as1.Xtra[18];     \
dm25     = as1.Xtra[19];     \
dm26     = as1.Xtra[20];     \
dm27     = as1.Xtra[21];     \
dm28     = as1.Xtra[22];     \
expmax   = as1.Xtra[26];     \
flx      = as1.Xtra[27];\
q        = as1.Xtra[28];\
r        = as1.Xtra[30];\
s        = as1.Xtra[32];\
sig      = as1.Xtra[34];\
stb5     = as1.Xtra[35];\
t        = as1.Xtra[36];\
xnm      = as1.Xtra[39];\
e         = as1.E;       \
f        = as1.F;       \
ix       = as1.Ix;       \
ir       = as1.Ir;      \
zone     = as1.Zone;    \
x0       = as1.X0;      \
w0       = as1.W0;      \
px0      = as1.Px0;      /*Px0[101][25]; */\
p0       = as1.P0;       /*P0[512][4];   */\
h0       = as1.H0;        /*H0[64][64];   */ \
rh0      = as1.Rh0;             /*Rh0[2048];    */\
vxne0    = as1.Vxne0;           /*Vxne0[101];   */ \
zr0      = as1.Zr0;      /*Zr0[7][101];  */  \
zu0      = as1.Zu0;      /*Zu0[7][101];  */  \
zv0      = as1.Zv0;      /*Zv0[7][101];  */  \
zz0      = as1.Zz0;      /*Zz0[7][101];  */  \
za0      = as1.Za0;      /*Za0[101][101];*/  /* was 101 25*/    \
stb50    = as1.Stb50;            /*Stb50;*/                   \
xx0      = as1.Xx0;              /*Xx*/\


#define init_kernel_vars \
double  *u        = as1.U;           /*U    [1001]*/  \
double  *v        = as1.V;           /*V    [1001]*/  \
double  *w        = as1.W;           /*W    [1001]*/  \
double  *x        = as1.X;           /*X    [1001]*/  \
double  *y        = as1.Y;           /*Y    [1001]*/  \
double  *z        = as1.Z;           /*Z    [1001]*/  \
double  *g        = as1.G;           /*G    [1001]*/  \
double  *du1      = as1.Du1;         /*Du1  [101]*/    \
double  *du2      = as1.Du2;         /*Du2  [101]*/    \
double  *du3      = as1.Du3;         /*Du3  [101]*/    \
double  *grd      = as1.Grd;         /*Grd  [1001]*/    \
double  *dex      = as1.Dex;         /*Dex  [1001]*/    \
double  *xi       = as1.Xi;          /*Xi   [1001]*/ \
double  *ex       = as1.Ex;          /*Ex   [1001]*/ \
double  *ex1      = as1.Ex1;         /*Ex1  [1001]*/    \
double  *dex1     = as1.Dex1;        /*Dex1 [1001]*/   \
double  *vx       = as1.Vx;          /*Vx   [1001]*/ \
double  *xx       = as1.Xx;          /*Xx   [1001]*/ \
double  *rx       = as1.Rx;          /*Rx   [1001]*/ \
double  *rh       = as1.Rh;          /*Rh   [2048]*/ \
double  *vsp      = as1.Vsp;         /*Vsp  [101]*/    \
double  *vstp     = as1.Vstp;        /*Vstp [101]*/   \
double  *vxne     = as1.Vxne;        /*Vxne [101]*/   \
double  *vxnd     = as1.Vxnd;        /*Vxnd [101]*/   \
double  *ve3      = as1.Ve3;         /*Ve3  [101]*/    \
double  *vlr      = as1.Vlr;         /*Vlr  [101]*/    \
double  *vlin     = as1.Vlin;        /*Vlin [101]*/   \
double  *b5       = as1.B5;          /*B5   [101]*/ \
double  *plan     = as1.Plan;        /*Plan [300]*/   \
double  *d        = as1.D;           /*D    [300]*/  \
double  *sa       = as1.Sa;          /*Sa   [101]*/ \
double  *sb       = as1.Sb;          /*Sb   [101]*/      \
double  (*p)[4]        = as1.P;            /*P [512][4];*/  \
double  (*px)[25]       = as1.Px;          /*Px[101][25];*/ \
double  (*cx)[25]       = as1.Cx;          /*Cx[101][25];*/ \
double  (*vy)[101]       = as1.Vy;         /*Vy[25][101];*/ \
double  (*vh)[101]       = as1.Vh;         /*Vh[7][101];*/ \
double  (*vf)[101]       = as1.Vf;         /*Vf[7][101];*/ \
double  (*vg)[101]       = as1.Vg;         /*Vg[7][101];*/ \
double  (*vs)[101]       = as1.Vs;         /*Vs[7][101];*/ \
double  (*za)[101]       = as1.Za;         /*Za[7][101];*/ \
double  (*zb)[101]       = as1.Zb;         /*Zp[7][101];*/ \
double  (*zp)[101]       = as1.Zp;         /*Zq[7][101];*/ \
double  (*zq)[101]       = as1.Zq;         /*Zr[7][101];*/ \
double  (*zr)[101]       = as1.Zr;         /*Zm[7][101];*/ \
double  (*zm)[101]       = as1.Zm;         /*Zb[7][101];*/ \
double  (*zz)[101]       = as1.Zz;         /*Zu[7][101];*/ \
double  (*zu)[101]       = as1.Zu;         /*Zv[7][101];*/ \
double  (*zv)[101]       = as1.Zv;         /*Zz[7][101];*/   \
double  (*b)[64]        = as1.B;           /*B [64][64];*/  \
double  (*c)[64]        = as1.C;           /*C [64][64];*/  \
double  (*h)[64]        = as1.H;           /*H [64][64];*/       \
double  (*u1)[101][5]       = as1.U1;      /*U1[2][101][5];*/ \
double  (*u2)[101][5]       = as1.U2;      /*U2[2][101][5];*/ \
double  (*u3)[101][5]       = as1.U3;      /*U3[2][101][5];*/ \
double  *xtra     = as1.Xtra;              /*Xtra[40];     */ \
double  a11      = as1.Xtra[1];      \
double  a12      = as1.Xtra[2];      \
double  a13      = as1.Xtra[3];      \
double  a21      = as1.Xtra[4];      \
double  a22      = as1.Xtra[5];      \
double  a23      = as1.Xtra[6];      \
double  a31      = as1.Xtra[7];      \
double  a32      = as1.Xtra[8];      \
double  a33      = as1.Xtra[9];      \
double  c0       = as1.Xtra[12];     \
double  dk       = as1.Xtra[15];     \
double  dm22     = as1.Xtra[16];     \
double  dm23     = as1.Xtra[17];     \
double  dm24     = as1.Xtra[18];     \
double  dm25     = as1.Xtra[19];     \
double  dm26     = as1.Xtra[20];     \
double  dm27     = as1.Xtra[21];     \
double  dm28     = as1.Xtra[22];     \
double  expmax   = as1.Xtra[26];     \
double  flx      = as1.Xtra[27];\
double  q        = as1.Xtra[28];\
double  r        = as1.Xtra[30];\
double  s        = as1.Xtra[32];\
double  sig      = as1.Xtra[34];\
double  stb5     = as1.Xtra[35];\
double  t        = as1.Xtra[36];\
double  xnm      = as1.Xtra[39];\
long    *e         = as1.E;       \
long    *f        = as1.F;       \
long    *ix       = as1.Ix;       \
long    *ir       = as1.Ir;      \
long    *zone     = as1.Zone;    \
double  *x0       = as1.X0;      \
double  *w0       = as1.W0;      \
double  (*px0)[25]      = as1.Px0;      /*Px0[101][25]; */\
double  (*p0)[4]       = as1.P0;       /*P0[512][4];   */\
double  (*h0)[64]       = as1.H0;        /*H0[64][64];   */ \
double  *rh0      = as1.Rh0;             /*Rh0[2048];    */\
double  *vxne0    = as1.Vxne0;           /*Vxne0[101];   */ \
double  (*zr0)[101]      = as1.Zr0;      /*Zr0[7][101];  */  \
double  (*zu0)[101]      = as1.Zu0;      /*Zu0[7][101];  */  \
double  (*zv0)[101]      = as1.Zv0;      /*Zv0[7][101];  */  \
double  (*zz0)[101]      = as1.Zz0;      /*Zz0[7][101];  */  \
double  (*za0)[101]      = as1.Za0;      /*Za0[101][101];*/  /* was 101 25*/    \
double  stb50    = as1.Stb50;            /*Stb50;*/                   \
double  xx0      = as1.Xx0;              /*Xx0; */ \



#define  set_scalar_vars  \
as1.Xtra[1]  = a11;\
as1.Xtra[2]  = a12;\
as1.Xtra[3]  = a13;\
as1.Xtra[4]  = a21;\
as1.Xtra[5]  = a22;\
as1.Xtra[6]  = a23;\
as1.Xtra[7]  = a31;\
as1.Xtra[8]  = a32;\
as1.Xtra[9]  = a33;\
as1.Xtra[12] = c0;\
as1.Xtra[15] = dk;\
as1.Xtra[16] = dm22;\
as1.Xtra[17] = dm23;\
as1.Xtra[18] = dm24;\
as1.Xtra[19] = dm25;\
as1.Xtra[20] = dm26;\
as1.Xtra[21] = dm27;\
as1.Xtra[22] = dm28;\
as1.Xtra[26] = expmax;\
as1.Xtra[27] = flx;\
as1.Xtra[28] = q;\
as1.Xtra[30] = r;\
as1.Xtra[32] = s;\
as1.Xtra[34] = sig;\
as1.Xtra[35] = stb5;\
as1.Xtra[36] = t;\
as1.Xtra[39] = xnm;\
as1.Stb50 = stb50;\
as1.Xx0= xx0; \








   struct Parameters
   {
       long   Inner_loops;
       long   Outer_loops;
       long   Loop_mult;
       double Flops_per_loop;
       double Sumcheck[3][25];
       long   Accuracy[3][25];
       double LoopTime[3][25];
       double LoopSpeed[3][25];
       double LoopFlos[3][25];
       long   Xflops[25];
       long   Xloops[3][25];
       long   Nspan[3][25];       
       double TimeStart;
       double TimeEnd;
       double Loopohead;
       long   Count;
       long   Count2;
       long   Pass;
       long   Extra_loops[3][25];
       long   K2;
       long   K3;
       long   M16;
       long   J5;
       long   Section;
       long   N16;
       double Mastersum;
       long   M24;
  
   }as2;
   
   #define n            as2.Inner_loops
   #define loop         as2.Outer_loops
   #define mult         as2.Loop_mult
   #define nflops       as2.Flops_per_loop
   #define Checksum     as2.Sumcheck
   #define accuracy     as2.Accuracy
   #define RunTime      as2.LoopTime
   #define Mflops       as2.LoopSpeed
   #define FPops        as2.LoopFlos
   #define nspan        as2.Nspan
   #define xflops       as2.Xflops
   #define xloops       as2.Xloops
   #define StartTime    as2.TimeStart
   #define EndTime      as2.TimeEnd
   #define overhead_l   as2.Loopohead
   #define count        as2.Count
   #define count2       as2.Count2
   #define pass         as2.Pass
   #define extra_loops  as2.Extra_loops
   #define k2           as2.K2
   #define k3           as2.K3
   #define m16          as2.M16
   #define j5           as2.J5
   #define _section      as2.Section
   #define n16          as2.N16
   #define MasterSum    as2.Mastersum
   #define m24          as2.M24

 // VERSION

 #ifdef CNNT
    #define options   "Non-optimised"
    #define opt "0"
 #else
//    #define options   "Optimised"
//    #define options   "Opt 3 32 Bit"
    #define options  "vfpv4 32 Bit"
     #define opt "3"
 #endif



double      runSecs = 0.1;
Boolean     reliability = TRUE;
Boolean     runRel;
Boolean     nsRes = FALSE;
double      sumscomp[3][25];
int         compareFail = 0;
int         failCount;
FILE        *outfile;

   void init(long which);



   
        /* Initialises arrays and variables  */
             
   long endloop(long which);
   
        /* Controls outer loops and stores results */

   long parameters(long which);
   
        /* Gets loop parameters and variables, starts timer */
        
   void kernels();
   
        /* The 24 kernels */
        
   void check(long which);
   
        /* Calculates checksum accuracy */
             
   void iqranf();
   
        /* Random number generator for Kernel 14 */

   void checkOut(int which);

              
int main(int argc, char *argv[])
{
    double pass_time, least, lmult, now = 1.0, wt;
    long   i, k, loop_passes;
    long   mul[3] = {1, 2, 8};
    double weight[3] = {1.0, 2.0, 1.0};
    long   Endit, which;
    double maximum[4];
    double minimum[4];
    double average[4];
    double harmonic[4];
    double geometric[4];
    long   xspan[4];
    char   general[9][80] = {" "};
    int    param;
    int    gg;
    int    nopause = 1;
    


     if (argc > 1)
     {
       switch (argv[1][0])
        {
             case 'N':
                nopause = 0;
                break;
             case 'n':
                nopause = 0;
                break;
        }
    }
    if (argc > 2)
    {
       sscanf(argv[2], "%d", &param);
       if (param > 0)
       {
           runSecs = param;
           reliability = TRUE;
           if (runSecs > 60) runSecs = 60; 
       }
    }
    

    printf ("L.L.N.L. 'C' KERNELS: MFLOPS   P.C.  VERSION 4.0\n\n");

     
    printf("Optimisation  %s\n\n",options);


/************************************************************************
 *                  Open results file LLloops.txt                       *
 ************************************************************************/
    outfile = fopen("LLloops.txt","w");
    if (outfile == NULL)
    {
        printf (" Cannot open results file LLloops.txt\n\n");
        printf(" Press Enter\n\n");
        gg = getchar();
        exit (0);
    }

    getDetails();
            
    local_time();

    fprintf (outfile, " #####################################################\n\n");                     
    fprintf (outfile, " Livermore Loops Benchmark %s via C/C++ %s\n", options, timeday);

    if (reliability)
    {
        fprintf (outfile, " Reliability test %3.0f seconds each loop x 24 x 3\n\n", runSecs);
    }
    fflush(outfile);    
                   
/************************************************************************
 *       Calculate overhead of executing endloop procedure              *
 ************************************************************************/
       
    printf ("Calculating outer loop overhead\n");
    pass = -20;
    extra_loops[0][0] = 1;
    loop = 1000;
    which = 0;
    _section = 0;
    runRel = FALSE;
    do
    {
        start_time();
        count = 0;
        count2 = 0;
        pass = pass + 1;        
        do
        {
            endloop (0);
        }
        while (count < loop);
        end_time();
        overhead_l = secs;
        printf ("%10ld times %6.2f seconds\n", loop, overhead_l);
        if (overhead_l > 0.2)
        {
            pass = 0;
        }
        if (pass < 0)
        {
            if (overhead_l < (double)runSecs / 50)
            {
                loop = loop * 10;
            }
            else
            {
                loop = loop * 2;
            }
        }
    }
    while (pass < 0);
        
    overhead_l = overhead_l / (double)(loop);
    printf ("Overhead for each loop %12.4e seconds\n\n", overhead_l);
                    
    printf("##########################################\n"); 
    printf ("\nFrom File /proc/cpuinfo\n");
    printf("%s\n", configdata[0]);
    printf ("\nFrom File /proc/version\n");
    printf("%s\n", configdata[1]);

/************************************************************************
 *      Execute the kernels three times at different Do Spans           *
 ************************************************************************/
    for ( _section=0 ; _section<3 ; _section++ )
    {
        loop_passes = 200 * mul[_section];
        pass = -20;
        mult = 2 * mul[_section];
        runRel = FALSE;
    
        for ( i=1; i<25; i++)
        {
            extra_loops[_section][i] = 1;
        }
        if (reliability)
        {
             local_time();
             fprintf (outfile, " Part %ld of 3 start at %s\n", _section + 1, timeday);
             fflush(outfile);
        }

/************************************************************************
 *   Calculate extra loops for running time of runSecs seconds per kernel     *
 ************************************************************************/

             printf ("Calibrating part %ld of 3\n\n", _section + 1);

        do
        
        /* Run a number of times with increased number of loops
         or until the time for each loop is at least 0.001 seconds */   

        {
            pass = pass + 1;
            mult = mult * 2;
        
            count2 = 0;            
            for ( i=1; i<25; i++)
            {
                 RunTime[_section][i] = 0.0;
            }
            start_time();

            kernels();

            end_time();
            pass_time = secs;
            least = 1.0;
            for ( i=1; i<25; i++)
            {
                if (RunTime[_section][i] < 0.001)
                {
                    least = 0.0;
                    RunTime[_section][i] = 0.0008;
                    extra_loops[_section][i] = extra_loops[_section][i] * 2;
                }
            }
            printf ("Loop count %10ld %5.2f seconds\n", mult, pass_time);
        
            if (least > 0.0 )
            {
                pass = 0;
            }
            else
            {
                if (pass_time < (double)runSecs / 5)
                {
                    mult = mult * 2;
                }
            }
        }
        while (pass < 0);

        lmult = (double)(mult) / (double)(loop_passes);
    
        for ( i=1; i<25; i++)
        {
            
          /* Calculate extra loops to produce a run time of about runSecs seconds
           for each kernel. For each of the extra loops the parameters
           are re-initialised. The time for initialising parameters is
           not included in the loop time. */
                             
            extra_loops[_section][i] = (long)(runSecs / RunTime[_section][i]
                                * (double)extra_loops[_section][i] * lmult) +1;
            RunTime[_section][i] = 0.0;
        }

        mult = loop_passes;
        
        printf ("\nLoops  200 x %2ld x Passes\n\n", mul[_section]);
        printf ("Kernel       Floating Pt ops\n");
        printf ("No  Passes E No    Total      Secs.  MFLOPS  Span     "
                                        "Checksums[EX]          ChecksumsActual       %%Error   OK\n");
        printf ("------------ -- ------------- ----- -------  ---- "
                                         "------------------- ------------------- --------- --\n");

        pass = 1;
        count2 = 0;
        if (reliability) runRel = TRUE;

/************************************************************************
 *                      Execute the kernels                             *
 ************************************************************************/
        
        kernels();

        maximum[_section] = 0.0;
        minimum[_section] = Mflops[_section][1];
        average[_section] = 0.0;
        harmonic[_section] = 0.0;
        geometric[_section] = 0.0;
        xspan[_section] = 0;

/************************************************************************
 *                        Calculate averages etc.                       *
 ************************************************************************/
        
            for ( k=1 ; k<=24 ; k++ )
        {
           average[_section] = average[_section] + Mflops[_section][k];
           harmonic[_section] = harmonic[_section] + 1 / Mflops[_section][k];
           geometric[_section] = geometric[_section] + log(Mflops[_section][k]);
           xspan[_section] = xspan[_section] + nspan[_section][k];
           if (Mflops[_section][k] < minimum[_section])
           {
               minimum[_section] = Mflops[_section][k];
           }
           if (Mflops[_section][k] > maximum[_section])
           {
               maximum[_section] = Mflops[_section][k];
           }
        }
        average[_section] = average[_section] / 24.0;
        harmonic[_section] = 24.0 / harmonic[_section];
        geometric[_section] = exp(geometric[_section] / 24.0);
        xspan[_section] = xspan[_section] / 24;

        if (pass > 0)

/************************************************************************
 *        Display averages etc. except during calibration               *
 ************************************************************************/
        
        {
           printf ("\n");
           printf ("                     Maximum   Rate%8.2f \n",
                                                  maximum[_section]);
           printf ("                     Average   Rate%8.2f \n",
                                                  average[_section]);
           printf ("                     Geometric Mean%8.2f \n",
                                                  geometric[_section]);
           printf ("                     Harmonic  Mean%8.2f \n",
                                                  harmonic[_section]);
           printf ("                     Minimum   Rate%8.2f \n\n",
                                                  minimum[_section]);
           printf ("                     Do Span   %4ld\n\n",
                                                  xspan[_section]);
        }        
    }

/************************************************************************
 *    End of executing the kernels three times at different Do Spans    *
 ************************************************************************/
    
    maximum[3] = 0.0;
    minimum[3] = Mflops[0][1];
    average[3] = 0.0;
    harmonic[3] = 0.0;
    geometric[3] = 0.0;
    xspan[3] = 0;
    wt = 0.0;
    
/************************************************************************
 *     Calculate weighted averages for all Do Spans and display         *
 ************************************************************************/
    
    for ( _section=0 ; _section<3 ; _section++ )
    {
        for ( k=1 ; k<=24 ; k++ )
        {
           average[3] = average[3] + weight[_section]
                                     * Mflops[_section][k];
           harmonic[3] = harmonic[3] + weight[_section]
                                     / Mflops[_section][k];
           geometric[3] = geometric[3] + weight[_section]
                                     * log(Mflops[_section][k]);
           xspan[3] = xspan[3] + (long)weight[_section]
                                     * nspan[_section][k]; 
           if (Mflops[_section][k] < minimum[3])
           {
               minimum[3] = Mflops[_section][k];
           }
           if (Mflops[_section][k] > maximum[3])
           {
               maximum[3] = Mflops[_section][k];
           }
        }
        wt = wt + weight[_section];
    }
    average[3] = average[3] / (24.0 * wt);
    harmonic[3] = 24.0 * wt / harmonic[3];
    geometric[3] = exp(geometric[3] / (24.0 * wt));
    xspan[3] = (long)((double)xspan[3] / (24.0 * wt));

    printf ("                Overall\n\n");
    printf ("                Part 1 weight 1\n");
    printf ("                Part 2 weight 2\n");
    printf ("                Part 3 weight 1\n\n");
    printf ("                     Maximum   Rate%8.2f \n", maximum[3]);
    printf ("                     Average   Rate%8.2f \n", average[3]);
    printf ("                     Geometric Mean%8.2f \n", geometric[3]);
    printf ("                     Harmonic  Mean%8.2f \n", harmonic[3]);
    printf ("                     Minimum   Rate%8.2f \n\n", minimum[3]);
    printf ("                     Do Span   %4ld\n\n", xspan[3]);
            
    if (reliability)
    { 
        if (!compareFail)
        {
            if (nsRes)
            {
               fprintf(outfile, "\n Numeric results were consistent with first\n\n");
            }
            else
            {
               fprintf(outfile, " Numeric results were as expected\n\n");
            }
        }
        else
        {
            printf("\n ERRORS have occurred - see log file\n");         
            fprintf (outfile, "\n");
        }
    }
/************************************************************************
 *              Add results to output file LLloops.txt                  *
 ************************************************************************/

    fprintf (outfile, " MFLOPS for 24 loops\n");
    for ( which=1; which<13 ; which++ )
    {
       if (Mflops[0][which] < 10000)
       {
          fprintf (outfile, "%7.1f", Mflops[0][which]);
       }
       else
       {
          fprintf (outfile, "%7.0f", Mflops[0][which]);
       }
    }
    fprintf (outfile, "\n");
    for ( which=13; which<25 ; which++ )
    {
       if (Mflops[0][which] < 10000)
       {
          fprintf (outfile, "%7.1f", Mflops[0][which]);
       }
       else
       {
          fprintf (outfile, "%7.0f", Mflops[0][which]);
       }
    }
    fprintf (outfile, "\n\n");
    
    fprintf (outfile, " Overall Ratings\n");
    fprintf (outfile, " Maximum Average Geomean Harmean Minimum\n");
    fprintf (outfile, "%8.1f%8.1f%8.1f%8.1f%8.1f\n\n",
               maximum[3], average[3], geometric[3], harmonic[3], minimum[3]);

    if (!reliability)
    {
        checkOut((int)which);
    }
    
    fprintf (outfile, " ########################################################\n\n");  
    fprintf(outfile, "\n");
    fprintf (outfile, "SYSTEM INFORMATION\n\nFrom File /proc/cpuinfo\n");
    fprintf (outfile, "%s \n", configdata[0]);
    fprintf (outfile, "\nFrom File /proc/version\n");
    fprintf (outfile, "%s \n", configdata[1]);
    fprintf (outfile, "\n");
    fflush(outfile);
    
    printf ("\n");

    if (nopause)
    {
         char moredata[1024];
         printf("Type additional information to include in linpack.txt - Press Enter\n");
         if (fgets (moredata, sizeof(moredata), stdin) != NULL)
         fprintf (outfile, "Additional information - %s\n", moredata);
    }
    fclose (outfile);
    return 0;
}
    
/************************************************************************
 *                          The Kernels                                 *
 ************************************************************************/

void kernels()
 {
   long   lw;
   long   ipnt, ipntp, ii;
   double temp;
   long   nl1, nl2;
   long   kx, ky;
   double ar, br, cr;
   long   i, j, k, m;
   long   ip, i1, i2, j1, j2, j4, lb;
   long   ng, nz;
   double tmp;
   double scale, xnei, xnc, e3,e6;
   long   ink, jn, kn, kb5i;
   double di, dn; 
   double qa;

  init_kernel_vars
  
   for ( k=0 ; k<25; k++)
    {
        Checksum[_section][k] = 0.0;
    }
   
    /*
     *******************************************************************
     *   Kernel 1 -- hydro fragment
     *******************************************************************
     */



    parameters (1);
    long _n;

    do
    {  
        _n = n;
        refresh_vars
        //#pragma omp parallel for private(k) firstprivate(k1_n,r,t,q,x, y, z)
        for ( k=0 ; k<_n ; k++ )
        {
            x[k] = q + y[k]*( r*z[k+10] + t*z[k+11] );
        }
        
        
        endloop (1);
     }
     while (count < loop);
    /*
     *******************************************************************
     *   Kernel 2 -- ICCG excerpt (Incomplete Cholesky Conjugate Gradient)
     *******************************************************************
    */

    parameters (2);
     refresh_vars

    do
     {
        ii = n;
        ipntp = 0;
        do
         {
            ipnt = ipntp;
            ipntp += ii;
            ii /= 2;
            i = ipntp;
            #pragma omp parallel for private(k) firstprivate(ipnt, ipntp,v,x)
            for ( k=ipnt+1 ; k<ipntp ; k=k+2 )
             {                 
                i = ipntp +1 + (k - ipnt - 1)/2;
                x[i] = x[k] - v[k]*x[k-1] - v[k+1]*x[k+1];
             }
         } while ( ii>0 );
        

         set_scalar_vars
         endloop (2);
         refresh_vars
     }
     while (count < loop);
  
    /*
     *******************************************************************
     *   Kernel 3 -- inner product
     *******************************************************************
      */

    
    parameters (3);
    refresh_vars

    do
     {
        _n = n;
        q = 0.0;
        #pragma omp parallel for firstprivate(z,x, _n) private(k) reduction(+:q) 
        for ( k=0 ; k<_n; k++ )
        {
            q += z[k]*x[k];
        }
        as1.Xtra[28] = q;
        endloop (3);
        refresh_vars
     }
     while (count < loop);


    /*
     *******************************************************************
     *   Kernel 4 -- banded linear equations

     *******************************************************************
      */
    
    parameters (4);
    refresh_vars

    m = ( 1001-7 )/2;
    do
     {
         _n = n;
        #pragma omp parallel for private(lw, temp, k) shared(x,y)
        for ( k=6 ; k<1001 ; k=k+m )
         {
            lw = k - 6;
            temp = x[k-1];

            for ( j=4 ; j<_n ; j=j+5 )
             {
                temp -= x[lw]*y[j];
                lw++;
             }
            x[k-1] = y[4]*temp;
         }
         

         set_scalar_vars
         endloop (4);
         refresh_vars
     }
     while (count < loop);
 
    /*
     *******************************************************************
     *   Kernel 5 -- tri-diagonal elimination, below diagonal
     *******************************************************************
     */

    parameters (5);
    refresh_vars
    int num_threads = 1;
    //Since the array x may or may not retain values I zero everyting except the first element just to be sure
    
    for(i=1; i<_n;i++){
        x[i]= 0;
    }

    #pragma omp parallel shared(num_threads)
    {
        #pragma single
        num_threads = omp_get_num_threads();
    }

    int num_threads_async_mult = (int)fmax(floor(num_threads/2),1);
    int num_threads_dependent_work = (int)fmax(num_threads - num_threads_async_mult, 1);
    double coeff[1001]={0.0};
    double constant[1001]={0.0};
    do
     {
         _n = n;
        #pragma omp parallel
        {
            int num_threads = omp_get_num_threads();
            int tid = omp_get_thread_num();
            int even_jobs = (int)floor(_n /num_threads);
            int extra_jobs = _n %num_threads;
            if(tid == 0){
                even_jobs += extra_jobs;
            }
            int start_idx = even_jobs*tid + extra_jobs*(tid != 0);
            int end_idx = start_idx + even_jobs;

            //Stage one
            if(tid == 0){
                for(int loc = 1; loc < end_idx; loc++){
                    x[loc] =  z[loc]*( y[loc] - x[loc-1] );
                }
            }else{
                coeff[start_idx] = -z[start_idx];
                constant[start_idx] = constant[start_idx];
                for(int loc = start_idx+1; loc < end_idx; loc++){
                    coeff[loc] = -z[loc] *coeff[loc-1];
                    constant[loc] = -z[loc]*(constant[loc-1] - y[loc]); 
                }
            }
            #pragma omp barrier
            #pragma omp single
            {
                int last_xi_first_block_idx = even_jobs+extra_jobs-1;
                for(int last_x_block_i = last_xi_first_block_idx+even_jobs; last_x_block_i < _n-1; last_x_block_i+=even_jobs){
                    x[last_x_block_i] = coeff[last_x_block_i]*x[last_x_block_i - even_jobs] + constant[last_x_block_i];
                }
            }
            #pragma omp barrier
            if(tid > 0){
                for(int loc = start_idx; loc < end_idx; loc++ ){
                    x[loc] = z[loc]*(y[loc] - x[loc-1]);
                }
            }
            #pragma omp barrier
        }        
         endloop (5);
         refresh_vars
     }
     while (count < loop);
 
    /*
     *******************************************************************
     *   Kernel 6 -- general linear recurrence equations
     *******************************************************************
     */
          
    parameters (6);
    refresh_vars

    double w_temp = 0;

    do
     {
        _n = n;
        for ( i=1 ; i<_n ; i++ )
        {
           w_temp = 0.01;
           #pragma omp parallel for firstprivate(b,w,i) private(k)reduction(+:w_temp) 
           for ( k=0 ; k<i ; k++ )
            {
               w_temp += b[k][i] * w[(i-k)-1];
            }
            w[i] = w_temp;
        }

        
        // 
        //for ( i=1 ; i<_n ; i++ )
        // {
        //    w_temp = 0.01;
        //    #pragma omp parallel for firstprivate(b,w,i) private(k)reduction(+:w_temp) 
        //    for ( k=0 ; k<i ; k++ )
        //     {
        //        w_temp += b[k][i] * w[(i-k)-1];
        //     }
        //     w[i] = w_temp;
        // }
        //

        set_scalar_vars
        endloop (6);
        refresh_vars
     }
     while (count < loop);
 
    /*
     *******************************************************************
     *   Kernel 7 -- equation of state fragment
     *******************************************************************
     */
     
    parameters (7);
    refresh_vars 
    do
     {
         _n = n;
         #pragma omp parallel for shared(x,u,r,z,y,t,q) firstprivate(_n) private(k)
        for ( k=0 ; k<_n ; k++ )
         {
            x[k] = u[k] + r*( z[k] + r*y[k] ) +
                   t*( u[k+3] + r*( u[k+2] + r*u[k+1] ) +
                      t*( u[k+6] + q*( u[k+5] + q*u[k+4] ) ) );
         }
        

         set_scalar_vars
         endloop (7);
         refresh_vars
     }
     while (count < loop);
 
    /*
     *******************************************************************
     *   Kernel 8 -- ADI integration
     *******************************************************************
     */

    nl1 = 0;
    nl2 = 1;
    
    parameters (8);
    refresh_vars

    do
    {
        _n = n;
        #pragma omp parallel for private(ky)
                   for ( ky=1 ; ky<_n ; ky++ )
           {
        for ( kx=1 ; kx<3 ; kx++ )
        {


              du1[ky] = u1[nl1][ky+1][kx] - u1[nl1][ky-1][kx];
              du2[ky] = u2[nl1][ky+1][kx] - u2[nl1][ky-1][kx];
              du3[ky] = u3[nl1][ky+1][kx] - u3[nl1][ky-1][kx];

              u1[nl2][ky][kx]=
                 u1[nl1][ky][kx]+a11*du1[ky]+a12*du2[ky]+a13*du3[ky] + sig*
                  (u1[nl1][ky][kx+1]-2.0*u1[nl1][ky][kx]+u1[nl1][ky][kx-1]);
              u2[nl2][ky][kx]=
                 u2[nl1][ky][kx]+a21*du1[ky]+a22*du2[ky]+a23*du3[ky] + sig*
                  (u2[nl1][ky][kx+1]-2.0*u2[nl1][ky][kx]+u2[nl1][ky][kx-1]);
              u3[nl2][ky][kx]=
                 u3[nl1][ky][kx]+a31*du1[ky]+a32*du2[ky]+a33*du3[ky] + sig*
                  (u3[nl1][ky][kx+1]-2.0*u3[nl1][ky][kx]+u3[nl1][ky][kx-1]);
           }
        }
        

        set_scalar_vars
        endloop (8);
        refresh_vars
   }
    while (count < loop);
 
    /*
     *******************************************************************
     *   Kernel 9 -- integrate predictors
     *******************************************************************
     */

    parameters (9);
    refresh_vars
    
    do
    {
        _n = n;
        #pragma omp parallel for
        for ( i=0 ; i<_n ; i++ )
        {
            px[i][0] = dm28*px[i][12] + dm27*px[i][11] + dm26*px[i][10] +
                       dm25*px[i][ 9] + dm24*px[i][ 8] + dm23*px[i][ 7] +
                       dm22*px[i][ 6] + c0*( px[i][ 4] + px[i][ 5])
                                                       + px[i][ 2];
        }
        

        set_scalar_vars
        endloop (9);
        refresh_vars
   }
    while (count < loop);
    
    /*
     *******************************************************************
     *   Kernel 10 -- difference predictors
     *******************************************************************
     */
     
    parameters (10);
    refresh_vars
    
    do
    {
        _n = n;
        #pragma omp parallel for private(ar, br, cr) shared(cx,px)
        for ( i=0 ; i<_n ; i++ )
        {
            ar        =      cx[i][ 4];
            br        = ar - px[i][ 4];
            px[i][ 4] = ar;
            cr        = br - px[i][ 5];
            px[i][ 5] = br;
            ar        = cr - px[i][ 6];
            px[i][ 6] = cr;
            br        = ar - px[i][ 7];
            px[i][ 7] = ar;
            cr        = br - px[i][ 8];
            px[i][ 8] = br;
            ar        = cr - px[i][ 9];
            px[i][ 9] = cr;
            br        = ar - px[i][10];
            px[i][10] = ar;
            cr        = br - px[i][11];
            px[i][11] = br;
            px[i][13] = cr - px[i][12];
            px[i][12] = cr;
        }
        

        set_scalar_vars
        endloop (10);
        refresh_vars
   }
   while (count < loop);
     
    /*
     *******************************************************************
     *   Kernel 11 -- first sum
     *******************************************************************
     */
     
    parameters (11);
     refresh_vars
    
    do
    {

        _n = n;
        for(int i = 0; i < _n ; i++){
            x[i] = 0.0;
        }
       // #pragma omp parallel  firstprivate(x,y, _n)
        {
            int num_threads = omp_get_num_threads();
            int tid = omp_get_thread_num();
            int extra = _n%num_threads;
            int even = (_n-extra)/num_threads;
            int start_idx = tid*even + (tid != 0)*extra;
            int end_idx =  start_idx + even + (tid == 0)*extra;
            Boolean first_loop =TRUE;

            //printf("[%d] %ld:%ld\n",tid, start_idx, end_idx);
            for (k=start_idx ; k<end_idx ; k++ )
            {
                if(first_loop){
                    first_loop = FALSE;
                    x[k] = y[k];
                    continue;
                }
                x[k] = x[k-1] + y[k];
            }
            #pragma omp barrier
            double partials_to_add = 0;
            for(int thread_id = 0; thread_id < tid; thread_id++){
                int partial_sum_location = (thread_id+1)*even + extra -1;
                partials_to_add += x[partial_sum_location];
            }
            #pragma omp barrier
            if(tid >0){
                for(int _i=start_idx; _i<end_idx; _i++){
                    x[_i] += partials_to_add ;//+ y[_i]; 
                }
            }
        }

        endloop (11);
   }
   while (count < loop);
 
    /*
     *******************************************************************
     *   Kernel 12 -- first difference
     *******************************************************************
     */
     
    parameters (12);
    refresh_vars
    do
    {   
        _n = n;
        #pragma omp parallel for shared(x,y) private(k) firstprivate(_n)
        for ( k=0 ; k<_n ; k++ )
        {
            x[k] = y[k+1] - y[k];
        }


        set_scalar_vars
        endloop (12);
        refresh_vars
   }
   while (count < loop);
 

    /*
     *******************************************************************
     *   Kernel 13 -- 2-D PIC (Particle In Cell)
     *******************************************************************
     */

   parameters (13);
    refresh_vars
   do
    {
        _n = n;
        #pragma omp parallel for shared(p, b,c,y,z,e,f,h) firstprivate(_n) private(ip,i1,j1,i2,j2)
        for ( ip=0; ip<_n; ip++)
        {
            i1 = (long)p[ip][0];
            j1 = (long)p[ip][1];
            i1 &= 64-1;
            j1 &= 64-1;
            p[ip][2] += b[j1][i1];
            p[ip][3] += c[j1][i1];
            p[ip][0] += p[ip][2];
            p[ip][1] += p[ip][3];
            i2 = (long)p[ip][0];
            j2 = (long)p[ip][1];
            i2 = ( i2 & 64-1 ) - 1 ;
            j2 = ( j2 & 64-1 ) - 1 ;
            p[ip][0] += y[i2+32];
            p[ip][1] += z[j2+32];
            i2 += e[i2+32];
            j2 += f[j2+32];
            #pragma omp atomic
            h[j2][i2] += 1.0;
        }

        set_scalar_vars
        endloop (13);
        refresh_vars
   }
   while (count < loop);

    /*
     *******************************************************************
     *   Kernel 14 -- 1-D PIC (Particle In Cell)
     *******************************************************************
     */

    parameters (14);
    refresh_vars
    double val_one, val_two;
    long idx_one, idx_two;
    
    do
    {
        _n = n;
        #pragma omp parallel shared(vx,xx,ix,grd,xi,ex,ex1,dex1, dex, ir,rx) private( idx_one, idx_two, val_one, val_two) firstprivate(_n)
        {
            #pragma  omp for 
            for ( k=0 ; k<_n ; k++ )
            {
                vx[k] = 0.0;
                xx[k] = 0.0;
                ix[k] = (long) grd[k];
                xi[k] = (double) ix[k];
                ex1[k] = ex[ ix[k] - 1 ];
                dex1[k] = dex[ ix[k] - 1 ];
            }
            
            #pragma omp for 
            for ( k=0 ; k<_n ; k++ )
            {
                vx[k] = vx[k] + ex1[k] + ( xx[k] - xi[k] )*dex1[k];
                xx[k] = xx[k] + vx[k]  + flx;
                ir[k] = (long)xx[k];
                rx[k] = xx[k] - (double)ir[k];
                ir[k] = ( ir[k] & 2048-1 ) + 1;
                xx[k] = rx[k] + (double)ir[k];
                #pragma omp atomic
                rh[ ir[k]-1 ] += 1.0 - rx[k];
                #pragma omp atomic
                rh[ ir[k]   ] += rx[k];
                // val_one = 1.0 - rx[k];
                // val_two = rx[k];
                // idx_one = ir[k] -1;
                // idx_two = ir[k]+1;
                // //#pragma omp atomic update
                // rh[idx_one] = rh[idx_one] +  val_one;
                // //#pragma omp atomic update
                // rh[ idx_two ] = rh[ idx_two ] + val_two;
            }
            //#pragma omp for reduction(+:rh[:k14_n-1])

        }

          //#pragma omp parallel for  shared(k14_n, ir, rx) reduction(+:rh[:k14_n])
            //#pragma omp parallel for shared(ir, rx, rh) firstprivate(k14_n) 
            //for ( k=0 ; k<_n ; k++ )
            //{


           // }
        //for ( k=0 ; k<n ; k++ )
        //{
        //    vx[k] = 0.0;
        //    xx[k] = 0.0;
        //    ix[k] = (long) grd[k];
        //    xi[k] = (double) ix[k];
        //    ex1[k] = ex[ ix[k] - 1 ];
        //    dex1[k] = dex[ ix[k] - 1 ];
        //}
        //for ( k=0 ; k<n ; k++ )
        //{
        //    vx[k] = vx[k] + ex1[k] + ( xx[k] - xi[k] )*dex1[k];
        //    xx[k] = xx[k] + vx[k]  + flx;
        //    ir[k] = (long)xx[k];
        //    rx[k] = xx[k] - (double)ir[k];
        //    ir[k] = ( ir[k] & 2048-1 ) + 1;
        //    xx[k] = rx[k] + (double)ir[k];
        //}
        //for ( k=0 ; k<n ; k++ )
        //{
        //    rh[ ir[k]-1 ] += 1.0 - rx[k];
        //    rh[ ir[k]   ] += rx[k];
        //}
        




        set_scalar_vars
        endloop (14);
        refresh_vars
   }
   while (count < loop);

    /*
     *******************************************************************
     *   Kernel 15 -- Casual Fortran.  Development version
     *******************************************************************
    */
    
    parameters (15);
    refresh_vars

    do
    {
        ng = 7;
        nz = n;
        ar = 0.053;
        br = 0.073;
        #pragma omp parallel for
        for ( j=1 ; j<ng ; j++ )
        {
            for ( k=1 ; k<nz ; k++ )
            {
                if ( (j+1) >= ng )
                {
                    vy[j][k] = 0.0;
                    continue;
                }
                if ( vh[j+1][k] > vh[j][k] )
                {
                    t = ar;
                }
                else
                {
                    t = br;
                }
                if ( vf[j][k] < vf[j][k-1] )
                {
                    if ( vh[j][k-1] > vh[j+1][k-1] )
                        r = vh[j][k-1];
                    else
                        r = vh[j+1][k-1];
                    s = vf[j][k-1];
                }
                else
                {
                    if ( vh[j][k] > vh[j+1][k] )
                        r = vh[j][k];
                    else
                        r = vh[j+1][k];
                    s = vf[j][k];
                }
                vy[j][k] = sqrt( vg[j][k]*vg[j][k] + r*r )* t/s;
                if ( (k+1) >= nz )
                {
                    vs[j][k] = 0.0;
                    continue;
                }
                if ( vf[j][k] < vf[j-1][k] )
                {
                    if ( vg[j-1][k] > vg[j-1][k+1] )
                        r = vg[j-1][k];
                    else
                        r = vg[j-1][k+1];
                    s = vf[j-1][k];
                    t = br;
                }
                else
                {
                    if ( vg[j][k] > vg[j][k+1] )
                        r = vg[j][k];
                    else
                        r = vg[j][k+1];
                    s = vf[j][k];
                    t = ar;
                }
                vs[j][k] = sqrt( vh[j][k]*vh[j][k] + r*r )* t / s;
            }
        }
        

        set_scalar_vars
        endloop (15);
        refresh_vars
   }
   while (count < loop);

    /*
     *******************************************************************
     *   Kernel 16 -- Monte Carlo search loop
     *******************************************************************
     */

    parameters (16);
    refresh_vars
    
    ii = _n / 3;
    lb = ii + ii;
    long loc_k2, loc_k3 = 0;
    long loc_m16 = 0;
    long loc_j5 = j5;
    k3 = k2 = 0;
    long base_k2 = k2;
    // loc_m16 is never changed as it sits behind a if else statement where all branches jump to an earlier point in the loop

    j2 =1;
    do
    {
        long all_threads_k2 = 0;
        long all_threads_j5 = 0;
        _n = n;
        base_k2 = k2;
        i1 = loc_m16 = 1;
        loc_k3 = 0;
        loc_k2 = 0;
        //printf("n=%ld, j5=%ld, m16=%ld, k2=%ld, k3=%ld\n", _n,loc_j5,loc_m16,loc_k2,loc_k3);

        //The original forumula for j2 will always evaluate to 1 because m16 is invariant
        //j2 = ( _n + _n )*( loc_m16 - 1 ) + 1;
        omp_lock_t tobt_lock;
        omp_init_lock(&tobt_lock);
        int tid_of_breaking_thread = -1;

        #pragma omp parallel shared(tid_of_breaking_thread, tobt_lock, all_threads_k2) private(loc_k2,k,j4, loc_j5,tmp) firstprivate(base_k2,lb,zone,_n,plan,r,s,t) reduction(+:loc_k3)
        {
            int num_threads = omp_get_num_threads();
            int tid = omp_get_thread_num();
            int extra_indexes = _n%num_threads;
            int even_indexes = (_n-extra_indexes)/num_threads;
            int start_index = tid*even_indexes + (tid != 0)*extra_indexes + 1;
            int end_index = start_index + even_indexes + (tid == 0)*extra_indexes -1;
            int loc_tid_of_breaking_thread = -1;
            loc_k2 = base_k2 + start_index - 1;
            for ( k=start_index ; k<=end_index ; k++ )
            {
                int temp_b_tid_cpy = tid_of_breaking_thread;
                if(temp_b_tid_cpy != -1 && temp_b_tid_cpy < tid){
                    break;
                }
                loc_k2++;
                j4 = 1 + k + k;
                loc_j5 = zone[j4-1];
                if ( loc_j5 < _n )
                {
                    if ( loc_j5+lb < _n )
                    {                             /* 420 */
                        tmp = plan[loc_j5-1] - t;       /* 435 */
                    }
                    else
                    {
                        if ( loc_j5+ii < _n )
                        {                           /* 415 */
                            tmp = plan[loc_j5-1] - s;   /* 430 */
                        }
                        else
                        {
                            tmp = plan[loc_j5-1] - r;   /* 425 */
                        }
                    }
                }
                else if( loc_j5 == _n )
                {
                    if(tid_of_breaking_thread == -1 || tid_of_breaking_thread > tid){
                        omp_set_lock(&tobt_lock);
                        if(tid_of_breaking_thread == -1 || tid_of_breaking_thread > tid){
                            tid_of_breaking_thread = tid;
                        }
                        omp_unset_lock(&tobt_lock);
                    }
                    break;                        /* 475 */
                }
                else
                {
                    loc_k3++;                           /* 450 */
                    tmp=(d[loc_j5-1]-(d[loc_j5-2]*(t-d[loc_j5-3])*(t-d[loc_j5-3])+(s-d[loc_j5-4])*
                                (s-d[loc_j5-4])+(r-d[loc_j5-5])*(r-d[loc_j5-5])));
                }
                if ( tmp < 0.0 )
                {
                    if ( zone[j4-2] < 0 )            /* 445 */
                        continue;                   /* 470 */
                    else if ( !zone[j4-2] ){
                        if(tid_of_breaking_thread == -1 || tid_of_breaking_thread > tid){
                            omp_set_lock(&tobt_lock);
                            if(tid_of_breaking_thread == -1 || tid_of_breaking_thread > tid){
                                tid_of_breaking_thread = tid;
                            }
                            omp_unset_lock(&tobt_lock);
                        }
                        break;          
                    }
                }
                else if ( tmp )
                {
                    if ( zone[j4-2] > 0 )           /* 440 */
                        continue;                   /* 470 */
                    else if ( !zone[j4-2] ){
                        if(tid_of_breaking_thread == -1){
                            omp_set_lock(&tobt_lock);
                            if(tid_of_breaking_thread == -1 || tid_of_breaking_thread > tid){
                                tid_of_breaking_thread = tid;
                            }
                            omp_unset_lock(&tobt_lock);
                        }
                        break;
                    }
                        /* 480 */
                }
                if(tid_of_breaking_thread == -1 || tid_of_breaking_thread > tid){
                    omp_set_lock(&tobt_lock);
                    if(tid_of_breaking_thread == -1 || tid_of_breaking_thread > tid){
                        tid_of_breaking_thread = tid;
                    }
                    omp_unset_lock(&tobt_lock);
                }
                break;  
            }

            #pragma omp barrier
            #pragma single
            {
                if(tid_of_breaking_thread == -1){
                    tid_of_breaking_thread = num_threads -1;
                }
            }
            #pragma omp barrier
            if(tid == tid_of_breaking_thread){
                all_threads_k2 = loc_k2;
                all_threads_j5 = loc_j5;
            }
            if(tid > tid_of_breaking_thread){
                loc_k3 = 0;
            }
        }
        k3 += loc_k3;  
        j5 = all_threads_j5;      
        k2 = all_threads_k2;
        m16 = 1;
        endloop (16);
        base_k2 = k2;
        loc_k3 = k3;
        refresh_vars
   }
   while (count < loop);
   
    /*
     *******************************************************************
     *   Kernel 17 -- implicit, conditional computation
     *******************************************************************
     */

    parameters (17);
     refresh_vars 

    do
    {
    _n = n;
    scale = 5.0 / 3.0;
    xnm = 1.0 / 3.0;
    e6 = 1.03 / 3.07;
    long f_count,s_count;
    f_count = s_count = 0;


    double xnm_arr[101];



    //Unroll first iteration of the loop so we can ignore e6
    xnm_arr[_n-1] = xnm;
    xnm = xnm*vsp[_n-1] + vstp[_n-1];
    vxne[_n-1] = xnm;
    ve3[_n-1] = xnm;
    vxnd[_n-1] = e6;

    for(int i = _n -2; i > 0; i--){
        xnm_arr[i] = xnm;
        //l61 first half
        e3 = xnm*vlr[i] + vlin[i];
        xnei = vxne[i];
        //___________________________________ Below is neccesary for mine above is for checsum variables

        //vxnd[i] = xnm;
        xnc = scale*xnm*vlr[i] + scale*vlin[i];
        
        //l60
        // xnc = 1.6*xnm
        // therefore xnm > xnc will always be false
        if ( xnm > xnc || vxne[i] > xnc ){
            //printf("Took first rout on %d\n", i);
            //ve3[i] = xnm*vsp[i]*vsp[i] + vstp[i]*vsp[i] + vstp[i];
            //______________________________________________-

            xnm =     xnm*vsp[i] + vstp[i];
            //vxne[i] = xnm*vsp[i] + vstp[i];
            
        }else
        //l61 second half
        {
            
            //ve3[i] = xnm*vlr[i] + vlin[i];


            //_____________________________________
            xnm = 2*(xnm*vlr[i] + vlin[i]) - xnm;
            //vxne[i] = 2*(xnm*vlr[i] + vlin[i])  - vxne[i];
        }
    }


    #pragma omp parallel for private(i) firstprivate(_n,vxnd,xnm_arr,scale, vlr, vlin, ve3, vsp, vstp, vxne)
    for(int i = _n -2; i > 0; i--){
        //printf("%d\n",__LINE__);
        vxnd[i] = xnm_arr[i];
        xnc = scale*xnm_arr[i]*(vlr[i] + vlin[i]);
        if(xnm_arr[i] > xnc || vxne[i] > xnc){
            ve3[i] = xnm_arr[i]*vsp[i]  + vstp[i];
            vxne[i] = xnm_arr[i-1]*vsp[i] + vstp[i];
        }else{
            ve3[i] = xnm_arr[i]*vlr[i] + vlin[i];
            vxne[i] = 2*ve3[i] - vxne[i];
        }
    }

    as1.Xtra[39] = xnm;        
        endloop (17);
        refresh_vars
    }
    while (count < loop);

    /*
     *******************************************************************
     *   Kernel 18 - 2-D explicit hydrodynamics fragment
     *******************************************************************
     */

    parameters (18);
     refresh_vars 

    do
    {
       t = 0.0037;
       s = 0.0041;
       kn = 6;
       jn = n;
       #pragma omp parallel for shared(za,zb,zp,zr,zq,zm) firstprivate(kn,jn) private(k, j)
       for ( k=1 ; k<kn ; k++ )
       {
            for ( j=1 ; j<jn ; j++ ){
                za[k][j] = ( zp[k+1][j-1] +zq[k+1][j-1] -zp[k][j-1] -zq[k][j-1] )*
                            ( zr[k][j] +zr[k][j-1] ) / ( zm[k][j-1] +zm[k+1][j-1]);
                zb[k][j] = ( zp[k][j-1] +zq[k][j-1] -zp[k][j] -zq[k][j] ) *
                            ( zr[k][j] +zr[k-1][j] ) / ( zm[k][j] +zm[k][j-1]);
            }
       }
       #pragma omp parallel for shared(za,zb,zp,zr,zq,zm,zz,zu,zv) firstprivate(kn,jn) private(k, j)
        for ( k=1 ; k<kn ; k++ )
        {

            for ( j=1 ; j<jn ; j++ )
            {
                zu[k][j] += s*( za[k][j]   *( zz[k][j] - zz[k][j+1] ) -
                                za[k][j-1] *( zz[k][j] - zz[k][j-1] ) -
                                zb[k][j]   *( zz[k][j] - zz[k-1][j] ) +
                                zb[k+1][j] *( zz[k][j] - zz[k+1][j] ) );
                zv[k][j] += s*( za[k][j]   *( zr[k][j] - zr[k][j+1] ) -
                                za[k][j-1] *( zr[k][j] - zr[k][j-1] ) -
                                zb[k][j]   *( zr[k][j] - zr[k-1][j] ) +
                                zb[k+1][j] *( zr[k][j] - zr[k+1][j] ) );
            }
        }
        #pragma omp parallel for shared(zr,zz,zu,zv) firstprivate(kn,jn,t) private(k, j)
        for ( k=1 ; k<kn ; k++ )
        {

            for ( j=1 ; j<jn ; j++ )
            {
                zr[k][j] = zr[k][j] + t*zu[k][j];
                zz[k][j] = zz[k][j] + t*zv[k][j];
            }
        }
        

        set_scalar_vars
        endloop (18);
        refresh_vars
    }
    while (count < loop);

    /*
     *******************************************************************
     *   Kernel 19 -- general linear recurrence equations
     *******************************************************************
     */

    parameters (19);
    refresh_vars
    
    
    kb5i = 0;
    
    // N/nt > kb5i
    do
    {
        long kb5i_idx_offset = imax( kb5i + 1 , 1);
        double stb5_arr[1001]={0.0};
        stb5 = as1.Xtra[35];
        b5[0] = sa[0-kb5i] + stb5*sb[k-kb5i];
        stb5_arr[0] = sa[0] + stb5*(sb[0] - 1);
        _n = n;                

        //b5[k] = sa[k-kb5i] + stb5*sb[k-kb5i];

        //stb5[k] = sa[k-kb5i] + stb5*sb[k- kb5i] - stb5;
        //stb5 = sa[k - kb5i] + stb5*(sb[k-kb5i] - 1);
    
         _n = n;
        #pragma omp parallel
        {
            int num_threads = omp_get_num_threads();
            int tid = omp_get_thread_num();
            int even_jobs = (int)floor(_n /num_threads);
            int extra_jobs = _n %num_threads;
            if(tid == 0){
                even_jobs += extra_jobs;
            }
            int start_idx = even_jobs*tid + extra_jobs*(tid != 0);
            int end_idx = start_idx + even_jobs;

            //Stage one
            if(tid == 0){
                for(int loc = kb5i_idx_offset ; loc < end_idx; loc++){
                    b5[loc] = sa[loc - kb5i] + stb5_arr[loc-1]*sb[loc-kb5i];
                    stb5_arr[loc] =  b5[loc] - stb5_arr[loc-1];
                }
            }else{
                //z*(y-x) = z*y - z*x
                
                //coeff[start_idx] = -z[start_idx];
                //constant[start_idx] = constant[start_idx];
                //for(int loc = start_idx+1; loc < end_idx; loc++){
                //    coeff[loc] = -z[loc] *coeff[loc-1];
                //    constant[loc] = -z[loc]*(constant[loc-1] - y[loc]); 
                //}

                //sa[k] + stb5[k-1]*(sb[k] - 1)


                coeff[start_idx] = sb[start_idx-kb5i] -1;
                constant[start_idx] = sa[start_idx - kb5i];
                for(int loc = start_idx+1; loc < end_idx; loc++){
                    coeff[loc] = (sb[loc-kb5i] -1)*coeff[loc-1];
                    constant[loc] = constant[loc-1] + sa[loc-kb5i]; 
                }
            }
            #pragma omp barrier
            #pragma omp single
            {
                int last_xi_first_block_idx = even_jobs+extra_jobs-1;
                for(int last_x_block_i = last_xi_first_block_idx+even_jobs; last_x_block_i < _n-1; last_x_block_i+=even_jobs){
                    x[last_x_block_i] = coeff[last_x_block_i]*stb5_arr[last_x_block_i - even_jobs] + constant[last_x_block_i];
                }
            }
            #pragma omp barrier
            if(tid > 0){
                for(int loc = start_idx; loc < end_idx; loc++ ){
                    double temp_b5 = sa[loc-kb5i] + stb5_arr[loc-1]*sb[loc-kb5i];
                    b5[loc] = temp_b5;
                    stb5_arr[loc] = temp_b5 - stb5_arr[loc-1];
                }
            }
            #pragma omp barrier
        }        
        stb5 = stb5_arr[_n-1];
        //sa[_n-1] + stb5_arr[_n-1]*(sb[_n-1]-1.0);
        // This can be reformulated into the same linear recurrance as before
        // b5[k] = sa[k-kb5i] + stb5*sb[k-kb5i]
        //stb5 =  sa[k-kb5i] + stb5*sb[k-kb5i] -stb5
        // Since the variable stb5 on the right hasn't been update we can write
        // stb5[k] = sa[k-kb5i] + stb5[k-1]*(sb[k-kb5i]- 1 )
        // which since the variable kb5i is always 0 can be written
        // stb5[k] = sa[k] + stb5[k-1]*(sb[k] - 1)
        //// The same parallelization technique as in kernel 5 can be used
        //for ( k=kb5i ; k<n ; k++ )
        //{
        //    
        //    stb5 = sa[k - kb5i] + stb5*(sb[k-kb5i] - 1);
        //    b5[k] = sa[k-kb5i] + stb5*sb[k-kb5i];
        //    //b5[k+kb5i] = sa[k] + stb5*sb[k];
        //    //stb5 = b5[k+kb5i] - stb5;
        //}
        //// Same thing but with the loop indexes reversed assume an attempt to obsuficate the recurrence relation
        // As these loops were developed as compiler competence tests
        // Since it's the same thing I don't parallelize it
        for ( i=1 ; i<=n ; i++ )
        {
            k = n - i;
            b5[k+kb5i] = sa[k] + stb5*sb[k];
            stb5 = b5[k+kb5i] - stb5;
        }

        as1.Xtra[35] = stb5;
        endloop (19);
        refresh_vars
        
    }
    while (count < loop);
    
    /*
     *******************************************************************
     * Kernel 20 - Discrete ordinates transport, conditional recurrence on xx
     *******************************************************************
    */

    parameters (20);
    refresh_vars
    
    do
    {
        for ( k=0 ; k<n ; k++ )
        {
           di = y[k] - g[k] / ( xx[k] + dk );
           dn = 0.2;

           if ( di )
           {
               dn = z[k]/di ;
               if ( t < dn ) dn = t;
               if ( s > dn ) dn = s;
           }
           x[k] = ( ( w[k] + v[k]*dn )* xx[k] + u[k] ) / ( vx[k] + v[k]*dn );
           xx[k+1] = ( x[k] - xx[k] )* dn + xx[k];
        }
        
        set_scalar_vars
        endloop (20);
        refresh_vars
        
    }
    while (count < loop);

    /*
     *******************************************************************
     *   Kernel 21 -- matrix*matrix product
     *******************************************************************
     */

    parameters (21);
    refresh_vars
    double partial_mult;
    do
    {
        #pragma omp parallel for shared(px, vy,cx) private(partial_mult,i,j)
        for ( k=0 ; k<25 ; k++ )
        {
            for ( i=0 ; i<25 ; i++ )
            {
                for ( j=0 ; j<n ; j++ )
                {
                    partial_mult = vy[k][i] * cx[j][k];
                    #pragma omp atomic update
                    px[j][i] += partial_mult;
                }
            }
        }
        

        set_scalar_vars
        endloop (21);
        refresh_vars
    }
    while (count < loop);
    
    /*
     *******************************************************************
     *   Kernel 22 -- Planckian distribution
     *******************************************************************
     */

    parameters (22);
    refresh_vars

    expmax = 20.0;
    u[n-1] = 0.99*expmax*v[n-1];
    do
    {   
        #pragma omp parallel for
        for ( k=0 ; k<n ; k++ )
        {
            y[k] = u[k] / v[k];
            w[k] = x[k] / ( exp( y[k] ) -1.0 );
        }
        

        set_scalar_vars
        endloop (22);
        refresh_vars
    }
    while (count < loop);

    /*
     *******************************************************************
     *   Kernel 23 -- 2-D implicit hydrodynamics fragment
     *******************************************************************
     */


    parameters (23);
    refresh_vars
    long k23_n = n;
    do
    {
            //Note that while can be done due to each location on the j-k grid requiring updates from all neighbors
            // This creates an infinite loop of dependcies
            // Therefore so long as we are careful with the update, as to not lose any values from the grid
            // the update order will not matter so no attempt is made
            //Passes numerical reliablity tests with a matching checksum
            #pragma omp parallel for private(j,qa,k), firstprivate(za,zr,zb,zv,zz)
            for ( j=1 ; j<6 ; j++ )
            {
                for ( k=1 ; k<k23_n ; k++ )
                {
                    qa = za[j+1][k]*zr[j][k] + za[j-1][k]*zb[j][k] +
                    za[j][k+1]*zu[j][k] + za[j][k-1]*zv[j][k] + zz[j][k];
                    #pragma omp atomic update
                    za[j][k] += 0.175*( qa - za[j][k] );
                }
        
            }

        set_scalar_vars
        endloop (23);
        refresh_vars
    }
    while (count < loop);

    /*
     *******************************************************************
     *   Kernel 24 -- find location of first minimum in array
     *******************************************************************
     */

    parameters (24);
    refresh_vars
    long k24_n = n;
    x[n/2] = -1.0e+10;
    do
    {

        //Define a reduction over the struct min_index which contains two fields, the location of the value and the value
        //Reduces to find the minimum value and index of the array
        // value is initialized to __MAX_DBL__
        #pragma omp declare reduction\
        (loc_first_min:min_index:omp_out=reduce_min_index(omp_out, omp_in) )\
        initializer(omp_priv=MIN_INDEX())

        m24 = 0;
       
        min_index min_val_loc = MIN_INDEX();
        #pragma omp parallel for firstprivate(k24_n) shared(x) private(k) reduction(loc_first_min:min_val_loc) 
        for ( k=1 ; k<k24_n ; k++ )
        {
            if ( x[k] < min_val_loc.val ){
                min_val_loc.index = k;
                min_val_loc.val = x[k];
            };
        }
        m24 = min_val_loc.index;
        set_scalar_vars
        endloop (24);
        refresh_vars
    }
    while (count < loop);
   
   return;
 } // kernels

/************************************************************************
 *        endloop procedure - calculate checksums and MFLOPS            *
 ************************************************************************/
 
long endloop(long which)
{
    init_kernel_vars
  double now = 1.0, useflops;
  long   i, j, k, m;
  double Scale = 1000000.0;
  Boolean  reinit = TRUE;
  Boolean  getend = FALSE;
    
  count = count + 1;
  if (count >= loop)   /* else return */
  {

/************************************************************************
 *               End of standard set of loops for one kernel            *
 ************************************************************************/
      
     count2 = count2 + 1;
     if (count2 == extra_loops[_section][which]) getend = TRUE;
     if (count2 == extra_loops[_section][which] || runRel)
                          /* else re-initialise parameters if required */
     {
         reinit = FALSE;

/************************************************************************
 *           End of extra loops for runSecs seconds execution time            *
 ************************************************************************/
                        
       Checksum[_section][which] = 0;
       if (which == 1)
       {     
           for ( k=0 ; k<n ; k++ )
           {
                Checksum[_section][1] =  Checksum[_section][1] + x[k]*(double)(k+1);
           }
           useflops = nflops * (double)(n * loop);
       }
       if (which == 2)
       {
          for ( k=0 ; k<n*2 ; k++ )
          {
               Checksum[_section][2] = Checksum[_section][2] + x[k]
                                         * (double)(k+1);
          }
          useflops = nflops * (double)((n-4) * loop);
       }
       if (which == 3)
       {
           Checksum[_section][3] = q;
           useflops = nflops * (double)(n * loop);
       }
       if (which == 4)
       {
          for ( k=0 ; k<3 ; k++ )
          {
                Checksum[_section][4] = Checksum[_section][4] + v[k]
                                          * (double)(k+1);
          }
          useflops = nflops * (double) ((((n-5)/5)+1) * 3 * loop); 
       }
       if (which == 5)
       {
          for ( k=1 ; k<n ; k++ )
          {
              Checksum[_section][5] = Checksum[_section][5] + x[k]
                                        * (double)(k);
          }
          useflops = nflops * (double)((n-1) * loop);
       }
       if (which == 6)
       {
          for ( k=0 ; k<n ; k++ )
          {
         
             Checksum[_section][6] = Checksum[_section][6] + w[k]
                                       * (double)(k+1);
         
          }
          useflops = nflops * (double)(n * ((n - 1) / 2) * loop);
       } 
       if (which == 7)
       {      
          for ( k=0 ; k<n ; k++ )
          {
              Checksum[_section][7] = Checksum[_section][7] + x[k]
                                        * (double)(k+1);
          }
          useflops = nflops * (double)(n * loop);
       }
       if (which == 8)
       {
          for ( i=0 ; i<2 ; i++ )
          {        
              for ( j=0 ; j<101 ; j++ )
              {
                  for ( k=0 ; k<5 ; k++ )
                  { 
                      m = 101 * 5 * i + 5 * j + k + 1;
                      if (m < 10 * n + 1)
                      {
                          Checksum[_section][8] = Checksum[_section][8]
                                  + u1[i][j][k] * (double)m
                                  + u2[i][j][k] * (double)m + u3[i][j][k] * (double)m;
                      }
                  }
              }
          }
          useflops = nflops * (double)(2 * (n - 1) * loop);
       }
       if (which == 9)
       {
           for ( i=0 ; i<n  ; i++ )
           {
               for ( j=0 ; j<25 ; j++ )
               {
                   m = 25 * i + j + 1;
                   if (m < 15 * n + 1)
                   {
                       Checksum[_section][9] = Checksum[_section][9]
                                             + px[i][j] * (double)(m);
                   }
               }
           }
           useflops = nflops * (double)(n * loop);
       }
       if (which == 10)
       {
           for ( i=0 ; i<n ; i++ )
           {
               for (j=0 ; j<25 ; j++ )
              {
                   m = 25 * i + j + 1;
                   if (m < 15 * n + 1)
                   {
                       Checksum[_section][10] = Checksum[_section][10]
                                              + px[i][j] * (double)(m);
                   }                  
              }
           }
           useflops = nflops * (double)(n * loop);
       }
       if (which == 11)
       { 
           for ( k=1 ; k<n ; k++ )
           {
                Checksum[_section][11] = Checksum[_section][11]
                                           + x[k] * (double)(k);
           }
           useflops = nflops * (double)((n - 1) * loop);
       }
       if (which == 12)
       { 
           for ( k=0 ; k<n-1 ; k++ )
           {
                Checksum[_section][12] = Checksum[_section][12] + x[k]
                                           * (double)(k+1);
           }
           useflops = nflops * (double)(n * loop);
       }
       if (which == 13)
       {
          for ( k=0 ; k<2*n ; k++ )                  
          {
             for ( j=0 ; j<4 ; j++ )    
              {
                  m = 4 * k + j + 1;
                  Checksum[_section][13] = Checksum[_section][13]
                                             + p[k][j]* (double)(m);
              }
          }
          for ( i=0 ; i<8*n/64 ; i++ )
          {
              for ( j=0 ; j<64 ; j++ )
              {
                  m = 64 * i + j + 1;
                  if (m < 8 * n + 1)
                  {
                      Checksum[_section][13] = Checksum[_section][13]
                                                  + h[i][j] * (double)(m);
                  }
              }
         }
         useflops = nflops * (double)(n * loop);  
       }
       if (which == 14)
       {
          for ( k=0 ; k<n ; k++ )
          {
                Checksum[_section][14] = Checksum[_section][14]
                                           + (xx[k] + vx[k]) * (double)(k+1);
          }
          for ( k=0 ; k<67 ; k++ )
          {
              Checksum[_section][14] = Checksum[_section][14] + rh[k]
                                         * (double)(k+1);
          }
          useflops = nflops * (double)(n * loop);
       }
       if (which == 15)
       {
           for ( j=0 ; j<7 ; j++ )
           {
               for ( k=0 ; k<101 ; k++ )
               {
                  m = 101 * j + k + 1;
                  if (m < n * 7 + 1)
                  {
                      Checksum[_section][15] = Checksum[_section][15]
                                       + (vs[j][k] + vy[j][k]) * (double)(m);
                  }
               }
           }
           useflops = nflops * (double)((n - 1) * 5 * loop);
       }
       if (which == 16)
       {
           Checksum[_section][16] =  (double)(k3 + k2 + j5 + m16);
           useflops = (double)(k2 + k2 + 10 * k3);
       }
       if (which == 17)
       {
           Checksum[_section][17] = xnm;
           for ( k=0 ; k<n ; k++ )
           {
               Checksum[_section][17] = Checksum[_section][17]
                                       + (vxne[k] + vxnd[k]) * (double)(k+1);
           }
           useflops = nflops * (double)(n * loop); 
       }
       if (which == 18)
       {
          for ( k=0 ; k<7 ; k++ )    
           {
               for ( j=0 ; j<101 ; j++ )
               {
                   m = 101 * k + j + 1;
                   if (m < 7 * n + 1)
                   {
                       Checksum[_section][18] = Checksum[_section][18]
                                        + (zz[k][j] + zr[k][j]) * (double)(m);
                   }
               }
           }
           useflops = nflops * (double)((n - 1) * 5 * loop);
       }
       if (which == 19)
       {
          Checksum[_section][19] = stb5;
          for ( k=0 ; k<n ; k++ )
          {
              Checksum[_section][19] = Checksum[_section][19] + b5[k]
                                         * (double)(k+1);
          }             
          useflops = nflops * (double)(n * loop);
       } 
       if (which == 20)
       {
            for ( k=1 ; k<n+1 ; k++ )
            {
                Checksum[_section][20] = Checksum[_section][20] + xx[k]
                                           * (double)(k);
            }
            useflops = nflops * (double)(n * loop);
       }
       if (which == 21)
       {
           for ( k=0 ; k<n ; k++ )          
           {
               for ( i=0 ; i<25 ; i++ )
               {
                  m = 25 * k + i + 1;
                  Checksum[_section][21] = Checksum[_section][21]
                                             + px[k][i] * (double)(m);
               }
           }
           useflops = nflops * (double)(n * 625 * loop);      

       }
       if (which == 22)
       {
           for ( k=0 ; k<n ; k++ )
           {
                Checksum[_section][22] = Checksum[_section][22] + w[k]
                                           * (double)(k+1);
           }
           useflops = nflops * (double)(n * loop);      
       }
       if (which == 23)
       {
           for ( j=0 ; j<7 ; j++ )
           {        
                for ( k=0 ; k<101 ; k++ )
                {
                    m = 101 * j + k + 1;
                    if (m < 7 * n + 1)
                    {
                         Checksum[_section][23] = Checksum[_section][23]
                                                + za[j][k] * (double)(m);
                    }
                }
           }
           useflops = nflops * (double)((n-1) * 5 * loop);       
       }
       if (which == 24)
       {
           Checksum[_section][24] =  (double)(m24);
           useflops = nflops * (double)((n - 1) * loop); 
       }
       if (runRel) checkOut((int)which);
       if (getend)
       {
/************************************************************************
 *                           End of timing                              *
 ************************************************************************/
          count2 = 0;    
          end_time();
          RunTime[_section][which] = secs;

/************************************************************************
 *     Deduct overheads from time, calculate MFLOPS, display results    *
 ************************************************************************/

          RunTime[_section][which] = RunTime[_section][which]
                       - (double)(loop * extra_loops[_section][which]) * (double)overhead_l;
          FPops[_section][which] =  useflops * (double)extra_loops[_section][which];   
          Mflops[_section][which] = FPops[_section][which] / Scale
                                            / RunTime[_section][which];
          if (pass > 0)
          {


double sumsOut[3][25] = 
        {
             { 0.0,
             5.114652693224671e+04, 1.539721811668385e+03, 1.000742883066363e+01,
             5.999250595473891e-01, 4.548871642387267e+03, 4.375116344729986e+03,
             6.104251075174761e+04, 1.501268005625795e+05, 1.189443609974981e+05,
             7.310369784325296e+04, 3.342910972650109e+07, 2.907141294167248e-05,
             1.202533961842805e+11, 3.165553044000335e+09, 3.943816690352044e+04,
             5.650760000000000e+05, 1.114641772902486e+03, 1.015727037502299e+05,
             5.421816960147207e+02, 3.040644339351239e+07, 1.597308280710199e+08,
             2.938604376566697e+02, 3.549900501563623e+04, 5.000000000000000e+02
                                                                               },
    
             { 0.0,
             5.253344778937972e+02, 1.539721811668385e+03, 1.009741436578952e+00,
             5.999250595473891e-01, 4.589031939600982e+01, 8.631675645333210e+01,
             6.345586315784055e+02, 1.501268005625795e+05, 1.189443609974981e+05,
             7.310369784325296e+04, 3.433560407475758e+04, 7.127569130821465e-06,
             9.816387810944356e+10, 3.039983465145392e+07, 3.943816690352044e+04,
    
             6.480410000000000e+05, 1.114641772902486e+03, 1.015727037502299e+05,
             5.421816960147207e+02, 3.126205178815431e+04, 7.824524877232093e+07,
             2.938604376566697e+02, 3.549900501563623e+04, 5.000000000000000e+01
                                                                              },
         
             { 0.0,
             3.855104502494961e+01, 3.953296986903059e+01, 2.699309089320672e-01,
             5.999250595473891e-01, 3.182615248447483e+00, 1.120309393467088e+00,
             2.845720217644024e+01, 2.960543667875005e+03, 2.623968460874250e+03,
             1.651291227698265e+03, 6.551161335845770e+02, 1.943435981130448e-06,
             3.847124199949431e+10, 2.923540598672009e+06, 1.108997288134785e+03,
             5.152160000000000e+05, 2.947368618589361e+01, 9.700646212337041e+02,
             1.268230698051003e+01, 5.987713249475302e+02, 5.009945671204667e+07,
             6.109968728263972e+00, 4.850340602749970e+02, 1.300000000000000e+01
             }
        };






/************************************************************************
 *      Compare sumcheck with standard result, calculate accuracy       *
 ************************************************************************/
           
             check(which);
           
             printf ("%2ld %3ld x%4ld %2ld %13.6e %5.2f %8.2f %4ld %15.6e     %15.6e         %8.6f     %2ld\n",
                  which, xloops[_section][which], extra_loops[_section][which],
                  xflops[which], FPops[_section][which], RunTime[_section][which],
                  Mflops[_section][which], nspan[_section][which],sumsOut[_section][which],
                  Checksum[_section][which], relative_error(sumsOut[_section][which],Checksum[_section][which]), accuracy[_section][which]);
             if (reliability)
             { 
                 if (compareFail)
                 {
                     printf(" ERRORS have occurred - see log file\n");
                 }
             }
    
          }
       }
     }
     if (reinit || runRel && !getend)
     {
/************************************************************************
 *                     Re-initialise data if reqired                    *
 ************************************************************************/
       count = 0;  
       if (which == 2)
       {
          for ( k=0 ; k<n ; k++ )
          {
              x[k] = x0[k];
          }
       }
       if (which == 4)
       {
          m = (1001-7)/2;
          for ( k=6 ; k<1001 ; k=k+m )
          {
              x[k] = x0[k];
          }
       }
       if (which == 5)
       {
          for ( k=0 ; k<n ; k++ )
          {
              x[k] = x0[k];
          }
       }
       if (which == 6)
       {
          for ( k=0 ; k<n ; k++ )
          {
              w[k] = w0[k];
          }
       }
       if (which == 10)
       {
           for ( i=0 ; i<n ; i++ )
           {
               for (j=4 ; j<13 ; j++ )
              {
                  px[i][j] = px0[i][j];
              }
           }
       }
       if (which == 13)
       {           
           for ( i=0 ; i<n ; i++ )
           {
               for (j=0 ; j<4 ; j++ )
               {
                   p[i][j] = p0[i][j];
               }
           }
           for ( i=0 ; i<64 ; i++ )
           {
               for (j=0 ; j<64 ; j++ )
               {
                   h[i][j] = h0[i][j];
               }
           }
       }
       if (which == 14)
       {
           for ( i=0; i<n ; i++ )
           {
               rh[ir[i] - 1] = rh0[ir[i] - 1];
               rh[ir[i] ] = rh0[ir[i] ];
           }
       }
       if (which == 17)
       {
           for ( i=0; i<n ; i++ )
           {
               vxne[i] = vxne0[i];
           }
       }
       if (which == 18)
       {
          for ( i=1 ; i<6 ; i++ )
          {
              for (j=1 ; j<n ; j++ )
              {
                  zr[i][j] = zr0[i][j];
                  zu[i][j] = zu0[i][j];
                  zv[i][j] = zv0[i][j];
                  zz[i][j] = zz0[i][j];  
              }
          }
       }
       if (which == 21)
       {
           for ( i=0 ; i<n ; i++ )
           {
               for (j=0 ; j<25 ; j++ )
              {
                  px[i][j] = px0[i][j];
              }
           }
       }
       if (which == 23)
       {
          for ( i=1 ; i<6 ; i++ )
          {
              for (j=1 ; j<n ; j++ )
              {
                  za[i][j] = za0[i][j];
              }
          }
       }
       k3 = k2 = 0;
       stb5 = stb50;
       xx[0] = xx0;
        set_scalar_vars
     }
  }
  
  return 0;
} // endloop

/************************************************************************
 *          init procedure - initialises data for all loops             *
 ************************************************************************/

 void init(long which)
 {
    init_kernel_vars
    long   i, j, k, l, m, nn;
    double ds, dw, rr, ss;
    double fuzz, fizz, buzz, scaled, one;  

     scaled =  (double)(10.0);
     scaled =  (double)(1.0) / scaled;
     fuzz =    (double)(0.0012345);
     buzz =    (double)(1.0) + fuzz;
     fizz =    (double)(1.1) * fuzz;
     one =     (double)(1.0);
     
//     for ( k=0 ; k<19977 + 34132 ; k++)
     for ( k=0 ; k<1001; k++)
     {
/*
         if (k == 19977)
         {
                fuzz = (double)(0.0012345);
                buzz = (double) (1.0) + fuzz;
                fizz = (double) (1.1) * fuzz;
         }         
*/
         buzz = (one - fuzz) * buzz + fuzz;
         fuzz = - fuzz;
         u[k] = (buzz - fizz) * scaled;

     }
     for ( k=1001 ; k<(19977 + 34132); k++)
     {
         if (k == 19977)
         {
                fuzz = (double)(0.0012345);
                buzz = (double) (1.0) + fuzz;
                fizz = (double) (1.1) * fuzz;
         }         
         buzz = (one - fuzz) * buzz + fuzz;
         fuzz = - fuzz;
         u[k] = (buzz - fizz) * scaled;
     }
     
     fuzz = (double)(0.0012345);
     buzz = (double) (1.0) + fuzz;
     fizz = (double) (1.1) * fuzz;
     
     for ( k=1 ; k<40 ; k++)
     {
         buzz = (one - fuzz) * buzz + fuzz;
         fuzz = - fuzz;
         xtra[k] = (buzz - fizz) * scaled;
     }
     refresh_vars

    ds = 1.0;
    dw = 0.5;
    for ( l=0 ; l<4 ; l++ )   
    {
         for ( i=0 ; i<512 ; i++ )
        {
            p[i][l] = ds;
            ds = ds + dw;
        }
    }
     for ( i=0 ; i<96 ; i++ )
     {
         e[i] = 1;
         f[i] = 1;
     }    

     
         iqranf();
         dw = -100.0;
         for ( i=0; i<1001 ; i++ )
         {
             dex[i] = dw * dex[i];
             grd[i] = (double)ix[i];
         }     
         flx = 0.001;
         as1.Xtra[27] = flx;

                  
         d[0]= 1.01980486428764;
         nn = n16;
    
         for ( l=1 ; l<300 ; l++ )
         {
              d[l] = d[l-1] + 1.000e-4 / d[l-1];
         }
         rr = d[nn-1];
         for ( l=1 ; l<=2 ; l++ )
         {
             m = (nn+nn)*(l-1);
             for ( j=1 ; j<=2 ; j++ )
             {
                 for ( k=1 ; k<=nn ; k++ )
                 {
                     m = m + 1;
                     ss = (double)(k);
                     plan[m-1] = rr * ((ss + 1.0) / ss);
                     zone[m-1] = k + k;
                 }
            }
        }
        k = nn + nn + 1;
        zone[k-1] = nn;
        
        if (which == 16)
        {
             r = d[n-1];
             s = d[n-2];
             t = d[n-3];
             k3 = k2 = 0;
             as1.Xtra[30] = r;
             as1.Xtra[32] = s;
             as1.Xtra[36] = t;
        }
        expmax = 20.0;
        as1.Xtra[26] = expmax;
        if (which == 22)
        {
             u[n-1] = 0.99*expmax*v[n-1];
        }
        if (which == 24)
        {
             x[n/2] = -1.0e+10;
        }

/************************************************************************
 *                 Make copies of data for extra loops                  *
 ************************************************************************/
 
        for ( i=0; i<1001 ; i++ )
        {
            x0[i] = x[i];
            w0[i] = w[i];
        }
        for ( i=0 ; i<101 ; i++ )
        {
            for (j=0 ; j<25 ; j++ )
            {
                px0[i][j] = px[i][j];
            }
        }
        for ( i=0 ; i<512 ; i++ )
        {
            for (j=0 ; j<4 ; j++ )
            {
                p0[i][j] = p[i][j];
            }
        }
        for ( i=0 ; i<64 ; i++ )
        {
            for (j=0 ; j<64 ; j++ )
            {
                h0[i][j] = h[i][j];
            }
        }
        for ( i=0; i<2048 ; i++ )
        {
            rh0[i] = rh[i];
        }
        for ( i=0; i<101 ; i++ )
        {
            vxne0[i] = vxne[i];
        }
        for ( i=0 ; i<7 ; i++ )
        {
            for (j=0 ; j<101 ; j++ )
            {
                zr0[i][j] = zr[i][j];
                zu0[i][j] = zu[i][j];
                zv0[i][j] = zv[i][j];
                zz0[i][j] = zz[i][j];
                za0[i][j] = za[i][j];
            }
        }

        as1.Stb50 = stb5;
        as1.Xx0 = xx[0];
    return;
 }

/************************************************************************
 *   parameters procedure for loop counts, Do spans, sumchecks, FLOPS   *
 ************************************************************************/

   long parameters(long which)
   {

       long nloops[3][25] =
            { {0, 1001, 101, 1001, 1001, 1001, 64, 995, 100,
                   101, 101, 1001, 1000, 64, 1001, 101, 75,
                   101, 100, 101, 1000, 101, 101, 100, 1001  },
              {0,  101, 101, 101, 101, 101,  32, 101, 100,
                   101, 101, 101, 100,  32, 101, 101,  40,
                   101, 100, 101, 100,  50, 101, 100, 101    },
              {0,   27, 15, 27, 27, 27,  8, 21, 14,
                    15, 15, 27, 26,  8, 27, 15, 15,
                    15, 14, 15, 26, 20, 15, 14, 27           }  };


                             
       long lpass[3][25] =
             { {0, 7, 67,  9, 14, 10,  3,  4, 10, 36, 34, 11, 12,
                  36, 2,  1,  25, 35,  2, 39,  1,  1, 11,  8,  5  },
               {0, 40, 40, 53, 70, 55,  7, 22,  6, 21, 19, 64, 68,
                   41, 10,  1, 27, 20,  1, 23,  8,  1,  7,  5, 31   },
               {0, 28, 46, 37, 38, 40, 21, 20,  9, 26, 25, 46, 48,
                   31,  8,  1, 14, 26,  2, 28,  7,  1,  8,  7, 23 } };

       double sums[3][25] = 
        {
         { 0.0,
         //1                        2                       3
         5.114652693224671e+04, 1.539721811668385e+03, 1.000742883066363e+01,
         5.999250595473891e-01, 4.548871642387267e+03, 4.375116344729986e+03,
         6.104251075174761e+04, 1.501268005625798e+05, 1.189443609974981e+05,
         7.310369784325296e+04, 3.342910972650109e+07, 2.907141294167248e-05,
         1.202533961842803e+11, 3.165553044000334e+09, 3.943816690352042e+04,
         5.650760000000000e+05, 1.114641772902486e+03, 1.015727037502300e+05,
         5.421816960147207e+02, 3.040644339351239e+07, 1.597308280710199e+08,
         2.938604376566697e+02, 3.549900501563623e+04, 5.000000000000000e+02
                                                                           },

         { 0.0,
         5.253344778937972e+02, 1.539721811668385e+03, 1.009741436578952e+00,
         5.999250595473891e-01, 4.589031939600982e+01, 8.631675645333210e+01,
         6.345586315784055e+02, 1.501268005625798e+05, 1.189443609974981e+05,
         7.310369784325296e+04, 3.433560407475758e+04, 7.127569130821465e-06,
         9.816387810944345e+10, 3.039983465145393e+07, 3.943816690352042e+04,
         6.480410000000000e+05, 1.114641772902486e+03, 1.015727037502300e+05,
         5.421816960147207e+02, 3.126205178815431e+04, 7.824524877232093e+07,
         2.938604376566697e+02, 3.549900501563623e+04, 5.000000000000000e+01
                                                                          },
     
         { 0.0,
         3.855104502494961e+01, 3.953296986903059e+01, 2.699309089320672e-01,
         5.999250595473891e-01, 3.182615248447483e+00, 1.120309393467088e+00,
         2.845720217644024e+01, 2.960543667875003e+03, 2.623968460874250e+03,
         1.651291227698265e+03, 6.551161335845770e+02, 1.943435981130448e-06,
         3.847124199949426e+10, 2.923540598672011e+06, 1.108997288134785e+03,
         5.152160000000000e+05, 2.947368618589360e+01, 9.700646212337040e+02,
         1.268230698051003e+01, 5.987713249475302e+02, 5.009945671204667e+07,
         6.109968728263972e+00, 4.850340602749970e+02, 1.300000000000000e+01
                                                                         } };
                               
 
     
       double number_flops[25] = {0, 5., 4., 2., 2., 2., 2., 16., 36., 17.,
                                      9., 1., 1., 7., 11., 33.,10., 9., 44.,
                                      6., 26., 2., 17., 11., 1.};
       double now = 1.0;
      
                           
       n = nloops[_section][which];
       nspan[_section][which] = n;
       n16 = nloops[_section][16];
       nflops = number_flops[which];
       xflops[which] = (long)nflops;
       loop = lpass[_section][which];
       xloops[_section][which] = loop;
       loop = loop * mult;
       MasterSum = sums[_section][which];
       count = 0;

       init(which);
       
/************************************************************************
 *                   Start timing first pass only                       *
 ************************************************************************/

       if (count2 == 0)
       {
           start_time();                 
       }
              
       return 0;
   } // parameters

/************************************************************************
 *          check procedure to check accuracy of calculations           *
 ************************************************************************/
   
   void check(long which)
   {
        long maxs = 16;
        double xm, ym, re, min1, max1;

        xm = MasterSum;
        ym = Checksum[_section][which];
      

       if (xm * ym < 0.0)
       {
           accuracy[_section][which] = 0;
       }
       else
       {
           if ( xm == ym)
           {
               accuracy[_section][which] = maxs;
           }
           else
           {
               xm = fabs(xm);
               ym = fabs(ym);
               min1 = xm;
               max1 = ym;
               if (ym < xm)
               {
                   min1 = ym;
                   max1 = xm;
               }
               re = 1.0 - min1 / max1;
               accuracy[_section][which] =
                                        (long)( fabs(log10(fabs(re))) + 0.5);
           }
       }

       return;
   } 
   
/************************************************************************
 *      iqranf procedure - random number generator for Kernel 14        *
 ************************************************************************/
  
    void iqranf()
      {
        long    *ix = as1.Ix;
        long   inset, Mmin, Mmax, nn, i, kk;
        double span, spin, realn, per, scale1, qq, dkk, dp, dq;
        long   seed[3] = { 256, 12491249, 1499352848 };

        nn = 1001;
        Mmin = 1;
        Mmax = 1001;
        kk = seed[_section];
        
        inset= Mmin;
        span= (double)(Mmax - Mmin);
        spin= 16807;
        per= 2147483647;
        realn= (double)nn;
        scale1= 1.00001;
        qq= scale1 * (span / realn);
        dkk= (double)kk;
        
        for ( i=0 ; i<nn ; i++)
        { 
            dp= dkk*spin;
            dkk= dp - (double)((long)( dp/per)*(long)per);
            dq= dkk*span;
            ix[i] = inset + (long)( dq/ per);
            if ((ix[i] < Mmin) | (ix[i] > Mmax))
            {
                ix[i] = inset + i + 1 * (long)qq;
            }
        }
        
        return;         
      }



void checkOut(int which)
{
    int i, j;
    int errors = 0;
    char chek1[30];
    char chek2[30]; 
 
       #ifdef WATCOM
        double sumsOut[3][25] = 
        {
             { 0.0,
             5.114652693224671e+04, 1.539721811668384e+03, 1.000742883066364e+01,
             5.999250595473891e-01, 4.548871642387267e+03, 4.375116344729986e+03,
             6.104251075174761e+04, 1.501268005625795e+05, 1.189443609974981e+05,
             7.310369784325296e+04, 3.342910972650109e+07, 2.907141294167248e-05,
             1.202533961842804e+11, 3.165553044000335e+09, 3.943816690352044e+04,
             5.650760000000000e+05, 1.114641772902486e+03, 1.015727037502299e+05,
             5.421816960147207e+02, 3.040644339351238e+07, 1.597308280710200e+08,
             2.938604376566698e+02, 3.549900501563624e+04, 5.000000000000000e+02
                                                                               },
    
             { 0.0,
             5.253344778937972e+02, 1.539721811668384e+03, 1.009741436578952e+00,
             5.999250595473891e-01, 4.589031939600982e+01, 8.631675645333210e+01,
             6.345586315784055e+02, 1.501268005625795e+05, 1.189443609974981e+05,
             7.310369784325296e+04, 3.433560407475758e+04, 7.127569130821465e-06,
             9.816387810944356e+10, 3.039983465145392e+07, 3.943816690352044e+04,
             6.480410000000000e+05, 1.114641772902486e+03, 1.015727037502299e+05,
             5.421816960147207e+02, 3.126205178815432e+04, 7.824524877232093e+07,
             2.938604376566698e+02, 3.549900501563624e+04, 5.000000000000000e+01
                                                                              },
         
             { 0.0,
             3.855104502494961e+01, 3.953296986903060e+01, 2.699309089320672e-01,
             5.999250595473891e-01, 3.182615248447483e+00, 1.120309393467088e+00,
             2.845720217644024e+01, 2.960543667875005e+03, 2.623968460874250e+03,
             1.651291227698265e+03, 6.551161335845770e+02, 1.943435981130448e-06,
             3.847124199949431e+10, 2.923540598672009e+06, 1.108997288134785e+03,
             5.152160000000000e+05, 2.947368618589360e+01, 9.700646212337040e+02,
             1.268230698051004e+01, 5.987713249475302e+02, 5.009945671204667e+07,
             6.109968728263973e+00, 4.850340602749970e+02, 1.300000000000000e+01
             }
        };
       #endif
    
       #ifdef VISUALC
        double sumsOut[3][25] = 
        {
             { 0.0,
             5.114652693224671e+04, 1.539721811668385e+03, 1.000742883066364e+01,
             5.999250595473891e-01, 4.548871642387267e+03, 4.375116344729986e+03,
             6.104251075174761e+04, 1.501268005625799e+05, 1.189443609974981e+05,
             7.310369784325296e+04, 3.342910972650109e+07, 2.907141294167248e-05,
             1.202533961842805e+11, 3.165553044000335e+09, 3.943816690352044e+04,
             5.650760000000000e+05, 1.114641772902486e+03, 1.015727037502299e+05,
             5.421816960147207e+02, 3.040644339351239e+07, 1.597308280710200e+08,
             2.938604376566698e+02, 3.549900501563623e+04, 5.000000000000000e+02
                                                                               },
    
             { 0.0,
             5.253344778937972e+02, 1.539721811668385e+03, 1.009741436578952e+00,
             5.999250595473891e-01, 4.589031939600982e+01, 8.631675645333210e+01,
             6.345586315784055e+02, 1.501268005625799e+05, 1.189443609974981e+05,
             7.310369784325296e+04, 3.433560407475758e+04, 7.127569130821465e-06,
             9.816387810944356e+10, 3.039983465145392e+07, 3.943816690352044e+04,
             6.480410000000000e+05, 1.114641772902486e+03, 1.015727037502299e+05,
             5.421816960147207e+02, 3.126205178815431e+04, 7.824524877232093e+07,
             2.938604376566698e+02, 3.549900501563623e+04, 5.000000000000000e+01
                                                                              },
         
             { 0.0,
             3.855104502494961e+01, 3.953296986903060e+01, 2.699309089320672e-01,
             5.999250595473891e-01, 3.182615248447483e+00, 1.120309393467088e+00,
             2.845720217644024e+01, 2.960543667875003e+03, 2.623968460874251e+03,
             1.651291227698265e+03, 6.551161335845770e+02, 1.943435981130448e-06,
             3.847124199949431e+10, 2.923540598672009e+06, 1.108997288134785e+03,
             5.152160000000000e+05, 2.947368618589361e+01, 9.700646212337041e+02,
             1.268230698051004e+01, 5.987713249475302e+02, 5.009945671204667e+07,
             6.109968728263973e+00, 4.850340602749970e+02, 1.300000000000000e+01
             }
        };
       #endif
    
       #ifdef GCCINTEL32 
        double sumsOut[3][25] = 
        {
             { 0.0,
             5.114652693224706e+04, 1.539721811668509e+03, 1.000742883066623e+01,
             5.999250595474070e-01, 4.548871642388545e+03, 4.375116344743014e+03,
             6.104251075174961e+04, 1.501268005627157e+05, 1.189443609975086e+05,
             7.310369784325988e+04, 3.342910972650531e+07, 2.907141429123183e-05,
             1.202533961843096e+11, 3.165553044001604e+09, 3.943816690352310e+04,
             5.650760000000000e+05, 1.114641772903092e+03, 1.015727037502793e+05,
             5.421816960150400e+02, 3.040644339317274e+07, 1.597308280710857e+08,
             2.938604376567100e+02, 3.549900501566157e+04, 5.000000000000000e+02
                                                                               },
    
             { 0.0,
             5.253344778938001e+02, 1.539721811668509e+03, 1.009741436579188e+00,
             5.999250595474070e-01, 4.589031939602133e+01, 8.631675645345986e+01,
             6.345586315784150e+02, 1.501268005627157e+05, 1.189443609975086e+05,
             7.310369784325988e+04, 3.433560407476163e+04, 7.127569145018442e-06,
             9.816387817138106e+10, 3.039983465147494e+07, 3.943816690352310e+04,
             6.480410000000000e+05, 1.114641772903092e+03, 1.015727037502793e+05,
             5.421816960150400e+02, 3.126205178811007e+04, 7.824524877235141e+07,
             2.938604376567100e+02, 3.549900501566157e+04, 5.000000000000000e+01
                                                                              },     
             { 0.0,
             3.855104502494984e+01, 3.953296986903387e+01, 2.699309089321297e-01,
             5.999250595474070e-01, 3.182615248448272e+00, 1.120309393467599e+00,
             2.845720217644062e+01, 2.960543667877650e+03, 2.623968460874420e+03,
             1.651291227698377e+03, 6.551161335846538e+02, 1.943435981782704e-06,
             3.847124173932906e+10, 2.923540598699676e+06, 1.108997288135067e+03,
             5.152160000000000e+05, 2.947368618590714e+01, 9.700646212341514e+02,
             1.268230698051747e+01, 5.987713249471802e+02, 5.009945671206567e+07,
             6.109968728264795e+00, 4.850340602751676e+02, 1.300000000000000e+01
             }
        };
       #endif
    
       #ifdef CCINTEL32 
        double sumsOut[3][25] = 
        {
             { 0.0,
             5.114652693224706e+04, 1.539721811668509e+03, 1.000742883066623e+01,
             5.999250595474070e-01, 4.548871642388545e+03, 4.375116344743014e+03,
             6.104251075174961e+04, 1.501268005627157e+05, 1.189443609975086e+05,
             7.310369784325988e+04, 3.342910972650531e+07, 2.907141429123183e-05,
             1.202533961843096e+11, 3.165553044001604e+09, 3.943816690352310e+04,
             5.650760000000000e+05, 1.114641772903092e+03, 1.015727037502793e+05,
             5.421816960150400e+02, 3.040644339317274e+07, 1.597308280710857e+08,
             2.938604376567100e+02, 3.549900501566157e+04, 5.000000000000000e+02
                                                                               },
    
             { 0.0,
             5.253344778938001e+02, 1.539721811668509e+03, 1.009741436579188e+00,
             5.999250595474070e-01, 4.589031939602133e+01, 8.631675645345986e+01,
             6.345586315784150e+02, 1.501268005627157e+05, 1.189443609975086e+05,
             7.310369784325988e+04, 3.433560407476163e+04, 7.127569145018442e-06,
             9.816387817138104e+10, 3.039983465147494e+07, 3.943816690352310e+04,
             6.480410000000000e+05, 1.114641772903092e+03, 1.015727037502793e+05,
             5.421816960150400e+02, 3.126205178811007e+04, 7.824524877235141e+07,
             2.938604376567100e+02, 3.549900501566157e+04, 5.000000000000000e+01
                                                                              },     
             { 0.0,
             3.855104502494984e+01, 3.953296986903387e+01, 2.699309089321297e-01,
             5.999250595474070e-01, 3.182615248448272e+00, 1.120309393467599e+00,
             2.845720217644062e+01, 2.960543667877650e+03, 2.623968460874420e+03,
             1.651291227698377e+03, 6.551161335846538e+02, 1.943435981782704e-06,
             3.847124173932906e+10, 2.923540598699676e+06, 1.108997288135067e+03,
             5.152160000000000e+05, 2.947368618590714e+01, 9.700646212341514e+02,
             1.268230698051747e+01, 5.987713249471802e+02, 5.009945671206567e+07,
             6.109968728264795e+00, 4.850340602751676e+02, 1.300000000000000e+01
             }
        };
       #endif
    
       #ifdef GCCINTEL64
        double sumsOut[3][25] = 
        {
             { 0.0,
             5.114652693224671e+04, 1.539721811668385e+03, 1.000742883066363e+01,
             5.999250595473891e-01, 4.548871642387267e+03, 4.375116344729986e+03,
             6.104251075174761e+04, 1.501268005625795e+05, 1.189443609974981e+05,
             7.310369784325296e+04, 3.342910972650109e+07, 2.907141294167248e-05,
             1.202533961842805e+11, 3.165553044000335e+09, 3.943816690352044e+04,
             5.650760000000000e+05, 1.114641772902486e+03, 1.015727037502299e+05,
             5.421816960147207e+02, 3.040644339351239e+07, 1.597308280710199e+08,
             2.938604376566697e+02, 3.549900501563623e+04, 5.000000000000000e+02
                                                                               },
    
             { 0.0,
             5.253344778937972e+02, 1.539721811668385e+03, 1.009741436578952e+00,
             5.999250595473891e-01, 4.589031939600982e+01, 8.631675645333210e+01,
             6.345586315784055e+02, 1.501268005625795e+05, 1.189443609974981e+05,
             7.310369784325296e+04, 3.433560407475758e+04, 7.127569130821465e-06,
             9.816387810944356e+10, 3.039983465145392e+07, 3.943816690352044e+04,
    
             6.480410000000000e+05, 1.114641772902486e+03, 1.015727037502299e+05,
             5.421816960147207e+02, 3.126205178815431e+04, 7.824524877232093e+07,
             2.938604376566697e+02, 3.549900501563623e+04, 5.000000000000000e+01
                                                                              },
         
             { 0.0,
             3.855104502494961e+01, 3.953296986903059e+01, 2.699309089320672e-01,
             5.999250595473891e-01, 3.182615248447483e+00, 1.120309393467088e+00,
             2.845720217644024e+01, 2.960543667875005e+03, 2.623968460874250e+03,
             1.651291227698265e+03, 6.551161335845770e+02, 1.943435981130448e-06,
             3.847124199949431e+10, 2.923540598672009e+06, 1.108997288134785e+03,
             5.152160000000000e+05, 2.947368618589361e+01, 9.700646212337041e+02,
             1.268230698051003e+01, 5.987713249475302e+02, 5.009945671204667e+07,
             6.109968728263972e+00, 4.850340602749970e+02, 1.300000000000000e+01
             }
        };
       #endif
    
       #ifdef GCCARMDP
        double sumsOut[3][25] = 
        {
             { 0.0,
             5.114652693224706e+04, 1.539721811668509e+03, 1.000742883066623e+01,
             5.999250595474070e-01, 4.548871642388545e+03, 4.375116344743014e+03,
             6.104251075174961e+04, 1.501268005627157e+05, 1.189443609975086e+05,
             7.310369784325988e+04, 3.342910972650531e+07, 2.907141429123183e-05,
             1.202533961843096e+11, 3.165553044001604e+09, 3.943816690352310e+04,
             5.650760000000000e+05, 1.114641772903092e+03, 1.015727037502793e+05,
             5.421816960150400e+02, 3.040644339317274e+07, 1.597308280710857e+08,
             2.938604376567100e+02, 3.549900501566157e+04, 5.000000000000000e+02
                                                                               },
             { 0.0,
             5.253344778938001e+02, 1.539721811668509e+03, 1.009741436579188e+00,
             5.999250595474070e-01, 4.589031939602133e+01, 8.631675645345986e+01,
             6.345586315784150e+02, 1.501268005627157e+05, 1.189443609975086e+05,
             7.310369784325988e+04, 3.433560407476163e+04, 7.127569145018442e-06,
             9.816387817138104e+10, 3.039983465147494e+07, 3.943816690352310e+04,
             6.480410000000000e+05, 1.114641772903092e+03, 1.015727037502793e+05,
             5.421816960150400e+02, 3.126205178811007e+04, 7.824524877235141e+07,
             2.938604376567100e+02, 3.549900501566157e+04, 5.000000000000000e+01
                                                                              },     
             { 0.0,
             3.855104502494984e+01, 3.953296986903387e+01, 2.699309089321297e-01,
             5.999250595474070e-01, 3.182615248448272e+00, 1.120309393467599e+00,
             2.845720217644062e+01, 2.960543667877650e+03, 2.623968460874420e+03,
             1.651291227698377e+03, 6.551161335846538e+02, 1.943435981782704e-06,
             3.847124173932906e+10, 2.923540598699676e+06, 1.108997288135067e+03,
             5.152160000000000e+05, 2.947368618590714e+01, 9.700646212341514e+02,
             1.268230698051747e+01, 5.987713249471802e+02, 5.009945671206567e+07,
             6.109968728264795e+00, 4.850340602751676e+02, 1.300000000000000e+01
             }
        };
       #endif 

       #ifdef GCCARMPI
        double sumsOut[3][25] = 
        {
             { 0.0,
             5.114652693224671e+04, 1.539721811668385e+03, 1.000742883066363e+01,
             5.999250595473891e-01, 4.548871642387267e+03, 4.375116344729986e+03,
             6.104251075174761e+04, 1.501268005625795e+05, 1.189443609974981e+05,
             7.310369784325296e+04, 3.342910972650109e+07, 2.907141294167248e-05,
             1.202533961842805e+11, 3.165553044000335e+09, 3.943816690352044e+04,
             5.650760000000000e+05, 1.114641772902486e+03, 1.015727037502299e+05,
             5.421816960147207e+02, 3.040644339351239e+07, 1.597308280710199e+08,
             2.938604376566697e+02, 3.549900501563623e+04, 5.000000000000000e+02
                                                                               },
             { 0.0,
             5.253344778937972e+02, 1.539721811668385e+03, 1.009741436578952e+00,
             5.999250595473891e-01, 4.589031939600982e+01, 8.631675645333210e+01,
             6.345586315784055e+02, 1.501268005625795e+05, 1.189443609974981e+05,
             7.310369784325296e+04, 3.433560407475758e+04, 7.127569130821465e-06,
             9.816387810944356e+10, 3.039983465145392e+07, 3.943816690352044e+04,
             6.480410000000000e+05, 1.114641772902486e+03, 1.015727037502299e+05,
             5.421816960147207e+02, 3.126205178815431e+04, 7.824524877232093e+07,
             2.938604376566697e+02, 3.549900501563623e+04, 5.000000000000000e+01
                                                                               },     
             { 0.0,
             3.855104502494961e+01, 3.953296986903059e+01, 2.699309089320672e-01,
             5.999250595473891e-01, 3.182615248447483e+00, 1.120309393467088e+00,
             2.845720217644024e+01, 2.960543667875005e+03, 2.623968460874250e+03,
             1.651291227698265e+03, 6.551161335845770e+02, 1.943435981130448e-06,
             3.847124199949431e+10, 2.923540598672009e+06, 1.108997288134785e+03,
             5.152160000000000e+05, 2.947368618589361e+01, 9.700646212337041e+02,
             1.268230698051003e+01, 5.987713249475302e+02, 5.009945671204667e+07,
             6.109968728263972e+00, 4.850340602749970e+02, 1.300000000000000e+01
             }
        };
       #endif 

    if (reliability)
    {
        i = (int)_section;
        j = which;
        if (count2 == 1)
        {
            failCount = 0;
            sumscomp[i][j] = sumsOut[i][j];   
            sprintf(chek1, "%22.10e", Checksum[i][j]);
            sprintf(chek2, "%22.10e", sumscomp[i][j]);
            if (!withinErrorThreshold(sumscomp[i][j], Checksum[i][j], FP_ERROR_THRESHOLD))
            {
                nsRes = TRUE;
                sumscomp[i][j] = Checksum[i][j];
                fprintf(outfile, " Section %d Test %2d pass %5ld Non-standard result was %s expected %s. The percent error was %e\n",
                                    i+1, j, count2, chek1, chek2, 100*relative_error(Checksum[i][j], sumscomp[i][j]));
            }
        }
        else
        {
            sprintf(chek1, "%22.10e", Checksum[i][j]);
            sprintf(chek2, "%22.10e", sumscomp[i][j]);
            if (!withinErrorThreshold(sumscomp[i][j], Checksum[i][j], FP_ERROR_THRESHOLD))
            {
                compareFail = compareFail + 1;
                failCount = failCount + 1;
                if (compareFail == 1)
                {
                    fprintf(outfile, " ERRORS - maximum of 5 reported per loop\n");
                }
                if (failCount < 100)
                {
                    fprintf(outfile, " Section %d Test %2d pass %5ld Different result was %s expected %s. Percent error was %e\n",
                                       i+1, j, count2, chek1, chek2,100*relative_error(Checksum[i][j], sumscomp[i][j]));
                    fflush(outfile);
                }
            }
        }
    }
    else
    {
        for (i=0; i<3; i++)
        {
            for (j=1; j<25; j++)
            {
                sprintf(chek1, "%22.10e", Checksum[i][j]);
                sprintf(chek2, "%22.10e", sumsOut[i][j]);
                if (!withinErrorThreshold(sumscomp[i][j], Checksum[i][j], FP_ERROR_THRESHOLD))
                {
                    errors = errors + 1;
                    fprintf(outfile, " Section %d Test %2d Non-standard result was %s expected %s. The percent error was %e\n",
                                        i+1, j, chek1, chek2, 100*relative_error(Checksum[i][j], sumscomp[i][j]));
                }
            }
        }
        if (errors == 0)
        {
            fprintf(outfile, " Numeric results were as expected\n");
        }
        else
        {
            fprintf(outfile, "\n Compiler #define or values in linpack.c might to be changed\n");
        }
        fprintf(outfile, "\n");
    }   
}    
