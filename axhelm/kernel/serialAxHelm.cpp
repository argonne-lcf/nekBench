#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <occa.hpp>

using namespace std;
using namespace occa;
/*
The MIT License (MIT)
Copyright (c) 2017 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

// default to element-per-threadblock

extern "C" void axhelm_v0(const long int & Nelements,
                          const long int & offset,
                          const double * __restrict__ ggeo,
                          const double * __restrict__ D,
                          const double * __restrict__ lambda,
                          const double * __restrict__ q,
                          double * __restrict__ Aq) {
  for (long int e = 0; e < Nelements; ++e) {
    int _occa_exclusive_index;
    double s_D[4][4];
    double s_q[4][4];
    double s_Gqr[4][4];
    double s_Gqs[4][4];
    double r_qt[256], r_Gqt[256], r_Auk[256];
    double r_q[256][4];
    // register array to hold u(i,j,0:N) private to thread
    double r_Aq[256][4];
    // array for results Au(i,j,0:N)

    double r_G00[256], r_G01[256], r_G02[256], r_G11[256], r_G12[256], r_G22[256], r_GwJ[256];
    double r_lam0[256], r_lam1[256];

    // array of threads
    _occa_exclusive_index = 0;
    for (int j = 0; j < 4; ++j) {
      for (int i = 0; i < 4; ++i) {
        //load D into local memory
        // s_D[i][j] = d \phi_i at node j
        s_D[j][i] = D[4 * j + i];
        // D is column major

        // load pencil of u into register
        const long int base = i + j * 4 + e * p_Np;
#pragma unroll 4
        for (int k = 0; k < 4; ++k) {
          r_q[_occa_exclusive_index][k] = q[base + k * 4 * 4];
          // prefetch operation
          r_Aq[_occa_exclusive_index][k] = 0.f;
          // zero the accumulator
        }
        ++_occa_exclusive_index;
      }
    }

    // Layer by layer
#pragma unroll 4
    for (int k = 0; k < 4; ++k) {
      _occa_exclusive_index = 0;
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          const long int id = e * p_Np + k * 4 * 4 + j * 4 + i;

          // prefetch geometric factors
          const long int gbase = e * p_Nggeo * p_Np + k * 4 * 4 + j * 4 + i;
          r_G00[_occa_exclusive_index] = ggeo[gbase + p_G00ID * p_Np];
          r_G01[_occa_exclusive_index] = ggeo[gbase + p_G01ID * p_Np];
          r_G02[_occa_exclusive_index] = ggeo[gbase + p_G02ID * p_Np];
          r_G11[_occa_exclusive_index] = ggeo[gbase + p_G11ID * p_Np];
          r_G12[_occa_exclusive_index] = ggeo[gbase + p_G12ID * p_Np];
          r_G22[_occa_exclusive_index] = ggeo[gbase + p_G22ID * p_Np];
          r_GwJ[_occa_exclusive_index] = ggeo[gbase + p_GWJID * p_Np];
          r_lam0[_occa_exclusive_index] = lambda[id + 0 * offset];
          r_lam1[_occa_exclusive_index] = lambda[id + 1 * offset];
          ++_occa_exclusive_index;
        }
      }

      //@barrier("local");

      _occa_exclusive_index = 0;
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          // share u(:,:,k)
          s_q[j][i] = r_q[_occa_exclusive_index][k];
          r_qt[_occa_exclusive_index] = 0;
#pragma unroll 4
          for (int m = 0; m < 4; ++m) {
            r_qt[_occa_exclusive_index] += s_D[k][m] * r_q[_occa_exclusive_index][m];
          }
          ++_occa_exclusive_index;
        }
      }

      //@barrier("local");

      _occa_exclusive_index = 0;
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          double qr = 0.f;
          double qs = 0.f;
#pragma unroll 4
          for (int m = 0; m < 4; ++m) {
            qr += s_D[i][m] * s_q[j][m];
            qs += s_D[j][m] * s_q[m][i];
          }
          s_Gqs[j][i] = r_lam0[_occa_exclusive_index] * (r_G01[_occa_exclusive_index] * qr + r_G11[_occa_exclusive_index] * qs + r_G12[_occa_exclusive_index] * r_qt[_occa_exclusive_index]);
          s_Gqr[j][i] = r_lam0[_occa_exclusive_index] * (r_G00[_occa_exclusive_index] * qr + r_G01[_occa_exclusive_index] * qs + r_G02[_occa_exclusive_index] * r_qt[_occa_exclusive_index]);

          // put this here for a performance bump
          r_Gqt[_occa_exclusive_index] = r_lam0[_occa_exclusive_index] * (r_G02[_occa_exclusive_index] * qr + r_G12[_occa_exclusive_index] * qs + r_G22[_occa_exclusive_index] * r_qt[_occa_exclusive_index]);
          r_Auk[_occa_exclusive_index] = r_GwJ[_occa_exclusive_index] * r_lam1[_occa_exclusive_index] * r_q[_occa_exclusive_index][k];
          ++_occa_exclusive_index;
        }
      }

      //@barrier("local");

      _occa_exclusive_index = 0;
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
#pragma unroll 4
          for (int m = 0; m < 4; ++m) {
            r_Auk[_occa_exclusive_index] += s_D[m][j] * s_Gqs[m][i];
            r_Aq[_occa_exclusive_index][m] += s_D[k][m] * r_Gqt[_occa_exclusive_index];
            // DT(m,k)*ut(i,j,k,e)
            r_Auk[_occa_exclusive_index] += s_D[m][i] * s_Gqr[j][m];
          }
          r_Aq[_occa_exclusive_index][k] += r_Auk[_occa_exclusive_index];
          ++_occa_exclusive_index;
        }
      }
    }

    // write out

    _occa_exclusive_index = 0;
    for (int j = 0; j < 4; ++j) {
      for (int i = 0; i < 4; ++i) {
#pragma unroll 4
        for (int k = 0; k < 4; ++k) {
          const long int id = e * p_Np + k * 4 * 4 + j * 4 + i;
          Aq[id] = r_Aq[_occa_exclusive_index][k];
        }
        ++_occa_exclusive_index;
      }
    }
  }
}

extern "C" void axhelmPartial_v0(const long int & Nelements,
                                 const long int & offset,
                                 const long int * __restrict__ elementList,
                                 const double * __restrict__ ggeo,
                                 const double * __restrict__ D,
                                 const double * __restrict__ lambda,
                                 const double * __restrict__ q,
                                 double * __restrict__ Aq) {
  for (long int e = 0; e < Nelements; ++e) {
    int _occa_exclusive_index;
    double s_D[4][4];
    double s_q[4][4];
    double s_Gqr[4][4];
    double s_Gqs[4][4];
    double r_qt[256], r_Gqt[256], r_Auk[256];
    double r_q[256][4];
    // register array to hold u(i,j,0:N) private to thread
    double r_Aq[256][4];
    // array for results Au(i,j,0:N)

    long int element[256];
    double r_G00[256], r_G01[256], r_G02[256], r_G11[256], r_G12[256], r_G22[256], r_GwJ[256];
    double r_lam0[256], r_lam1[256];

    // array of threads
    _occa_exclusive_index = 0;
    for (int j = 0; j < 4; ++j) {
      for (int i = 0; i < 4; ++i) {
        //load D into local memory
        // s_D[i][j] = d \phi_i at node j
        s_D[j][i] = D[4 * j + i];
        // D is column major
        element[_occa_exclusive_index] = elementList[e];

        // load pencil of u into register
        const long int base = i + j * 4 + element[_occa_exclusive_index] * p_Np;
#pragma unroll 4
        for (int k = 0; k < 4; ++k) {
          r_q[_occa_exclusive_index][k] = q[base + k * 4 * 4];
          // prefetch operation
          r_Aq[_occa_exclusive_index][k] = 0.f;
          // zero the accumulator
        }
        ++_occa_exclusive_index;
      }
    }

    // Layer by layer
#pragma unroll 4
    for (int k = 0; k < 4; ++k) {
      _occa_exclusive_index = 0;
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          const long int id = element[_occa_exclusive_index] * p_Np + k * 4 * 4 + j * 4 + i;

          // prefetch geometric factors
          const long int gbase = element[_occa_exclusive_index] * p_Nggeo * p_Np + k * 4 * 4 + j * 4 + i;
          r_G00[_occa_exclusive_index] = ggeo[gbase + p_G00ID * p_Np];
          r_G01[_occa_exclusive_index] = ggeo[gbase + p_G01ID * p_Np];
          r_G02[_occa_exclusive_index] = ggeo[gbase + p_G02ID * p_Np];
          r_G11[_occa_exclusive_index] = ggeo[gbase + p_G11ID * p_Np];
          r_G12[_occa_exclusive_index] = ggeo[gbase + p_G12ID * p_Np];
          r_G22[_occa_exclusive_index] = ggeo[gbase + p_G22ID * p_Np];
          r_GwJ[_occa_exclusive_index] = ggeo[gbase + p_GWJID * p_Np];
          r_lam0[_occa_exclusive_index] = lambda[id + 0 * offset];
          r_lam1[_occa_exclusive_index] = lambda[id + 1 * offset];
          ++_occa_exclusive_index;
        }
      }

      //@barrier("local");

      _occa_exclusive_index = 0;
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          // share u(:,:,k)
          s_q[j][i] = r_q[_occa_exclusive_index][k];
          r_qt[_occa_exclusive_index] = 0;
#pragma unroll 4
          for (int m = 0; m < 4; ++m) {
            r_qt[_occa_exclusive_index] += s_D[k][m] * r_q[_occa_exclusive_index][m];
          }
          ++_occa_exclusive_index;
        }
      }

      //@barrier("local");

      _occa_exclusive_index = 0;
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          double qr = 0.f;
          double qs = 0.f;
#pragma unroll 4
          for (int m = 0; m < 4; ++m) {
            qr += s_D[i][m] * s_q[j][m];
            qs += s_D[j][m] * s_q[m][i];
          }
          s_Gqs[j][i] = r_lam0[_occa_exclusive_index] * (r_G01[_occa_exclusive_index] * qr + r_G11[_occa_exclusive_index] * qs + r_G12[_occa_exclusive_index] * r_qt[_occa_exclusive_index]);
          s_Gqr[j][i] = r_lam0[_occa_exclusive_index] * (r_G00[_occa_exclusive_index] * qr + r_G01[_occa_exclusive_index] * qs + r_G02[_occa_exclusive_index] * r_qt[_occa_exclusive_index]);

          // put this here for a performance bump
          r_Gqt[_occa_exclusive_index] = r_lam0[_occa_exclusive_index] * (r_G02[_occa_exclusive_index] * qr + r_G12[_occa_exclusive_index] * qs + r_G22[_occa_exclusive_index] * r_qt[_occa_exclusive_index]);
          r_Auk[_occa_exclusive_index] = r_GwJ[_occa_exclusive_index] * r_lam1[_occa_exclusive_index] * r_q[_occa_exclusive_index][k];
          ++_occa_exclusive_index;
        }
      }

      //@barrier("local");

      _occa_exclusive_index = 0;
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
#pragma unroll 4
          for (int m = 0; m < 4; ++m) {
            r_Auk[_occa_exclusive_index] += s_D[m][j] * s_Gqs[m][i];
            r_Aq[_occa_exclusive_index][m] += s_D[k][m] * r_Gqt[_occa_exclusive_index];
            // DT(m,k)*ut(i,j,k,e)
            r_Auk[_occa_exclusive_index] += s_D[m][i] * s_Gqr[j][m];
          }
          r_Aq[_occa_exclusive_index][k] += r_Auk[_occa_exclusive_index];
          ++_occa_exclusive_index;
        }
      }
    }

    // write out

    _occa_exclusive_index = 0;
    for (int j = 0; j < 4; ++j) {
      for (int i = 0; i < 4; ++i) {
#pragma unroll 4
        for (int k = 0; k < 4; ++k) {
          const long int id = element[_occa_exclusive_index] * p_Np + k * 4 * 4 + j * 4 + i;
          Aq[id] = r_Aq[_occa_exclusive_index][k];
        }
        ++_occa_exclusive_index;
      }
    }
  }
}

extern "C" void axhelmPartial_bk_n3_v0(const long int & Nelements,
                                       const long int & offset,
                                       const long int * __restrict__ elementList,
                                       const double * __restrict__ ggeo,
                                       const double * __restrict__ D,
                                       const double * __restrict__ lambda,
                                       const double * __restrict__ q,
                                       double * __restrict__ Aq) {
  for (long int e = 0; e < Nelements; ++e) {
    int _occa_exclusive_index;
    double s_D[4][4];
    double s_U[4][4];
    double s_V[4][4];
    double s_W[4][4];
    double s_GUr[4][4];
    double s_GUs[4][4];
    double s_GVr[4][4];
    double s_GVs[4][4];
    double s_GWr[4][4];
    double s_GWs[4][4];
    double r_Ut[256], r_Vt[256], r_Wt[256];
    double r_U[256][4], r_V[256][4], r_W[256][4];
    double r_AU[256][4], r_AV[256][4], r_AW[256][4];
    long int element[256];
    double r_G00[256], r_G01[256], r_G02[256], r_G11[256], r_G12[256], r_G22[256], r_GwJ[256];

    // array of threads
    _occa_exclusive_index = 0;
    for (int j = 0; j < 4; ++j) {
      for (int i = 0; i < 4; ++i) {
        //load D into local memory
        // s_D[i][j] = d \phi_i at node j
        s_D[j][i] = D[4 * j + i];
        // D is column major

        element[_occa_exclusive_index] = elementList[e];

        // load pencil of u into register
        const long int base = i + j * 4 + element[_occa_exclusive_index] * p_Np;
#pragma unroll 4
        for (int k = 0; k < 4; k++) {
          //
          r_U[_occa_exclusive_index][k] = q[base + k * 4 * 4 + 0 * offset];
          r_V[_occa_exclusive_index][k] = q[base + k * 4 * 4 + 1 * offset];
          r_W[_occa_exclusive_index][k] = q[base + k * 4 * 4 + 2 * offset];
          //
          r_AU[_occa_exclusive_index][k] = 0.f;
          r_AV[_occa_exclusive_index][k] = 0.f;
          r_AW[_occa_exclusive_index][k] = 0.f;
        }
        ++_occa_exclusive_index;
      }
    }

    // Layer by layer
#pragma unroll 4
    for (int k = 0; k < 4; ++k) {
      _occa_exclusive_index = 0;
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          const long int id = element[_occa_exclusive_index] * p_Np + k * 4 * 4 + j * 4 + i;

          // prefetch geometric factors
          const long int gbase = element[_occa_exclusive_index] * p_Nggeo * p_Np + k * 4 * 4 + j * 4 + i;
          r_G00[_occa_exclusive_index] = ggeo[gbase + p_G00ID * p_Np];
          r_G01[_occa_exclusive_index] = ggeo[gbase + p_G01ID * p_Np];
          r_G02[_occa_exclusive_index] = ggeo[gbase + p_G02ID * p_Np];
          r_G11[_occa_exclusive_index] = ggeo[gbase + p_G11ID * p_Np];
          r_G12[_occa_exclusive_index] = ggeo[gbase + p_G12ID * p_Np];
          r_G22[_occa_exclusive_index] = ggeo[gbase + p_G22ID * p_Np];
          r_GwJ[_occa_exclusive_index] = ggeo[gbase + p_GWJID * p_Np];
          ++_occa_exclusive_index;
        }
      }

      //@barrier("local");

      _occa_exclusive_index = 0;
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          // share u(:,:,k)
          s_U[j][i] = r_U[_occa_exclusive_index][k];
          s_V[j][i] = r_V[_occa_exclusive_index][k];
          s_W[j][i] = r_W[_occa_exclusive_index][k];
          r_Ut[_occa_exclusive_index] = 0;
          r_Vt[_occa_exclusive_index] = 0;
          r_Wt[_occa_exclusive_index] = 0;
#pragma unroll 4
          for (int m = 0; m < 4; m++) {
            double Dkm = s_D[k][m];
            r_Ut[_occa_exclusive_index] += Dkm * r_U[_occa_exclusive_index][m];
            r_Vt[_occa_exclusive_index] += Dkm * r_V[_occa_exclusive_index][m];
            r_Wt[_occa_exclusive_index] += Dkm * r_W[_occa_exclusive_index][m];
          }
          ++_occa_exclusive_index;
        }
      }

      //@barrier("local");

      _occa_exclusive_index = 0;
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          double Ur = 0.f, Us = 0.f;
          double Vr = 0.f, Vs = 0.f;
          double Wr = 0.f, Ws = 0.f;
#pragma unroll 4
          for (int m = 0; m < 4; m++) {
            double Dim = s_D[i][m];
            double Djm = s_D[j][m];
            Ur += Dim * s_U[j][m];
            Us += Djm * s_U[m][i];
            Vr += Dim * s_V[j][m];
            Vs += Djm * s_V[m][i];
            Wr += Dim * s_W[j][m];
            Ws += Djm * s_W[m][i];
          }
          s_GUr[j][i] = (r_G00[_occa_exclusive_index] * Ur + r_G01[_occa_exclusive_index] * Us + r_G02[_occa_exclusive_index] * r_Ut[_occa_exclusive_index]);
          s_GVr[j][i] = (r_G00[_occa_exclusive_index] * Vr + r_G01[_occa_exclusive_index] * Vs + r_G02[_occa_exclusive_index] * r_Vt[_occa_exclusive_index]);
          s_GWr[j][i] = (r_G00[_occa_exclusive_index] * Wr + r_G01[_occa_exclusive_index] * Ws + r_G02[_occa_exclusive_index] * r_Wt[_occa_exclusive_index]);
          s_GUs[j][i] = (r_G01[_occa_exclusive_index] * Ur + r_G11[_occa_exclusive_index] * Us + r_G12[_occa_exclusive_index] * r_Ut[_occa_exclusive_index]);
          s_GVs[j][i] = (r_G01[_occa_exclusive_index] * Vr + r_G11[_occa_exclusive_index] * Vs + r_G12[_occa_exclusive_index] * r_Vt[_occa_exclusive_index]);
          s_GWs[j][i] = (r_G01[_occa_exclusive_index] * Wr + r_G11[_occa_exclusive_index] * Ws + r_G12[_occa_exclusive_index] * r_Wt[_occa_exclusive_index]);
          r_Ut[_occa_exclusive_index] = (r_G02[_occa_exclusive_index] * Ur + r_G12[_occa_exclusive_index] * Us + r_G22[_occa_exclusive_index] * r_Ut[_occa_exclusive_index]);
          r_Vt[_occa_exclusive_index] = (r_G02[_occa_exclusive_index] * Vr + r_G12[_occa_exclusive_index] * Vs + r_G22[_occa_exclusive_index] * r_Vt[_occa_exclusive_index]);
          r_Wt[_occa_exclusive_index] = (r_G02[_occa_exclusive_index] * Wr + r_G12[_occa_exclusive_index] * Ws + r_G22[_occa_exclusive_index] * r_Wt[_occa_exclusive_index]);
          ++_occa_exclusive_index;
        }
      }

      //@barrier("local");

      _occa_exclusive_index = 0;
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          double AUtmp = 0, AVtmp = 0, AWtmp = 0;
#pragma unroll 4
          for (int m = 0; m < 4; m++) {
            // 9 shared, 18 flops => 12TB/s*18/(9*8) = 3TFLOPS/s
            double Dmi = s_D[m][i];
            double Dmj = s_D[m][j];
            double Dkm = s_D[k][m];
            AUtmp += Dmi * s_GUr[j][m];
            AUtmp += Dmj * s_GUs[m][i];
            AVtmp += Dmi * s_GVr[j][m];
            AVtmp += Dmj * s_GVs[m][i];
            AWtmp += Dmi * s_GWr[j][m];
            AWtmp += Dmj * s_GWs[m][i];
            r_AU[_occa_exclusive_index][m] += Dkm * r_Ut[_occa_exclusive_index];
            r_AV[_occa_exclusive_index][m] += Dkm * r_Vt[_occa_exclusive_index];
            r_AW[_occa_exclusive_index][m] += Dkm * r_Wt[_occa_exclusive_index];
          }
          r_AU[_occa_exclusive_index][k] += AUtmp;
          r_AV[_occa_exclusive_index][k] += AVtmp;
          r_AW[_occa_exclusive_index][k] += AWtmp;
          ++_occa_exclusive_index;
        }
      }
    }

    // write out

    _occa_exclusive_index = 0;
    for (int j = 0; j < 4; ++j) {
      for (int i = 0; i < 4; ++i) {
#pragma unroll 4
        for (int k = 0; k < 4; k++) {
          const long int id = element[_occa_exclusive_index] * p_Np + k * 4 * 4 + j * 4 + i;
          Aq[id + 0 * offset] = r_AU[_occa_exclusive_index][k];
          Aq[id + 1 * offset] = r_AV[_occa_exclusive_index][k];
          Aq[id + 2 * offset] = r_AW[_occa_exclusive_index][k];
        }
        ++_occa_exclusive_index;
      }
    }
  }
}

extern "C" void axhelm_bk_n3_v0(const long int & Nelements,
                                const long int & offset,
                                const double * __restrict__ ggeo,
                                const double * __restrict__ D,
                                const double * __restrict__ lambda,
                                const double * __restrict__ q,
                                double * __restrict__ Aq) {
  for (long int e = 0; e < Nelements; ++e) {
    int _occa_exclusive_index;
    double s_D[4][4];
    double s_U[4][4];
    double s_V[4][4];
    double s_W[4][4];
    double s_GUr[4][4];
    double s_GUs[4][4];
    double s_GVr[4][4];
    double s_GVs[4][4];
    double s_GWr[4][4];
    double s_GWs[4][4];
    double r_Ut[256], r_Vt[256], r_Wt[256];
    double r_U[256][4], r_V[256][4], r_W[256][4];
    double r_AU[256][4], r_AV[256][4], r_AW[256][4];
    double r_G00[256], r_G01[256], r_G02[256], r_G11[256], r_G12[256], r_G22[256], r_GwJ[256];

    // array of threads
    _occa_exclusive_index = 0;
    for (int j = 0; j < 4; ++j) {
      for (int i = 0; i < 4; ++i) {
        //load D into local memory
        // s_D[i][j] = d \phi_i at node j
        s_D[j][i] = D[4 * j + i];
        // D is column major

        // load pencil of u into register
        const long int base = i + j * 4 + e * p_Np;
#pragma unroll 4
        for (int k = 0; k < 4; k++) {
          //
          r_U[_occa_exclusive_index][k] = q[base + k * 4 * 4 + 0 * offset];
          r_V[_occa_exclusive_index][k] = q[base + k * 4 * 4 + 1 * offset];
          r_W[_occa_exclusive_index][k] = q[base + k * 4 * 4 + 2 * offset];
          //
          r_AU[_occa_exclusive_index][k] = 0.f;
          r_AV[_occa_exclusive_index][k] = 0.f;
          r_AW[_occa_exclusive_index][k] = 0.f;
        }
        ++_occa_exclusive_index;
      }
    }

    // Layer by layer
#pragma unroll 4
    for (int k = 0; k < 4; ++k) {
      _occa_exclusive_index = 0;
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          const long int id = e * p_Np + k * 4 * 4 + j * 4 + i;

          // prefetch geometric factors
          const long int gbase = e * p_Nggeo * p_Np + k * 4 * 4 + j * 4 + i;
          r_G00[_occa_exclusive_index] = ggeo[gbase + p_G00ID * p_Np];
          r_G01[_occa_exclusive_index] = ggeo[gbase + p_G01ID * p_Np];
          r_G02[_occa_exclusive_index] = ggeo[gbase + p_G02ID * p_Np];
          r_G11[_occa_exclusive_index] = ggeo[gbase + p_G11ID * p_Np];
          r_G12[_occa_exclusive_index] = ggeo[gbase + p_G12ID * p_Np];
          r_G22[_occa_exclusive_index] = ggeo[gbase + p_G22ID * p_Np];
          r_GwJ[_occa_exclusive_index] = ggeo[gbase + p_GWJID * p_Np];
          ++_occa_exclusive_index;
        }
      }

      //@barrier("local");

      _occa_exclusive_index = 0;
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          // share u(:,:,k)
          s_U[j][i] = r_U[_occa_exclusive_index][k];
          s_V[j][i] = r_V[_occa_exclusive_index][k];
          s_W[j][i] = r_W[_occa_exclusive_index][k];
          r_Ut[_occa_exclusive_index] = 0;
          r_Vt[_occa_exclusive_index] = 0;
          r_Wt[_occa_exclusive_index] = 0;
#pragma unroll 4
          for (int m = 0; m < 4; m++) {
            double Dkm = s_D[k][m];
            r_Ut[_occa_exclusive_index] += Dkm * r_U[_occa_exclusive_index][m];
            r_Vt[_occa_exclusive_index] += Dkm * r_V[_occa_exclusive_index][m];
            r_Wt[_occa_exclusive_index] += Dkm * r_W[_occa_exclusive_index][m];
          }
          ++_occa_exclusive_index;
        }
      }

      //@barrier("local");

      _occa_exclusive_index = 0;
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          double Ur = 0.f, Us = 0.f;
          double Vr = 0.f, Vs = 0.f;
          double Wr = 0.f, Ws = 0.f;
#pragma unroll 4
          for (int m = 0; m < 4; m++) {
            double Dim = s_D[i][m];
            double Djm = s_D[j][m];
            Ur += Dim * s_U[j][m];
            Us += Djm * s_U[m][i];
            Vr += Dim * s_V[j][m];
            Vs += Djm * s_V[m][i];
            Wr += Dim * s_W[j][m];
            Ws += Djm * s_W[m][i];
          }
          s_GUr[j][i] = (r_G00[_occa_exclusive_index] * Ur + r_G01[_occa_exclusive_index] * Us + r_G02[_occa_exclusive_index] * r_Ut[_occa_exclusive_index]);
          s_GVr[j][i] = (r_G00[_occa_exclusive_index] * Vr + r_G01[_occa_exclusive_index] * Vs + r_G02[_occa_exclusive_index] * r_Vt[_occa_exclusive_index]);
          s_GWr[j][i] = (r_G00[_occa_exclusive_index] * Wr + r_G01[_occa_exclusive_index] * Ws + r_G02[_occa_exclusive_index] * r_Wt[_occa_exclusive_index]);
          s_GUs[j][i] = (r_G01[_occa_exclusive_index] * Ur + r_G11[_occa_exclusive_index] * Us + r_G12[_occa_exclusive_index] * r_Ut[_occa_exclusive_index]);
          s_GVs[j][i] = (r_G01[_occa_exclusive_index] * Vr + r_G11[_occa_exclusive_index] * Vs + r_G12[_occa_exclusive_index] * r_Vt[_occa_exclusive_index]);
          s_GWs[j][i] = (r_G01[_occa_exclusive_index] * Wr + r_G11[_occa_exclusive_index] * Ws + r_G12[_occa_exclusive_index] * r_Wt[_occa_exclusive_index]);
          r_Ut[_occa_exclusive_index] = (r_G02[_occa_exclusive_index] * Ur + r_G12[_occa_exclusive_index] * Us + r_G22[_occa_exclusive_index] * r_Ut[_occa_exclusive_index]);
          r_Vt[_occa_exclusive_index] = (r_G02[_occa_exclusive_index] * Vr + r_G12[_occa_exclusive_index] * Vs + r_G22[_occa_exclusive_index] * r_Vt[_occa_exclusive_index]);
          r_Wt[_occa_exclusive_index] = (r_G02[_occa_exclusive_index] * Wr + r_G12[_occa_exclusive_index] * Ws + r_G22[_occa_exclusive_index] * r_Wt[_occa_exclusive_index]);
          ++_occa_exclusive_index;
        }
      }

      //@barrier("local");

      _occa_exclusive_index = 0;
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          double AUtmp = 0, AVtmp = 0, AWtmp = 0;
#pragma unroll 4
          for (int m = 0; m < 4; m++) {
            // 9 shared, 18 flops => 12TB/s*18/(9*8) = 3TFLOPS/s
            double Dmi = s_D[m][i];
            double Dmj = s_D[m][j];
            double Dkm = s_D[k][m];
            AUtmp += Dmi * s_GUr[j][m];
            AUtmp += Dmj * s_GUs[m][i];
            AVtmp += Dmi * s_GVr[j][m];
            AVtmp += Dmj * s_GVs[m][i];
            AWtmp += Dmi * s_GWr[j][m];
            AWtmp += Dmj * s_GWs[m][i];
            r_AU[_occa_exclusive_index][m] += Dkm * r_Ut[_occa_exclusive_index];
            r_AV[_occa_exclusive_index][m] += Dkm * r_Vt[_occa_exclusive_index];
            r_AW[_occa_exclusive_index][m] += Dkm * r_Wt[_occa_exclusive_index];
          }
          r_AU[_occa_exclusive_index][k] += AUtmp;
          r_AV[_occa_exclusive_index][k] += AVtmp;
          r_AW[_occa_exclusive_index][k] += AWtmp;
          ++_occa_exclusive_index;
        }
      }
    }

    // write out

    _occa_exclusive_index = 0;
    for (int j = 0; j < 4; ++j) {
      for (int i = 0; i < 4; ++i) {
#pragma unroll 4
        for (int k = 0; k < 4; k++) {
          const long int id = e * p_Np + k * 4 * 4 + j * 4 + i;
          Aq[id + 0 * offset] = r_AU[_occa_exclusive_index][k];
          Aq[id + 1 * offset] = r_AV[_occa_exclusive_index][k];
          Aq[id + 2 * offset] = r_AW[_occa_exclusive_index][k];
        }
        ++_occa_exclusive_index;
      }
    }
  }
}

extern "C" void axhelm_n3_v0(const long int & Nelements,
                             const long int & offset,
                             const double * __restrict__ ggeo,
                             const double * __restrict__ D,
                             const double * __restrict__ lambda,
                             const double * __restrict__ q,
                             double * __restrict__ Aq) {
  for (long int e = 0; e < Nelements; ++e) {
    int _occa_exclusive_index;
    double s_D[4][4];
    double s_U[4][4];
    double s_V[4][4];
    double s_W[4][4];
    double s_GUr[4][4];
    double s_GUs[4][4];
    double s_GVr[4][4];
    double s_GVs[4][4];
    double s_GWr[4][4];
    double s_GWs[4][4];
    double r_Ut[256], r_Vt[256], r_Wt[256];
    double r_U[256][4], r_V[256][4], r_W[256][4];
    double r_AU[256][4], r_AV[256][4], r_AW[256][4];
    double r_G00[256], r_G01[256], r_G02[256], r_G11[256], r_G12[256], r_G22[256], r_GwJ[256];
    double r_lam0[256], r_lam1[256];

    // array of threads
    _occa_exclusive_index = 0;
    for (int j = 0; j < 4; ++j) {
      for (int i = 0; i < 4; ++i) {
        //load D into local memory
        // s_D[i][j] = d \phi_i at node j
        s_D[j][i] = D[4 * j + i];
        // D is column major

        // load pencil of u into register
        const long int base = i + j * 4 + e * p_Np;
#pragma unroll 4
        for (int k = 0; k < 4; k++) {
          //
          r_U[_occa_exclusive_index][k] = q[base + k * 4 * 4 + 0 * offset];
          r_V[_occa_exclusive_index][k] = q[base + k * 4 * 4 + 1 * offset];
          r_W[_occa_exclusive_index][k] = q[base + k * 4 * 4 + 2 * offset];
          //
          r_AU[_occa_exclusive_index][k] = 0.f;
          r_AV[_occa_exclusive_index][k] = 0.f;
          r_AW[_occa_exclusive_index][k] = 0.f;
        }
        ++_occa_exclusive_index;
      }
    }

    // Layer by layer
#pragma unroll 4
    for (int k = 0; k < 4; ++k) {
      _occa_exclusive_index = 0;
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          const long int id = e * p_Np + k * 4 * 4 + j * 4 + i;

          // prefetch geometric factors
          const long int gbase = e * p_Nggeo * p_Np + k * 4 * 4 + j * 4 + i;
          r_G00[_occa_exclusive_index] = ggeo[gbase + p_G00ID * p_Np];
          r_G01[_occa_exclusive_index] = ggeo[gbase + p_G01ID * p_Np];
          r_G02[_occa_exclusive_index] = ggeo[gbase + p_G02ID * p_Np];
          r_G11[_occa_exclusive_index] = ggeo[gbase + p_G11ID * p_Np];
          r_G12[_occa_exclusive_index] = ggeo[gbase + p_G12ID * p_Np];
          r_G22[_occa_exclusive_index] = ggeo[gbase + p_G22ID * p_Np];
          r_GwJ[_occa_exclusive_index] = ggeo[gbase + p_GWJID * p_Np];
          r_lam0[_occa_exclusive_index] = lambda[id + 0 * offset];
          r_lam1[_occa_exclusive_index] = lambda[id + 1 * offset];
          ++_occa_exclusive_index;
        }
      }

      //@barrier("local");

      _occa_exclusive_index = 0;
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          // share u(:,:,k)
          s_U[j][i] = r_U[_occa_exclusive_index][k];
          s_V[j][i] = r_V[_occa_exclusive_index][k];
          s_W[j][i] = r_W[_occa_exclusive_index][k];
          r_Ut[_occa_exclusive_index] = 0;
          r_Vt[_occa_exclusive_index] = 0;
          r_Wt[_occa_exclusive_index] = 0;
#pragma unroll 4
          for (int m = 0; m < 4; m++) {
            double Dkm = s_D[k][m];
            r_Ut[_occa_exclusive_index] += Dkm * r_U[_occa_exclusive_index][m];
            r_Vt[_occa_exclusive_index] += Dkm * r_V[_occa_exclusive_index][m];
            r_Wt[_occa_exclusive_index] += Dkm * r_W[_occa_exclusive_index][m];
          }
          ++_occa_exclusive_index;
        }
      }

      //@barrier("local");

      _occa_exclusive_index = 0;
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          double Ur = 0.f, Us = 0.f;
          double Vr = 0.f, Vs = 0.f;
          double Wr = 0.f, Ws = 0.f;
#pragma unroll 4
          for (int m = 0; m < 4; m++) {
            double Dim = s_D[i][m];
            double Djm = s_D[j][m];
            Ur += Dim * s_U[j][m];
            Us += Djm * s_U[m][i];
            Vr += Dim * s_V[j][m];
            Vs += Djm * s_V[m][i];
            Wr += Dim * s_W[j][m];
            Ws += Djm * s_W[m][i];
          }
          s_GUr[j][i] = r_lam0[_occa_exclusive_index] * (r_G00[_occa_exclusive_index] * Ur + r_G01[_occa_exclusive_index] * Us + r_G02[_occa_exclusive_index] * r_Ut[_occa_exclusive_index]);
          s_GVr[j][i] = r_lam0[_occa_exclusive_index] * (r_G00[_occa_exclusive_index] * Vr + r_G01[_occa_exclusive_index] * Vs + r_G02[_occa_exclusive_index] * r_Vt[_occa_exclusive_index]);
          s_GWr[j][i] = r_lam0[_occa_exclusive_index] * (r_G00[_occa_exclusive_index] * Wr + r_G01[_occa_exclusive_index] * Ws + r_G02[_occa_exclusive_index] * r_Wt[_occa_exclusive_index]);
          s_GUs[j][i] = r_lam0[_occa_exclusive_index] * (r_G01[_occa_exclusive_index] * Ur + r_G11[_occa_exclusive_index] * Us + r_G12[_occa_exclusive_index] * r_Ut[_occa_exclusive_index]);
          s_GVs[j][i] = r_lam0[_occa_exclusive_index] * (r_G01[_occa_exclusive_index] * Vr + r_G11[_occa_exclusive_index] * Vs + r_G12[_occa_exclusive_index] * r_Vt[_occa_exclusive_index]);
          s_GWs[j][i] = r_lam0[_occa_exclusive_index] * (r_G01[_occa_exclusive_index] * Wr + r_G11[_occa_exclusive_index] * Ws + r_G12[_occa_exclusive_index] * r_Wt[_occa_exclusive_index]);
          r_Ut[_occa_exclusive_index] = r_lam0[_occa_exclusive_index] * (r_G02[_occa_exclusive_index] * Ur + r_G12[_occa_exclusive_index] * Us + r_G22[_occa_exclusive_index] * r_Ut[_occa_exclusive_index]);
          r_Vt[_occa_exclusive_index] = r_lam0[_occa_exclusive_index] * (r_G02[_occa_exclusive_index] * Vr + r_G12[_occa_exclusive_index] * Vs + r_G22[_occa_exclusive_index] * r_Vt[_occa_exclusive_index]);
          r_Wt[_occa_exclusive_index] = r_lam0[_occa_exclusive_index] * (r_G02[_occa_exclusive_index] * Wr + r_G12[_occa_exclusive_index] * Ws + r_G22[_occa_exclusive_index] * r_Wt[_occa_exclusive_index]);
          r_AU[_occa_exclusive_index][k] += r_GwJ[_occa_exclusive_index] * r_lam1[_occa_exclusive_index] * r_U[_occa_exclusive_index][k];
          r_AV[_occa_exclusive_index][k] += r_GwJ[_occa_exclusive_index] * r_lam1[_occa_exclusive_index] * r_V[_occa_exclusive_index][k];
          r_AW[_occa_exclusive_index][k] += r_GwJ[_occa_exclusive_index] * r_lam1[_occa_exclusive_index] * r_W[_occa_exclusive_index][k];
          ++_occa_exclusive_index;
        }
      }

      //@barrier("local");

      _occa_exclusive_index = 0;
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          double AUtmp = 0, AVtmp = 0, AWtmp = 0;
#pragma unroll 4
          for (int m = 0; m < 4; m++) {
            // 9 shared, 18 flops => 12TB/s*18/(9*8) = 3TFLOPS/s
            double Dmi = s_D[m][i];
            double Dmj = s_D[m][j];
            double Dkm = s_D[k][m];
            AUtmp += Dmi * s_GUr[j][m];
            AUtmp += Dmj * s_GUs[m][i];
            AVtmp += Dmi * s_GVr[j][m];
            AVtmp += Dmj * s_GVs[m][i];
            AWtmp += Dmi * s_GWr[j][m];
            AWtmp += Dmj * s_GWs[m][i];
            r_AU[_occa_exclusive_index][m] += Dkm * r_Ut[_occa_exclusive_index];
            r_AV[_occa_exclusive_index][m] += Dkm * r_Vt[_occa_exclusive_index];
            r_AW[_occa_exclusive_index][m] += Dkm * r_Wt[_occa_exclusive_index];
          }
          r_AU[_occa_exclusive_index][k] += AUtmp;
          r_AV[_occa_exclusive_index][k] += AVtmp;
          r_AW[_occa_exclusive_index][k] += AWtmp;
          ++_occa_exclusive_index;
        }
      }
    }

    // write out

    _occa_exclusive_index = 0;
    for (int j = 0; j < 4; ++j) {
      for (int i = 0; i < 4; ++i) {
#pragma unroll 4
        for (int k = 0; k < 4; k++) {
          const long int id = e * p_Np + k * 4 * 4 + j * 4 + i;
          Aq[id + 0 * offset] = r_AU[_occa_exclusive_index][k];
          Aq[id + 1 * offset] = r_AV[_occa_exclusive_index][k];
          Aq[id + 2 * offset] = r_AW[_occa_exclusive_index][k];
        }
        ++_occa_exclusive_index;
      }
    }
  }
}

extern "C" void axhelmPartial_n3_v0(const long int & Nelements,
                                    const long int & offset,
                                    const long int * __restrict__ elementList,
                                    const double * __restrict__ ggeo,
                                    const double * __restrict__ D,
                                    const double * __restrict__ lambda,
                                    const double * __restrict__ q,
                                    double * __restrict__ Aq) {
  for (long int e = 0; e < Nelements; ++e) {
    int _occa_exclusive_index;
    double s_D[4][4];
    double s_U[4][4];
    double s_V[4][4];
    double s_W[4][4];
    double s_GUr[4][4];
    double s_GUs[4][4];
    double s_GVr[4][4];
    double s_GVs[4][4];
    double s_GWr[4][4];
    double s_GWs[4][4];
    double r_Ut[256], r_Vt[256], r_Wt[256];
    double r_U[256][4], r_V[256][4], r_W[256][4];
    double r_AU[256][4], r_AV[256][4], r_AW[256][4];
    long int element[256];
    double r_G00[256], r_G01[256], r_G02[256], r_G11[256], r_G12[256], r_G22[256], r_GwJ[256];
    double r_lam0[256], r_lam1[256];

    // array of threads
    _occa_exclusive_index = 0;
    for (int j = 0; j < 4; ++j) {
      for (int i = 0; i < 4; ++i) {
        //load D into local memory
        // s_D[i][j] = d \phi_i at node j
        s_D[j][i] = D[4 * j + i];
        // D is column major
        element[_occa_exclusive_index] = elementList[e];

        // load pencil of u into register
        const long int base = i + j * 4 + element[_occa_exclusive_index] * p_Np;
#pragma unroll 4
        for (int k = 0; k < 4; k++) {
          //
          r_U[_occa_exclusive_index][k] = q[base + k * 4 * 4 + 0 * offset];
          r_V[_occa_exclusive_index][k] = q[base + k * 4 * 4 + 1 * offset];
          r_W[_occa_exclusive_index][k] = q[base + k * 4 * 4 + 2 * offset];
          //
          r_AU[_occa_exclusive_index][k] = 0.f;
          r_AV[_occa_exclusive_index][k] = 0.f;
          r_AW[_occa_exclusive_index][k] = 0.f;
        }
        ++_occa_exclusive_index;
      }
    }

    // Layer by layer
#pragma unroll 4
    for (int k = 0; k < 4; ++k) {
      _occa_exclusive_index = 0;
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          const long int id = element[_occa_exclusive_index] * p_Np + k * 4 * 4 + j * 4 + i;

          // prefetch geometric factors
          const long int gbase = element[_occa_exclusive_index] * p_Nggeo * p_Np + k * 4 * 4 + j * 4 + i;
          r_G00[_occa_exclusive_index] = ggeo[gbase + p_G00ID * p_Np];
          r_G01[_occa_exclusive_index] = ggeo[gbase + p_G01ID * p_Np];
          r_G02[_occa_exclusive_index] = ggeo[gbase + p_G02ID * p_Np];
          r_G11[_occa_exclusive_index] = ggeo[gbase + p_G11ID * p_Np];
          r_G12[_occa_exclusive_index] = ggeo[gbase + p_G12ID * p_Np];
          r_G22[_occa_exclusive_index] = ggeo[gbase + p_G22ID * p_Np];
          r_GwJ[_occa_exclusive_index] = ggeo[gbase + p_GWJID * p_Np];
          r_lam0[_occa_exclusive_index] = lambda[id + 0 * offset];
          r_lam1[_occa_exclusive_index] = lambda[id + 1 * offset];
          ++_occa_exclusive_index;
        }
      }

      //@barrier("local");

      _occa_exclusive_index = 0;
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          // share u(:,:,k)
          s_U[j][i] = r_U[_occa_exclusive_index][k];
          s_V[j][i] = r_V[_occa_exclusive_index][k];
          s_W[j][i] = r_W[_occa_exclusive_index][k];
          r_Ut[_occa_exclusive_index] = 0;
          r_Vt[_occa_exclusive_index] = 0;
          r_Wt[_occa_exclusive_index] = 0;
#pragma unroll 4
          for (int m = 0; m < 4; m++) {
            double Dkm = s_D[k][m];
            r_Ut[_occa_exclusive_index] += Dkm * r_U[_occa_exclusive_index][m];
            r_Vt[_occa_exclusive_index] += Dkm * r_V[_occa_exclusive_index][m];
            r_Wt[_occa_exclusive_index] += Dkm * r_W[_occa_exclusive_index][m];
          }
          ++_occa_exclusive_index;
        }
      }

      //@barrier("local");

      _occa_exclusive_index = 0;
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          double Ur = 0.f, Us = 0.f;
          double Vr = 0.f, Vs = 0.f;
          double Wr = 0.f, Ws = 0.f;
#pragma unroll 4
          for (int m = 0; m < 4; m++) {
            double Dim = s_D[i][m];
            double Djm = s_D[j][m];
            Ur += Dim * s_U[j][m];
            Us += Djm * s_U[m][i];
            Vr += Dim * s_V[j][m];
            Vs += Djm * s_V[m][i];
            Wr += Dim * s_W[j][m];
            Ws += Djm * s_W[m][i];
          }
          s_GUr[j][i] = r_lam0[_occa_exclusive_index] * (r_G00[_occa_exclusive_index] * Ur + r_G01[_occa_exclusive_index] * Us + r_G02[_occa_exclusive_index] * r_Ut[_occa_exclusive_index]);
          s_GVr[j][i] = r_lam0[_occa_exclusive_index] * (r_G00[_occa_exclusive_index] * Vr + r_G01[_occa_exclusive_index] * Vs + r_G02[_occa_exclusive_index] * r_Vt[_occa_exclusive_index]);
          s_GWr[j][i] = r_lam0[_occa_exclusive_index] * (r_G00[_occa_exclusive_index] * Wr + r_G01[_occa_exclusive_index] * Ws + r_G02[_occa_exclusive_index] * r_Wt[_occa_exclusive_index]);
          s_GUs[j][i] = r_lam0[_occa_exclusive_index] * (r_G01[_occa_exclusive_index] * Ur + r_G11[_occa_exclusive_index] * Us + r_G12[_occa_exclusive_index] * r_Ut[_occa_exclusive_index]);
          s_GVs[j][i] = r_lam0[_occa_exclusive_index] * (r_G01[_occa_exclusive_index] * Vr + r_G11[_occa_exclusive_index] * Vs + r_G12[_occa_exclusive_index] * r_Vt[_occa_exclusive_index]);
          s_GWs[j][i] = r_lam0[_occa_exclusive_index] * (r_G01[_occa_exclusive_index] * Wr + r_G11[_occa_exclusive_index] * Ws + r_G12[_occa_exclusive_index] * r_Wt[_occa_exclusive_index]);
          r_Ut[_occa_exclusive_index] = r_lam0[_occa_exclusive_index] * (r_G02[_occa_exclusive_index] * Ur + r_G12[_occa_exclusive_index] * Us + r_G22[_occa_exclusive_index] * r_Ut[_occa_exclusive_index]);
          r_Vt[_occa_exclusive_index] = r_lam0[_occa_exclusive_index] * (r_G02[_occa_exclusive_index] * Vr + r_G12[_occa_exclusive_index] * Vs + r_G22[_occa_exclusive_index] * r_Vt[_occa_exclusive_index]);
          r_Wt[_occa_exclusive_index] = r_lam0[_occa_exclusive_index] * (r_G02[_occa_exclusive_index] * Wr + r_G12[_occa_exclusive_index] * Ws + r_G22[_occa_exclusive_index] * r_Wt[_occa_exclusive_index]);
          r_AU[_occa_exclusive_index][k] += r_GwJ[_occa_exclusive_index] * r_lam1[_occa_exclusive_index] * r_U[_occa_exclusive_index][k];
          r_AV[_occa_exclusive_index][k] += r_GwJ[_occa_exclusive_index] * r_lam1[_occa_exclusive_index] * r_V[_occa_exclusive_index][k];
          r_AW[_occa_exclusive_index][k] += r_GwJ[_occa_exclusive_index] * r_lam1[_occa_exclusive_index] * r_W[_occa_exclusive_index][k];
          ++_occa_exclusive_index;
        }
      }

      //@barrier("local");

      _occa_exclusive_index = 0;
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          double AUtmp = 0, AVtmp = 0, AWtmp = 0;
#pragma unroll 4
          for (int m = 0; m < 4; m++) {
            // 9 shared, 18 flops => 12TB/s*18/(9*8) = 3TFLOPS/s
            double Dmi = s_D[m][i];
            double Dmj = s_D[m][j];
            double Dkm = s_D[k][m];
            AUtmp += Dmi * s_GUr[j][m];
            AUtmp += Dmj * s_GUs[m][i];
            AVtmp += Dmi * s_GVr[j][m];
            AVtmp += Dmj * s_GVs[m][i];
            AWtmp += Dmi * s_GWr[j][m];
            AWtmp += Dmj * s_GWs[m][i];
            r_AU[_occa_exclusive_index][m] += Dkm * r_Ut[_occa_exclusive_index];
            r_AV[_occa_exclusive_index][m] += Dkm * r_Vt[_occa_exclusive_index];
            r_AW[_occa_exclusive_index][m] += Dkm * r_Wt[_occa_exclusive_index];
          }
          r_AU[_occa_exclusive_index][k] += AUtmp;
          r_AV[_occa_exclusive_index][k] += AVtmp;
          r_AW[_occa_exclusive_index][k] += AWtmp;
          ++_occa_exclusive_index;
        }
      }
    }

    // write out

    _occa_exclusive_index = 0;
    for (int j = 0; j < 4; ++j) {
      for (int i = 0; i < 4; ++i) {
#pragma unroll 4
        for (int k = 0; k < 4; k++) {
          const long int id = element[_occa_exclusive_index] * p_Np + k * 4 * 4 + j * 4 + i;
          Aq[id + 0 * offset] = r_AU[_occa_exclusive_index][k];
          Aq[id + 1 * offset] = r_AV[_occa_exclusive_index][k];
          Aq[id + 2 * offset] = r_AW[_occa_exclusive_index][k];
        }
        ++_occa_exclusive_index;
      }
    }
  }
}

extern "C" void axhelm_bk_v0(const long int & Nelements,
                             const long int & offset,
                             const double * __restrict__ ggeo,
                             const double * __restrict__ D,
                             const double * __restrict__ lambda,
                             const double * __restrict__ q,
                             double * __restrict__ Aq) {
  for (long int e = 0; e < Nelements; ++e) {
    int _occa_exclusive_index;
    double s_D[4][4];
    double s_q[4][4];
    double s_Gqr[4][4];
    double s_Gqs[4][4];
    double r_qt[256], r_Gqt[256], r_Auk[256];
    double r_q[256][4];
    // register array to hold u(i,j,0:N) private to thread
    double r_Aq[256][4];
    // array for results Au(i,j,0:N)

    double r_G00[256], r_G01[256], r_G02[256], r_G11[256], r_G12[256], r_G22[256], r_GwJ[256];

    // array of threads
    _occa_exclusive_index = 0;
    for (int j = 0; j < 4; ++j) {
      for (int i = 0; i < 4; ++i) {
        //load D into local memory
        // s_D[i][j] = d \phi_i at node j
        s_D[j][i] = D[4 * j + i];
        // D is column major

        // load pencil of u into register
        const long int base = i + j * 4 + e * p_Np;
#pragma unroll 4
        for (int k = 0; k < 4; ++k) {
          r_q[_occa_exclusive_index][k] = q[base + k * 4 * 4];
          // prefetch operation
          r_Aq[_occa_exclusive_index][k] = 0.f;
          // zero the accumulator
        }
        ++_occa_exclusive_index;
      }
    }

    // Layer by layer
#pragma unroll 4
    for (int k = 0; k < 4; ++k) {
      _occa_exclusive_index = 0;
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          // prefetch geometric factors
          const long int gbase = e * p_Nggeo * p_Np + k * 4 * 4 + j * 4 + i;
          r_G00[_occa_exclusive_index] = ggeo[gbase + p_G00ID * p_Np];
          r_G01[_occa_exclusive_index] = ggeo[gbase + p_G01ID * p_Np];
          r_G02[_occa_exclusive_index] = ggeo[gbase + p_G02ID * p_Np];
          r_G11[_occa_exclusive_index] = ggeo[gbase + p_G11ID * p_Np];
          r_G12[_occa_exclusive_index] = ggeo[gbase + p_G12ID * p_Np];
          r_G22[_occa_exclusive_index] = ggeo[gbase + p_G22ID * p_Np];
          r_GwJ[_occa_exclusive_index] = ggeo[gbase + p_GWJID * p_Np];
          ++_occa_exclusive_index;
        }
      }

      //@barrier("local");

      _occa_exclusive_index = 0;
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          // share u(:,:,k)
          s_q[j][i] = r_q[_occa_exclusive_index][k];
          r_qt[_occa_exclusive_index] = 0;
#pragma unroll 4
          for (int m = 0; m < 4; ++m) {
            r_qt[_occa_exclusive_index] += s_D[k][m] * r_q[_occa_exclusive_index][m];
          }
          ++_occa_exclusive_index;
        }
      }

      //@barrier("local");

      _occa_exclusive_index = 0;
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          double qr = 0.f;
          double qs = 0.f;
#pragma unroll 4
          for (int m = 0; m < 4; ++m) {
            qr += s_D[i][m] * s_q[j][m];
            qs += s_D[j][m] * s_q[m][i];
          }
          s_Gqs[j][i] = (r_G01[_occa_exclusive_index] * qr + r_G11[_occa_exclusive_index] * qs + r_G12[_occa_exclusive_index] * r_qt[_occa_exclusive_index]);
          s_Gqr[j][i] = (r_G00[_occa_exclusive_index] * qr + r_G01[_occa_exclusive_index] * qs + r_G02[_occa_exclusive_index] * r_qt[_occa_exclusive_index]);

          // put this here for a performance bump
          r_Gqt[_occa_exclusive_index] = (r_G02[_occa_exclusive_index] * qr + r_G12[_occa_exclusive_index] * qs + r_G22[_occa_exclusive_index] * r_qt[_occa_exclusive_index]);
          //r_Auk = r_GwJ*lambda[1]*r_q[k];
          r_Auk[_occa_exclusive_index] = 0;
          ++_occa_exclusive_index;
        }
      }

      //@barrier("local");

      _occa_exclusive_index = 0;
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
#pragma unroll 4
          for (int m = 0; m < 4; ++m) {
            r_Auk[_occa_exclusive_index] += s_D[m][j] * s_Gqs[m][i];
            r_Aq[_occa_exclusive_index][m] += s_D[k][m] * r_Gqt[_occa_exclusive_index];
            // DT(m,k)*ut(i,j,k,e)
            r_Auk[_occa_exclusive_index] += s_D[m][i] * s_Gqr[j][m];
          }
          r_Aq[_occa_exclusive_index][k] += r_Auk[_occa_exclusive_index];
          ++_occa_exclusive_index;
        }
      }
    }

    // write out

    _occa_exclusive_index = 0;
    for (int j = 0; j < 4; ++j) {
      for (int i = 0; i < 4; ++i) {
#pragma unroll 4
        for (int k = 0; k < 4; ++k) {
          const long int id = e * p_Np + k * 4 * 4 + j * 4 + i;
          Aq[id] = r_Aq[_occa_exclusive_index][k];
        }
        ++_occa_exclusive_index;
      }
    }
  }
}

extern "C" void axhelmPartial_bk_v0(const long int & Nelements,
                                    const long int & offset,
                                    const long int * __restrict__ elementList,
                                    const double * __restrict__ ggeo,
                                    const double * __restrict__ D,
                                    const double * __restrict__ lambda,
                                    const double * __restrict__ q,
                                    double * __restrict__ Aq) {
  for (long int e = 0; e < Nelements; ++e) {
    int _occa_exclusive_index;
    double s_D[4][4];
    double s_q[4][4];
    double s_Gqr[4][4];
    double s_Gqs[4][4];
    double r_qt[256], r_Gqt[256], r_Auk[256];
    double r_q[256][4];
    // register array to hold u(i,j,0:N) private to thread
    double r_Aq[256][4];
    // array for results Au(i,j,0:N)

    long int element[256];
    double r_G00[256], r_G01[256], r_G02[256], r_G11[256], r_G12[256], r_G22[256], r_GwJ[256];

    // array of threads
    _occa_exclusive_index = 0;
    for (int j = 0; j < 4; ++j) {
      for (int i = 0; i < 4; ++i) {
        //load D into local memory
        // s_D[i][j] = d \phi_i at node j
        s_D[j][i] = D[4 * j + i];
        // D is column major
        element[_occa_exclusive_index] = elementList[e];

        // load pencil of u into register
        const long int base = i + j * 4 + element[_occa_exclusive_index] * p_Np;
#pragma unroll 4
        for (int k = 0; k < 4; ++k) {
          r_q[_occa_exclusive_index][k] = q[base + k * 4 * 4];
          // prefetch operation
          r_Aq[_occa_exclusive_index][k] = 0.f;
          // zero the accumulator
        }
        ++_occa_exclusive_index;
      }
    }

    // Layer by layer
#pragma unroll 4
    for (int k = 0; k < 4; ++k) {
      _occa_exclusive_index = 0;
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          // prefetch geometric factors
          const long int gbase = element[_occa_exclusive_index] * p_Nggeo * p_Np + k * 4 * 4 + j * 4 + i;
          r_G00[_occa_exclusive_index] = ggeo[gbase + p_G00ID * p_Np];
          r_G01[_occa_exclusive_index] = ggeo[gbase + p_G01ID * p_Np];
          r_G02[_occa_exclusive_index] = ggeo[gbase + p_G02ID * p_Np];
          r_G11[_occa_exclusive_index] = ggeo[gbase + p_G11ID * p_Np];
          r_G12[_occa_exclusive_index] = ggeo[gbase + p_G12ID * p_Np];
          r_G22[_occa_exclusive_index] = ggeo[gbase + p_G22ID * p_Np];
          r_GwJ[_occa_exclusive_index] = ggeo[gbase + p_GWJID * p_Np];
          ++_occa_exclusive_index;
        }
      }

      //@barrier("local");

      _occa_exclusive_index = 0;
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          // share u(:,:,k)
          s_q[j][i] = r_q[_occa_exclusive_index][k];
          r_qt[_occa_exclusive_index] = 0;
#pragma unroll 4
          for (int m = 0; m < 4; ++m) {
            r_qt[_occa_exclusive_index] += s_D[k][m] * r_q[_occa_exclusive_index][m];
          }
          ++_occa_exclusive_index;
        }
      }

      //@barrier("local");

      _occa_exclusive_index = 0;
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          double qr = 0.f;
          double qs = 0.f;
#pragma unroll 4
          for (int m = 0; m < 4; ++m) {
            qr += s_D[i][m] * s_q[j][m];
            qs += s_D[j][m] * s_q[m][i];
          }
          s_Gqs[j][i] = (r_G01[_occa_exclusive_index] * qr + r_G11[_occa_exclusive_index] * qs + r_G12[_occa_exclusive_index] * r_qt[_occa_exclusive_index]);
          s_Gqr[j][i] = (r_G00[_occa_exclusive_index] * qr + r_G01[_occa_exclusive_index] * qs + r_G02[_occa_exclusive_index] * r_qt[_occa_exclusive_index]);

          // put this here for a performance bump
          r_Gqt[_occa_exclusive_index] = (r_G02[_occa_exclusive_index] * qr + r_G12[_occa_exclusive_index] * qs + r_G22[_occa_exclusive_index] * r_qt[_occa_exclusive_index]);
          //r_Auk = r_GwJ*lambda[1]*r_q[k];
          r_Auk[_occa_exclusive_index] = 0;
          ++_occa_exclusive_index;
        }
      }

      //@barrier("local");

      _occa_exclusive_index = 0;
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
#pragma unroll 4
          for (int m = 0; m < 4; ++m) {
            r_Auk[_occa_exclusive_index] += s_D[m][j] * s_Gqs[m][i];
            r_Aq[_occa_exclusive_index][m] += s_D[k][m] * r_Gqt[_occa_exclusive_index];
            // DT(m,k)*ut(i,j,k,e)
            r_Auk[_occa_exclusive_index] += s_D[m][i] * s_Gqr[j][m];
          }
          r_Aq[_occa_exclusive_index][k] += r_Auk[_occa_exclusive_index];
          ++_occa_exclusive_index;
        }
      }
    }

    // write out

    _occa_exclusive_index = 0;
    for (int j = 0; j < 4; ++j) {
      for (int i = 0; i < 4; ++i) {
#pragma unroll 4
        for (int k = 0; k < 4; ++k) {
          const long int id = element[_occa_exclusive_index] * p_Np + k * 4 * 4 + j * 4 + i;
          Aq[id] = r_Aq[_occa_exclusive_index][k];
        }
        ++_occa_exclusive_index;
      }
    }
  }
}

