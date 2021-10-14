#include <CL/sycl.hpp>
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

extern "C" void _occa_axhelm_v0_0(sycl::queue * queue_,
                                  sycl::nd_range<3> * range_,
                                  const int & Nelements,
                                  const int & offset,
                                  const double * __restrict__ ggeo,
                                  const double * __restrict__ D,
                                  const double * __restrict__ lambda,
                                  const double * __restrict__ q,
                                  double * __restrict__ Aq) {
  queue_->submit(
    [&](sycl::handler & handler_) {
      handler_.parallel_for(
        *range_,
        [=](sycl::nd_item<3> item_) {
          auto & s_Gqs = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_Gqr = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_q = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_D = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          {
            int e = 0 + item_.get_group(2);
            double r_qt, r_Gqt, r_Auk;
            double r_q[8];
            // register array to hold u(i,j,0:N) private to thread
            double r_Aq[8];
            // array for results Au(i,j,0:N)

            double r_G00, r_G01, r_G02, r_G11, r_G12, r_G22, r_GwJ;
            double r_lam0, r_lam1;

            // array of threads
            {
              int j = 0 + item_.get_local_id(1);
              {
                int i = 0 + item_.get_local_id(2);
                //load D into local memory
                // s_D[i][j] = d \phi_i at node j
                s_D[j][i] = D[8 * j + i];
                // D is column major

                // load pencil of u into register
                const int base = i + j * 8 + e * 512;
#pragma unroll 8
                for (int k = 0; k < 8; ++k) {
                  r_q[k] = q[base + k * 8 * 8];
                  // prefetch operation
                  r_Aq[k] = 0.f;
                  // zero the accumulator
                }
              }
            }
            item_.barrier(sycl::access::fence_space::local_space);

            // Layer by layer
#pragma unroll 8
            for (int k = 0; k < 8; ++k) {
              {
                int j = 0 + item_.get_local_id(1);
                {
                  int i = 0 + item_.get_local_id(2);
                  const int id = e * 512 + k * 8 * 8 + j * 8 + i;

                  // prefetch geometric factors
                  const int gbase = e * 7 * 512 + k * 8 * 8 + j * 8 + i;
                  r_G00 = ggeo[gbase + 1 * 512];
                  r_G01 = ggeo[gbase + 2 * 512];
                  r_G02 = ggeo[gbase + 3 * 512];
                  r_G11 = ggeo[gbase + 4 * 512];
                  r_G12 = ggeo[gbase + 5 * 512];
                  r_G22 = ggeo[gbase + 6 * 512];
                  r_GwJ = ggeo[gbase + 0 * 512];
                  r_lam0 = lambda[id + 0 * offset];
                  r_lam1 = lambda[id + 1 * offset];
                }
              }

              //@barrier("local");

              {
                int j = 0 + item_.get_local_id(1);
                {
                  int i = 0 + item_.get_local_id(2);
                  // share u(:,:,k)
                  s_q[j][i] = r_q[k];
                  r_qt = 0;
#pragma unroll 8
                  for (int m = 0; m < 8; ++m) {
                    r_qt += s_D[k][m] * r_q[m];
                  }
                }
              }
              item_.barrier(sycl::access::fence_space::local_space);

              //@barrier("local");

              {
                int j = 0 + item_.get_local_id(1);
                {
                  int i = 0 + item_.get_local_id(2);
                  double qr = 0.f;
                  double qs = 0.f;
#pragma unroll 8
                  for (int m = 0; m < 8; ++m) {
                    qr += s_D[i][m] * s_q[j][m];
                    qs += s_D[j][m] * s_q[m][i];
                  }
                  s_Gqs[j][i] = r_lam0 * (r_G01 * qr + r_G11 * qs + r_G12 * r_qt);
                  s_Gqr[j][i] = r_lam0 * (r_G00 * qr + r_G01 * qs + r_G02 * r_qt);

                  // put this here for a performance bump
                  r_Gqt = r_lam0 * (r_G02 * qr + r_G12 * qs + r_G22 * r_qt);
                  r_Auk = r_GwJ * r_lam1 * r_q[k];
                }
              }
              item_.barrier(sycl::access::fence_space::local_space);

              //@barrier("local");

              {
                int j = 0 + item_.get_local_id(1);
                {
                  int i = 0 + item_.get_local_id(2);
#pragma unroll 8
                  for (int m = 0; m < 8; ++m) {
                    r_Auk += s_D[m][j] * s_Gqs[m][i];
                    r_Aq[m] += s_D[k][m] * r_Gqt;
                    // DT(m,k)*ut(i,j,k,e)
                    r_Auk += s_D[m][i] * s_Gqr[j][m];
                  }
                  r_Aq[k] += r_Auk;
                }
              }
              item_.barrier(sycl::access::fence_space::local_space);
            }

            // write out

            {
              int j = 0 + item_.get_local_id(1);
              {
                int i = 0 + item_.get_local_id(2);
#pragma unroll 8
                for (int k = 0; k < 8; ++k) {
                  const int id = e * 512 + k * 8 * 8 + j * 8 + i;
                  Aq[id] = r_Aq[k];
                }
              }
            }
          }
        }
      );
    }
  );
}

extern "C" void _occa_axhelmPartial_v0_0(sycl::queue * queue_,
                                         sycl::nd_range<3> * range_,
                                         const int & Nelements,
                                         const int & offset,
                                         const int * __restrict__ elementList,
                                         const double * __restrict__ ggeo,
                                         const double * __restrict__ D,
                                         const double * __restrict__ lambda,
                                         const double * __restrict__ q,
                                         double * __restrict__ Aq) {
  queue_->submit(
    [&](sycl::handler & handler_) {
      handler_.parallel_for(
        *range_,
        [=](sycl::nd_item<3> item_) {
          auto & s_Gqs = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_Gqr = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_q = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_D = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          {
            int e = 0 + item_.get_group(2);
            double r_qt, r_Gqt, r_Auk;
            double r_q[8];
            // register array to hold u(i,j,0:N) private to thread
            double r_Aq[8];
            // array for results Au(i,j,0:N)

            int element;
            double r_G00, r_G01, r_G02, r_G11, r_G12, r_G22, r_GwJ;
            double r_lam0, r_lam1;

            // array of threads
            {
              int j = 0 + item_.get_local_id(1);
              {
                int i = 0 + item_.get_local_id(2);
                //load D into local memory
                // s_D[i][j] = d \phi_i at node j
                s_D[j][i] = D[8 * j + i];
                // D is column major
                element = elementList[e];

                // load pencil of u into register
                const int base = i + j * 8 + element * 512;
#pragma unroll 8
                for (int k = 0; k < 8; ++k) {
                  r_q[k] = q[base + k * 8 * 8];
                  // prefetch operation
                  r_Aq[k] = 0.f;
                  // zero the accumulator
                }
              }
            }
            item_.barrier(sycl::access::fence_space::local_space);

            // Layer by layer
#pragma unroll 8
            for (int k = 0; k < 8; ++k) {
              {
                int j = 0 + item_.get_local_id(1);
                {
                  int i = 0 + item_.get_local_id(2);
                  const int id = element * 512 + k * 8 * 8 + j * 8 + i;

                  // prefetch geometric factors
                  const int gbase = element * 7 * 512 + k * 8 * 8 + j * 8 + i;
                  r_G00 = ggeo[gbase + 1 * 512];
                  r_G01 = ggeo[gbase + 2 * 512];
                  r_G02 = ggeo[gbase + 3 * 512];
                  r_G11 = ggeo[gbase + 4 * 512];
                  r_G12 = ggeo[gbase + 5 * 512];
                  r_G22 = ggeo[gbase + 6 * 512];
                  r_GwJ = ggeo[gbase + 0 * 512];
                  r_lam0 = lambda[id + 0 * offset];
                  r_lam1 = lambda[id + 1 * offset];
                }
              }

              //@barrier("local");

              {
                int j = 0 + item_.get_local_id(1);
                {
                  int i = 0 + item_.get_local_id(2);
                  // share u(:,:,k)
                  s_q[j][i] = r_q[k];
                  r_qt = 0;
#pragma unroll 8
                  for (int m = 0; m < 8; ++m) {
                    r_qt += s_D[k][m] * r_q[m];
                  }
                }
              }
              item_.barrier(sycl::access::fence_space::local_space);

              //@barrier("local");

              {
                int j = 0 + item_.get_local_id(1);
                {
                  int i = 0 + item_.get_local_id(2);
                  double qr = 0.f;
                  double qs = 0.f;
#pragma unroll 8
                  for (int m = 0; m < 8; ++m) {
                    qr += s_D[i][m] * s_q[j][m];
                    qs += s_D[j][m] * s_q[m][i];
                  }
                  s_Gqs[j][i] = r_lam0 * (r_G01 * qr + r_G11 * qs + r_G12 * r_qt);
                  s_Gqr[j][i] = r_lam0 * (r_G00 * qr + r_G01 * qs + r_G02 * r_qt);

                  // put this here for a performance bump
                  r_Gqt = r_lam0 * (r_G02 * qr + r_G12 * qs + r_G22 * r_qt);
                  r_Auk = r_GwJ * r_lam1 * r_q[k];
                }
              }
              item_.barrier(sycl::access::fence_space::local_space);

              //@barrier("local");

              {
                int j = 0 + item_.get_local_id(1);
                {
                  int i = 0 + item_.get_local_id(2);
#pragma unroll 8
                  for (int m = 0; m < 8; ++m) {
                    r_Auk += s_D[m][j] * s_Gqs[m][i];
                    r_Aq[m] += s_D[k][m] * r_Gqt;
                    // DT(m,k)*ut(i,j,k,e)
                    r_Auk += s_D[m][i] * s_Gqr[j][m];
                  }
                  r_Aq[k] += r_Auk;
                }
              }
              item_.barrier(sycl::access::fence_space::local_space);
            }

            // write out

            {
              int j = 0 + item_.get_local_id(1);
              {
                int i = 0 + item_.get_local_id(2);
#pragma unroll 8
                for (int k = 0; k < 8; ++k) {
                  const int id = element * 512 + k * 8 * 8 + j * 8 + i;
                  Aq[id] = r_Aq[k];
                }
              }
            }
          }
        }
      );
    }
  );
}

extern "C" void _occa_axhelmPartial_bk_n3_v0_0(sycl::queue * queue_,
                                               sycl::nd_range<3> * range_,
                                               const int & Nelements,
                                               const int & offset,
                                               const int * __restrict__ elementList,
                                               const double * __restrict__ ggeo,
                                               const double * __restrict__ D,
                                               const double * __restrict__ lambda,
                                               const double * __restrict__ q,
                                               double * __restrict__ Aq) {
  queue_->submit(
    [&](sycl::handler & handler_) {
      handler_.parallel_for(
        *range_,
        [=](sycl::nd_item<3> item_) {
          auto & s_GWs = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_GWr = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_GVs = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_GVr = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_GUs = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_GUr = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_W = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_V = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_U = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_D = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          {
            int e = 0 + item_.get_group(2);
            double r_Ut, r_Vt, r_Wt;
            double r_U[8], r_V[8], r_W[8];
            double r_AU[8], r_AV[8], r_AW[8];
            int element;
            double r_G00, r_G01, r_G02, r_G11, r_G12, r_G22, r_GwJ;

            // array of threads
            {
              int j = 0 + item_.get_local_id(1);
              {
                int i = 0 + item_.get_local_id(2);
                //load D into local memory
                // s_D[i][j] = d \phi_i at node j
                s_D[j][i] = D[8 * j + i];
                // D is column major

                element = elementList[e];

                // load pencil of u into register
                const int base = i + j * 8 + element * 512;
#pragma unroll 8
                for (int k = 0; k < 8; k++) {
                  //
                  r_U[k] = q[base + k * 8 * 8 + 0 * offset];
                  r_V[k] = q[base + k * 8 * 8 + 1 * offset];
                  r_W[k] = q[base + k * 8 * 8 + 2 * offset];
                  //
                  r_AU[k] = 0.f;
                  r_AV[k] = 0.f;
                  r_AW[k] = 0.f;
                }
              }
            }
            item_.barrier(sycl::access::fence_space::local_space);

            // Layer by layer
#pragma unroll 8
            for (int k = 0; k < 8; ++k) {
              {
                int j = 0 + item_.get_local_id(1);
                {
                  int i = 0 + item_.get_local_id(2);
                  const int id = element * 512 + k * 8 * 8 + j * 8 + i;

                  // prefetch geometric factors
                  const int gbase = element * 7 * 512 + k * 8 * 8 + j * 8 + i;
                  r_G00 = ggeo[gbase + 1 * 512];
                  r_G01 = ggeo[gbase + 2 * 512];
                  r_G02 = ggeo[gbase + 3 * 512];
                  r_G11 = ggeo[gbase + 4 * 512];
                  r_G12 = ggeo[gbase + 5 * 512];
                  r_G22 = ggeo[gbase + 6 * 512];
                  r_GwJ = ggeo[gbase + 0 * 512];
                }
              }

              //@barrier("local");

              {
                int j = 0 + item_.get_local_id(1);
                {
                  int i = 0 + item_.get_local_id(2);
                  // share u(:,:,k)
                  s_U[j][i] = r_U[k];
                  s_V[j][i] = r_V[k];
                  s_W[j][i] = r_W[k];
                  r_Ut = 0;
                  r_Vt = 0;
                  r_Wt = 0;
#pragma unroll 8
                  for (int m = 0; m < 8; m++) {
                    double Dkm = s_D[k][m];
                    r_Ut += Dkm * r_U[m];
                    r_Vt += Dkm * r_V[m];
                    r_Wt += Dkm * r_W[m];
                  }
                }
              }
              item_.barrier(sycl::access::fence_space::local_space);

              //@barrier("local");

              {
                int j = 0 + item_.get_local_id(1);
                {
                  int i = 0 + item_.get_local_id(2);
                  double Ur = 0.f, Us = 0.f;
                  double Vr = 0.f, Vs = 0.f;
                  double Wr = 0.f, Ws = 0.f;
#pragma unroll 8
                  for (int m = 0; m < 8; m++) {
                    double Dim = s_D[i][m];
                    double Djm = s_D[j][m];
                    Ur += Dim * s_U[j][m];
                    Us += Djm * s_U[m][i];
                    Vr += Dim * s_V[j][m];
                    Vs += Djm * s_V[m][i];
                    Wr += Dim * s_W[j][m];
                    Ws += Djm * s_W[m][i];
                  }
                  s_GUr[j][i] = (r_G00 * Ur + r_G01 * Us + r_G02 * r_Ut);
                  s_GVr[j][i] = (r_G00 * Vr + r_G01 * Vs + r_G02 * r_Vt);
                  s_GWr[j][i] = (r_G00 * Wr + r_G01 * Ws + r_G02 * r_Wt);
                  s_GUs[j][i] = (r_G01 * Ur + r_G11 * Us + r_G12 * r_Ut);
                  s_GVs[j][i] = (r_G01 * Vr + r_G11 * Vs + r_G12 * r_Vt);
                  s_GWs[j][i] = (r_G01 * Wr + r_G11 * Ws + r_G12 * r_Wt);
                  r_Ut = (r_G02 * Ur + r_G12 * Us + r_G22 * r_Ut);
                  r_Vt = (r_G02 * Vr + r_G12 * Vs + r_G22 * r_Vt);
                  r_Wt = (r_G02 * Wr + r_G12 * Ws + r_G22 * r_Wt);
                }
              }
              item_.barrier(sycl::access::fence_space::local_space);

              //@barrier("local");

              {
                int j = 0 + item_.get_local_id(1);
                {
                  int i = 0 + item_.get_local_id(2);
                  double AUtmp = 0, AVtmp = 0, AWtmp = 0;
#pragma unroll 8
                  for (int m = 0; m < 8; m++) {
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
                    r_AU[m] += Dkm * r_Ut;
                    r_AV[m] += Dkm * r_Vt;
                    r_AW[m] += Dkm * r_Wt;
                  }
                  r_AU[k] += AUtmp;
                  r_AV[k] += AVtmp;
                  r_AW[k] += AWtmp;
                }
              }
              item_.barrier(sycl::access::fence_space::local_space);
            }

            // write out

            {
              int j = 0 + item_.get_local_id(1);
              {
                int i = 0 + item_.get_local_id(2);
#pragma unroll 8
                for (int k = 0; k < 8; k++) {
                  const int id = element * 512 + k * 8 * 8 + j * 8 + i;
                  Aq[id + 0 * offset] = r_AU[k];
                  Aq[id + 1 * offset] = r_AV[k];
                  Aq[id + 2 * offset] = r_AW[k];
                }
              }
            }
          }
        }
      );
    }
  );
}

extern "C" void _occa_axhelm_bk_n3_v0_0(sycl::queue * queue_,
                                        sycl::nd_range<3> * range_,
                                        const int & Nelements,
                                        const int & offset,
                                        const double * __restrict__ ggeo,
                                        const double * __restrict__ D,
                                        const double * __restrict__ lambda,
                                        const double * __restrict__ q,
                                        double * __restrict__ Aq) {
  queue_->submit(
    [&](sycl::handler & handler_) {
      handler_.parallel_for(
        *range_,
        [=](sycl::nd_item<3> item_) {
          auto & s_GWs = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_GWr = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_GVs = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_GVr = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_GUs = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_GUr = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_W = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_V = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_U = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_D = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          {
            int e = 0 + item_.get_group(2);
            double r_Ut, r_Vt, r_Wt;
            double r_U[8], r_V[8], r_W[8];
            double r_AU[8], r_AV[8], r_AW[8];
            double r_G00, r_G01, r_G02, r_G11, r_G12, r_G22, r_GwJ;

            // array of threads
            {
              int j = 0 + item_.get_local_id(1);
              {
                int i = 0 + item_.get_local_id(2);
                //load D into local memory
                // s_D[i][j] = d \phi_i at node j
                s_D[j][i] = D[8 * j + i];
                // D is column major

                // load pencil of u into register
                const int base = i + j * 8 + e * 512;
#pragma unroll 8
                for (int k = 0; k < 8; k++) {
                  //
                  r_U[k] = q[base + k * 8 * 8 + 0 * offset];
                  r_V[k] = q[base + k * 8 * 8 + 1 * offset];
                  r_W[k] = q[base + k * 8 * 8 + 2 * offset];
                  //
                  r_AU[k] = 0.f;
                  r_AV[k] = 0.f;
                  r_AW[k] = 0.f;
                }
              }
            }
            item_.barrier(sycl::access::fence_space::local_space);

            // Layer by layer
#pragma unroll 8
            for (int k = 0; k < 8; ++k) {
              {
                int j = 0 + item_.get_local_id(1);
                {
                  int i = 0 + item_.get_local_id(2);
                  const int id = e * 512 + k * 8 * 8 + j * 8 + i;

                  // prefetch geometric factors
                  const int gbase = e * 7 * 512 + k * 8 * 8 + j * 8 + i;
                  r_G00 = ggeo[gbase + 1 * 512];
                  r_G01 = ggeo[gbase + 2 * 512];
                  r_G02 = ggeo[gbase + 3 * 512];
                  r_G11 = ggeo[gbase + 4 * 512];
                  r_G12 = ggeo[gbase + 5 * 512];
                  r_G22 = ggeo[gbase + 6 * 512];
                  r_GwJ = ggeo[gbase + 0 * 512];
                }
              }

              //@barrier("local");

              {
                int j = 0 + item_.get_local_id(1);
                {
                  int i = 0 + item_.get_local_id(2);
                  // share u(:,:,k)
                  s_U[j][i] = r_U[k];
                  s_V[j][i] = r_V[k];
                  s_W[j][i] = r_W[k];
                  r_Ut = 0;
                  r_Vt = 0;
                  r_Wt = 0;
#pragma unroll 8
                  for (int m = 0; m < 8; m++) {
                    double Dkm = s_D[k][m];
                    r_Ut += Dkm * r_U[m];
                    r_Vt += Dkm * r_V[m];
                    r_Wt += Dkm * r_W[m];
                  }
                }
              }
              item_.barrier(sycl::access::fence_space::local_space);

              //@barrier("local");

              {
                int j = 0 + item_.get_local_id(1);
                {
                  int i = 0 + item_.get_local_id(2);
                  double Ur = 0.f, Us = 0.f;
                  double Vr = 0.f, Vs = 0.f;
                  double Wr = 0.f, Ws = 0.f;
#pragma unroll 8
                  for (int m = 0; m < 8; m++) {
                    double Dim = s_D[i][m];
                    double Djm = s_D[j][m];
                    Ur += Dim * s_U[j][m];
                    Us += Djm * s_U[m][i];
                    Vr += Dim * s_V[j][m];
                    Vs += Djm * s_V[m][i];
                    Wr += Dim * s_W[j][m];
                    Ws += Djm * s_W[m][i];
                  }
                  s_GUr[j][i] = (r_G00 * Ur + r_G01 * Us + r_G02 * r_Ut);
                  s_GVr[j][i] = (r_G00 * Vr + r_G01 * Vs + r_G02 * r_Vt);
                  s_GWr[j][i] = (r_G00 * Wr + r_G01 * Ws + r_G02 * r_Wt);
                  s_GUs[j][i] = (r_G01 * Ur + r_G11 * Us + r_G12 * r_Ut);
                  s_GVs[j][i] = (r_G01 * Vr + r_G11 * Vs + r_G12 * r_Vt);
                  s_GWs[j][i] = (r_G01 * Wr + r_G11 * Ws + r_G12 * r_Wt);
                  r_Ut = (r_G02 * Ur + r_G12 * Us + r_G22 * r_Ut);
                  r_Vt = (r_G02 * Vr + r_G12 * Vs + r_G22 * r_Vt);
                  r_Wt = (r_G02 * Wr + r_G12 * Ws + r_G22 * r_Wt);
                }
              }
              item_.barrier(sycl::access::fence_space::local_space);

              //@barrier("local");

              {
                int j = 0 + item_.get_local_id(1);
                {
                  int i = 0 + item_.get_local_id(2);
                  double AUtmp = 0, AVtmp = 0, AWtmp = 0;
#pragma unroll 8
                  for (int m = 0; m < 8; m++) {
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
                    r_AU[m] += Dkm * r_Ut;
                    r_AV[m] += Dkm * r_Vt;
                    r_AW[m] += Dkm * r_Wt;
                  }
                  r_AU[k] += AUtmp;
                  r_AV[k] += AVtmp;
                  r_AW[k] += AWtmp;
                }
              }
              item_.barrier(sycl::access::fence_space::local_space);
            }

            // write out

            {
              int j = 0 + item_.get_local_id(1);
              {
                int i = 0 + item_.get_local_id(2);
#pragma unroll 8
                for (int k = 0; k < 8; k++) {
                  const int id = e * 512 + k * 8 * 8 + j * 8 + i;
                  Aq[id + 0 * offset] = r_AU[k];
                  Aq[id + 1 * offset] = r_AV[k];
                  Aq[id + 2 * offset] = r_AW[k];
                }
              }
            }
          }
        }
      );
    }
  );
}

extern "C" void _occa_axhelm_n3_v0_0(sycl::queue * queue_,
                                     sycl::nd_range<3> * range_,
                                     const int & Nelements,
                                     const int & offset,
                                     const double * __restrict__ ggeo,
                                     const double * __restrict__ D,
                                     const double * __restrict__ lambda,
                                     const double * __restrict__ q,
                                     double * __restrict__ Aq) {
  queue_->submit(
    [&](sycl::handler & handler_) {
      handler_.parallel_for(
        *range_,
        [=](sycl::nd_item<3> item_) {
          auto & s_GWs = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_GWr = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_GVs = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_GVr = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_GUs = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_GUr = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_W = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_V = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_U = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_D = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          {
            int e = 0 + item_.get_group(2);
            double r_Ut, r_Vt, r_Wt;
            double r_U[8], r_V[8], r_W[8];
            double r_AU[8], r_AV[8], r_AW[8];
            double r_G00, r_G01, r_G02, r_G11, r_G12, r_G22, r_GwJ;
            double r_lam0, r_lam1;

            // array of threads
            {
              int j = 0 + item_.get_local_id(1);
              {
                int i = 0 + item_.get_local_id(2);
                //load D into local memory
                // s_D[i][j] = d \phi_i at node j
                s_D[j][i] = D[8 * j + i];
                // D is column major

                // load pencil of u into register
                const int base = i + j * 8 + e * 512;
#pragma unroll 8
                for (int k = 0; k < 8; k++) {
                  //
                  r_U[k] = q[base + k * 8 * 8 + 0 * offset];
                  r_V[k] = q[base + k * 8 * 8 + 1 * offset];
                  r_W[k] = q[base + k * 8 * 8 + 2 * offset];
                  //
                  r_AU[k] = 0.f;
                  r_AV[k] = 0.f;
                  r_AW[k] = 0.f;
                }
              }
            }
            item_.barrier(sycl::access::fence_space::local_space);

            // Layer by layer
#pragma unroll 8
            for (int k = 0; k < 8; ++k) {
              {
                int j = 0 + item_.get_local_id(1);
                {
                  int i = 0 + item_.get_local_id(2);
                  const int id = e * 512 + k * 8 * 8 + j * 8 + i;

                  // prefetch geometric factors
                  const int gbase = e * 7 * 512 + k * 8 * 8 + j * 8 + i;
                  r_G00 = ggeo[gbase + 1 * 512];
                  r_G01 = ggeo[gbase + 2 * 512];
                  r_G02 = ggeo[gbase + 3 * 512];
                  r_G11 = ggeo[gbase + 4 * 512];
                  r_G12 = ggeo[gbase + 5 * 512];
                  r_G22 = ggeo[gbase + 6 * 512];
                  r_GwJ = ggeo[gbase + 0 * 512];
                  r_lam0 = lambda[id + 0 * offset];
                  r_lam1 = lambda[id + 1 * offset];
                }
              }

              //@barrier("local");

              {
                int j = 0 + item_.get_local_id(1);
                {
                  int i = 0 + item_.get_local_id(2);
                  // share u(:,:,k)
                  s_U[j][i] = r_U[k];
                  s_V[j][i] = r_V[k];
                  s_W[j][i] = r_W[k];
                  r_Ut = 0;
                  r_Vt = 0;
                  r_Wt = 0;
#pragma unroll 8
                  for (int m = 0; m < 8; m++) {
                    double Dkm = s_D[k][m];
                    r_Ut += Dkm * r_U[m];
                    r_Vt += Dkm * r_V[m];
                    r_Wt += Dkm * r_W[m];
                  }
                }
              }
              item_.barrier(sycl::access::fence_space::local_space);

              //@barrier("local");

              {
                int j = 0 + item_.get_local_id(1);
                {
                  int i = 0 + item_.get_local_id(2);
                  double Ur = 0.f, Us = 0.f;
                  double Vr = 0.f, Vs = 0.f;
                  double Wr = 0.f, Ws = 0.f;
#pragma unroll 8
                  for (int m = 0; m < 8; m++) {
                    double Dim = s_D[i][m];
                    double Djm = s_D[j][m];
                    Ur += Dim * s_U[j][m];
                    Us += Djm * s_U[m][i];
                    Vr += Dim * s_V[j][m];
                    Vs += Djm * s_V[m][i];
                    Wr += Dim * s_W[j][m];
                    Ws += Djm * s_W[m][i];
                  }
                  s_GUr[j][i] = r_lam0 * (r_G00 * Ur + r_G01 * Us + r_G02 * r_Ut);
                  s_GVr[j][i] = r_lam0 * (r_G00 * Vr + r_G01 * Vs + r_G02 * r_Vt);
                  s_GWr[j][i] = r_lam0 * (r_G00 * Wr + r_G01 * Ws + r_G02 * r_Wt);
                  s_GUs[j][i] = r_lam0 * (r_G01 * Ur + r_G11 * Us + r_G12 * r_Ut);
                  s_GVs[j][i] = r_lam0 * (r_G01 * Vr + r_G11 * Vs + r_G12 * r_Vt);
                  s_GWs[j][i] = r_lam0 * (r_G01 * Wr + r_G11 * Ws + r_G12 * r_Wt);
                  r_Ut = r_lam0 * (r_G02 * Ur + r_G12 * Us + r_G22 * r_Ut);
                  r_Vt = r_lam0 * (r_G02 * Vr + r_G12 * Vs + r_G22 * r_Vt);
                  r_Wt = r_lam0 * (r_G02 * Wr + r_G12 * Ws + r_G22 * r_Wt);
                  r_AU[k] += r_GwJ * r_lam1 * r_U[k];
                  r_AV[k] += r_GwJ * r_lam1 * r_V[k];
                  r_AW[k] += r_GwJ * r_lam1 * r_W[k];
                }
              }
              item_.barrier(sycl::access::fence_space::local_space);

              //@barrier("local");

              {
                int j = 0 + item_.get_local_id(1);
                {
                  int i = 0 + item_.get_local_id(2);
                  double AUtmp = 0, AVtmp = 0, AWtmp = 0;
#pragma unroll 8
                  for (int m = 0; m < 8; m++) {
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
                    r_AU[m] += Dkm * r_Ut;
                    r_AV[m] += Dkm * r_Vt;
                    r_AW[m] += Dkm * r_Wt;
                  }
                  r_AU[k] += AUtmp;
                  r_AV[k] += AVtmp;
                  r_AW[k] += AWtmp;
                }
              }
              item_.barrier(sycl::access::fence_space::local_space);
            }

            // write out

            {
              int j = 0 + item_.get_local_id(1);
              {
                int i = 0 + item_.get_local_id(2);
#pragma unroll 8
                for (int k = 0; k < 8; k++) {
                  const int id = e * 512 + k * 8 * 8 + j * 8 + i;
                  Aq[id + 0 * offset] = r_AU[k];
                  Aq[id + 1 * offset] = r_AV[k];
                  Aq[id + 2 * offset] = r_AW[k];
                }
              }
            }
          }
        }
      );
    }
  );
}

extern "C" void _occa_axhelmPartial_n3_v0_0(sycl::queue * queue_,
                                            sycl::nd_range<3> * range_,
                                            const int & Nelements,
                                            const int & offset,
                                            const int * __restrict__ elementList,
                                            const double * __restrict__ ggeo,
                                            const double * __restrict__ D,
                                            const double * __restrict__ lambda,
                                            const double * __restrict__ q,
                                            double * __restrict__ Aq) {
  queue_->submit(
    [&](sycl::handler & handler_) {
      handler_.parallel_for(
        *range_,
        [=](sycl::nd_item<3> item_) {
          auto & s_GWs = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_GWr = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_GVs = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_GVr = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_GUs = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_GUr = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_W = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_V = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_U = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_D = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          {
            int e = 0 + item_.get_group(2);
            double r_Ut, r_Vt, r_Wt;
            double r_U[8], r_V[8], r_W[8];
            double r_AU[8], r_AV[8], r_AW[8];
            int element;
            double r_G00, r_G01, r_G02, r_G11, r_G12, r_G22, r_GwJ;
            double r_lam0, r_lam1;

            // array of threads
            {
              int j = 0 + item_.get_local_id(1);
              {
                int i = 0 + item_.get_local_id(2);
                //load D into local memory
                // s_D[i][j] = d \phi_i at node j
                s_D[j][i] = D[8 * j + i];
                // D is column major
                element = elementList[e];

                // load pencil of u into register
                const int base = i + j * 8 + element * 512;
#pragma unroll 8
                for (int k = 0; k < 8; k++) {
                  //
                  r_U[k] = q[base + k * 8 * 8 + 0 * offset];
                  r_V[k] = q[base + k * 8 * 8 + 1 * offset];
                  r_W[k] = q[base + k * 8 * 8 + 2 * offset];
                  //
                  r_AU[k] = 0.f;
                  r_AV[k] = 0.f;
                  r_AW[k] = 0.f;
                }
              }
            }
            item_.barrier(sycl::access::fence_space::local_space);

            // Layer by layer
#pragma unroll 8
            for (int k = 0; k < 8; ++k) {
              {
                int j = 0 + item_.get_local_id(1);
                {
                  int i = 0 + item_.get_local_id(2);
                  const int id = element * 512 + k * 8 * 8 + j * 8 + i;

                  // prefetch geometric factors
                  const int gbase = element * 7 * 512 + k * 8 * 8 + j * 8 + i;
                  r_G00 = ggeo[gbase + 1 * 512];
                  r_G01 = ggeo[gbase + 2 * 512];
                  r_G02 = ggeo[gbase + 3 * 512];
                  r_G11 = ggeo[gbase + 4 * 512];
                  r_G12 = ggeo[gbase + 5 * 512];
                  r_G22 = ggeo[gbase + 6 * 512];
                  r_GwJ = ggeo[gbase + 0 * 512];
                  r_lam0 = lambda[id + 0 * offset];
                  r_lam1 = lambda[id + 1 * offset];
                }
              }

              //@barrier("local");

              {
                int j = 0 + item_.get_local_id(1);
                {
                  int i = 0 + item_.get_local_id(2);
                  // share u(:,:,k)
                  s_U[j][i] = r_U[k];
                  s_V[j][i] = r_V[k];
                  s_W[j][i] = r_W[k];
                  r_Ut = 0;
                  r_Vt = 0;
                  r_Wt = 0;
#pragma unroll 8
                  for (int m = 0; m < 8; m++) {
                    double Dkm = s_D[k][m];
                    r_Ut += Dkm * r_U[m];
                    r_Vt += Dkm * r_V[m];
                    r_Wt += Dkm * r_W[m];
                  }
                }
              }
              item_.barrier(sycl::access::fence_space::local_space);

              //@barrier("local");

              {
                int j = 0 + item_.get_local_id(1);
                {
                  int i = 0 + item_.get_local_id(2);
                  double Ur = 0.f, Us = 0.f;
                  double Vr = 0.f, Vs = 0.f;
                  double Wr = 0.f, Ws = 0.f;
#pragma unroll 8
                  for (int m = 0; m < 8; m++) {
                    double Dim = s_D[i][m];
                    double Djm = s_D[j][m];
                    Ur += Dim * s_U[j][m];
                    Us += Djm * s_U[m][i];
                    Vr += Dim * s_V[j][m];
                    Vs += Djm * s_V[m][i];
                    Wr += Dim * s_W[j][m];
                    Ws += Djm * s_W[m][i];
                  }
                  s_GUr[j][i] = r_lam0 * (r_G00 * Ur + r_G01 * Us + r_G02 * r_Ut);
                  s_GVr[j][i] = r_lam0 * (r_G00 * Vr + r_G01 * Vs + r_G02 * r_Vt);
                  s_GWr[j][i] = r_lam0 * (r_G00 * Wr + r_G01 * Ws + r_G02 * r_Wt);
                  s_GUs[j][i] = r_lam0 * (r_G01 * Ur + r_G11 * Us + r_G12 * r_Ut);
                  s_GVs[j][i] = r_lam0 * (r_G01 * Vr + r_G11 * Vs + r_G12 * r_Vt);
                  s_GWs[j][i] = r_lam0 * (r_G01 * Wr + r_G11 * Ws + r_G12 * r_Wt);
                  r_Ut = r_lam0 * (r_G02 * Ur + r_G12 * Us + r_G22 * r_Ut);
                  r_Vt = r_lam0 * (r_G02 * Vr + r_G12 * Vs + r_G22 * r_Vt);
                  r_Wt = r_lam0 * (r_G02 * Wr + r_G12 * Ws + r_G22 * r_Wt);
                  r_AU[k] += r_GwJ * r_lam1 * r_U[k];
                  r_AV[k] += r_GwJ * r_lam1 * r_V[k];
                  r_AW[k] += r_GwJ * r_lam1 * r_W[k];
                }
              }
              item_.barrier(sycl::access::fence_space::local_space);

              //@barrier("local");

              {
                int j = 0 + item_.get_local_id(1);
                {
                  int i = 0 + item_.get_local_id(2);
                  double AUtmp = 0, AVtmp = 0, AWtmp = 0;
#pragma unroll 8
                  for (int m = 0; m < 8; m++) {
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
                    r_AU[m] += Dkm * r_Ut;
                    r_AV[m] += Dkm * r_Vt;
                    r_AW[m] += Dkm * r_Wt;
                  }
                  r_AU[k] += AUtmp;
                  r_AV[k] += AVtmp;
                  r_AW[k] += AWtmp;
                }
              }
              item_.barrier(sycl::access::fence_space::local_space);
            }

            // write out

            {
              int j = 0 + item_.get_local_id(1);
              {
                int i = 0 + item_.get_local_id(2);
#pragma unroll 8
                for (int k = 0; k < 8; k++) {
                  const int id = element * 512 + k * 8 * 8 + j * 8 + i;
                  Aq[id + 0 * offset] = r_AU[k];
                  Aq[id + 1 * offset] = r_AV[k];
                  Aq[id + 2 * offset] = r_AW[k];
                }
              }
            }
          }
        }
      );
    }
  );
}

extern "C" void _occa_axhelm_bk_v0_0(sycl::queue * queue_,
                                     sycl::nd_range<3> * range_,
                                     const int & Nelements,
                                     const int & offset,
                                     const double * __restrict__ ggeo,
                                     const double * __restrict__ D,
                                     const double * __restrict__ lambda,
                                     const double * __restrict__ q,
                                     double * __restrict__ Aq) {
  queue_->submit(
    [&](sycl::handler & handler_) {
      handler_.parallel_for(
        *range_,
        [=](sycl::nd_item<3> item_) {
          auto & s_Gqs = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_Gqr = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_q = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_D = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          {
            int e = 0 + item_.get_group(2);
            double r_qt, r_Gqt, r_Auk;
            double r_q[8];
            // register array to hold u(i,j,0:N) private to thread
            double r_Aq[8];
            // array for results Au(i,j,0:N)

            double r_G00, r_G01, r_G02, r_G11, r_G12, r_G22, r_GwJ;

            // array of threads
            {
              int j = 0 + item_.get_local_id(1);
              {
                int i = 0 + item_.get_local_id(2);
                //load D into local memory
                // s_D[i][j] = d \phi_i at node j
                s_D[j][i] = D[8 * j + i];
                // D is column major

                // load pencil of u into register
                const int base = i + j * 8 + e * 512;
#pragma unroll 8
                for (int k = 0; k < 8; ++k) {
                  r_q[k] = q[base + k * 8 * 8];
                  // prefetch operation
                  r_Aq[k] = 0.f;
                  // zero the accumulator
                }
              }
            }
            item_.barrier(sycl::access::fence_space::local_space);

            // Layer by layer
#pragma unroll 8
            for (int k = 0; k < 8; ++k) {
              {
                int j = 0 + item_.get_local_id(1);
                {
                  int i = 0 + item_.get_local_id(2);
                  // prefetch geometric factors
                  const int gbase = e * 7 * 512 + k * 8 * 8 + j * 8 + i;
                  r_G00 = ggeo[gbase + 1 * 512];
                  r_G01 = ggeo[gbase + 2 * 512];
                  r_G02 = ggeo[gbase + 3 * 512];
                  r_G11 = ggeo[gbase + 4 * 512];
                  r_G12 = ggeo[gbase + 5 * 512];
                  r_G22 = ggeo[gbase + 6 * 512];
                  r_GwJ = ggeo[gbase + 0 * 512];
                }
              }

              //@barrier("local");

              {
                int j = 0 + item_.get_local_id(1);
                {
                  int i = 0 + item_.get_local_id(2);
                  // share u(:,:,k)
                  s_q[j][i] = r_q[k];
                  r_qt = 0;
#pragma unroll 8
                  for (int m = 0; m < 8; ++m) {
                    r_qt += s_D[k][m] * r_q[m];
                  }
                }
              }
              item_.barrier(sycl::access::fence_space::local_space);

              //@barrier("local");

              {
                int j = 0 + item_.get_local_id(1);
                {
                  int i = 0 + item_.get_local_id(2);
                  double qr = 0.f;
                  double qs = 0.f;
#pragma unroll 8
                  for (int m = 0; m < 8; ++m) {
                    qr += s_D[i][m] * s_q[j][m];
                    qs += s_D[j][m] * s_q[m][i];
                  }
                  s_Gqs[j][i] = (r_G01 * qr + r_G11 * qs + r_G12 * r_qt);
                  s_Gqr[j][i] = (r_G00 * qr + r_G01 * qs + r_G02 * r_qt);

                  // put this here for a performance bump
                  r_Gqt = (r_G02 * qr + r_G12 * qs + r_G22 * r_qt);
                  //r_Auk = r_GwJ*lambda[1]*r_q[k];
                  r_Auk = 0;
                }
              }
              item_.barrier(sycl::access::fence_space::local_space);

              //@barrier("local");

              {
                int j = 0 + item_.get_local_id(1);
                {
                  int i = 0 + item_.get_local_id(2);
#pragma unroll 8
                  for (int m = 0; m < 8; ++m) {
                    r_Auk += s_D[m][j] * s_Gqs[m][i];
                    r_Aq[m] += s_D[k][m] * r_Gqt;
                    // DT(m,k)*ut(i,j,k,e)
                    r_Auk += s_D[m][i] * s_Gqr[j][m];
                  }
                  r_Aq[k] += r_Auk;
                }
              }
              item_.barrier(sycl::access::fence_space::local_space);
            }

            // write out

            {
              int j = 0 + item_.get_local_id(1);
              {
                int i = 0 + item_.get_local_id(2);
#pragma unroll 8
                for (int k = 0; k < 8; ++k) {
                  const int id = e * 512 + k * 8 * 8 + j * 8 + i;
                  Aq[id] = r_Aq[k];
                }
              }
            }
          }
        }
      );
    }
  );
}

extern "C" void _occa_axhelmPartial_bk_v0_0(sycl::queue * queue_,
                                            sycl::nd_range<3> * range_,
                                            const int & Nelements,
                                            const int & offset,
                                            const int * __restrict__ elementList,
                                            const double * __restrict__ ggeo,
                                            const double * __restrict__ D,
                                            const double * __restrict__ lambda,
                                            const double * __restrict__ q,
                                            double * __restrict__ Aq) {
  queue_->submit(
    [&](sycl::handler & handler_) {
      handler_.parallel_for(
        *range_,
        [=](sycl::nd_item<3> item_) {
          auto & s_Gqs = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_Gqr = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_q = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          auto & s_D = *(sycl::group_local_memory_for_overwrite<double[8][8]>(item_.get_group()));
          {
            int e = 0 + item_.get_group(2);
            double r_qt, r_Gqt, r_Auk;
            double r_q[8];
            // register array to hold u(i,j,0:N) private to thread
            double r_Aq[8];
            // array for results Au(i,j,0:N)

            int element;
            double r_G00, r_G01, r_G02, r_G11, r_G12, r_G22, r_GwJ;

            // array of threads
            {
              int j = 0 + item_.get_local_id(1);
              {
                int i = 0 + item_.get_local_id(2);
                //load D into local memory
                // s_D[i][j] = d \phi_i at node j
                s_D[j][i] = D[8 * j + i];
                // D is column major
                element = elementList[e];

                // load pencil of u into register
                const int base = i + j * 8 + element * 512;
#pragma unroll 8
                for (int k = 0; k < 8; ++k) {
                  r_q[k] = q[base + k * 8 * 8];
                  // prefetch operation
                  r_Aq[k] = 0.f;
                  // zero the accumulator
                }
              }
            }
            item_.barrier(sycl::access::fence_space::local_space);

            // Layer by layer
#pragma unroll 8
            for (int k = 0; k < 8; ++k) {
              {
                int j = 0 + item_.get_local_id(1);
                {
                  int i = 0 + item_.get_local_id(2);
                  // prefetch geometric factors
                  const int gbase = element * 7 * 512 + k * 8 * 8 + j * 8 + i;
                  r_G00 = ggeo[gbase + 1 * 512];
                  r_G01 = ggeo[gbase + 2 * 512];
                  r_G02 = ggeo[gbase + 3 * 512];
                  r_G11 = ggeo[gbase + 4 * 512];
                  r_G12 = ggeo[gbase + 5 * 512];
                  r_G22 = ggeo[gbase + 6 * 512];
                  r_GwJ = ggeo[gbase + 0 * 512];
                }
              }

              //@barrier("local");

              {
                int j = 0 + item_.get_local_id(1);
                {
                  int i = 0 + item_.get_local_id(2);
                  // share u(:,:,k)
                  s_q[j][i] = r_q[k];
                  r_qt = 0;
#pragma unroll 8
                  for (int m = 0; m < 8; ++m) {
                    r_qt += s_D[k][m] * r_q[m];
                  }
                }
              }
              item_.barrier(sycl::access::fence_space::local_space);

              //@barrier("local");

              {
                int j = 0 + item_.get_local_id(1);
                {
                  int i = 0 + item_.get_local_id(2);
                  double qr = 0.f;
                  double qs = 0.f;
#pragma unroll 8
                  for (int m = 0; m < 8; ++m) {
                    qr += s_D[i][m] * s_q[j][m];
                    qs += s_D[j][m] * s_q[m][i];
                  }
                  s_Gqs[j][i] = (r_G01 * qr + r_G11 * qs + r_G12 * r_qt);
                  s_Gqr[j][i] = (r_G00 * qr + r_G01 * qs + r_G02 * r_qt);

                  // put this here for a performance bump
                  r_Gqt = (r_G02 * qr + r_G12 * qs + r_G22 * r_qt);
                  //r_Auk = r_GwJ*lambda[1]*r_q[k];
                  r_Auk = 0;
                }
              }
              item_.barrier(sycl::access::fence_space::local_space);

              //@barrier("local");

              {
                int j = 0 + item_.get_local_id(1);
                {
                  int i = 0 + item_.get_local_id(2);
#pragma unroll 8
                  for (int m = 0; m < 8; ++m) {
                    r_Auk += s_D[m][j] * s_Gqs[m][i];
                    r_Aq[m] += s_D[k][m] * r_Gqt;
                    // DT(m,k)*ut(i,j,k,e)
                    r_Auk += s_D[m][i] * s_Gqr[j][m];
                  }
                  r_Aq[k] += r_Auk;
                }
              }
              item_.barrier(sycl::access::fence_space::local_space);
            }

            // write out

            {
              int j = 0 + item_.get_local_id(1);
              {
                int i = 0 + item_.get_local_id(2);
#pragma unroll 8
                for (int k = 0; k < 8; ++k) {
                  const int id = element * 512 + k * 8 * 8 + j * 8 + i;
                  Aq[id] = r_Aq[k];
                }
              }
            }
          }
        }
      );
    }
  );
}

