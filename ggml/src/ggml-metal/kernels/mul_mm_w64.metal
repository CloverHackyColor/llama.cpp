// The manual GEMM written for a 64-wide simdgroup. The one in mul_mm.metal splits every wave
// into two virtual 32-lane halves so its tile offsets hold on both widths; here a wave is one
// tile, so the lane mapping differs and the two cannot share a file: a change to either width
// stays on its own side.
//
// Same threadgroup shape as the 32-lane path (same thread count, same 8 KiB of A and B, same
// accumulators per lane), so only the lane-to-output mapping and the tile offsets move. Each
// wave owns 32 columns instead of 16, which drops the column index to a lane id and removes
// the divide from the inner loop.

#include "common.h"
#include "dequantize.h"

constant bool  FC_mul_mm_bc_inp        [[function_constant(FC_MUL_MM + 0)]];
constant bool  FC_mul_mm_bc_out        [[function_constant(FC_MUL_MM + 1)]];
constant short FC_mul_mm_ne12          [[function_constant(FC_MUL_MM + 2)]];
constant short FC_mul_mm_ne13          [[function_constant(FC_MUL_MM + 3)]];
constant short FC_mul_mm_r2            [[function_constant(FC_MUL_MM + 4)]];
constant short FC_mul_mm_r3            [[function_constant(FC_MUL_MM + 5)]];
constant bool  FC_mul_mm_manual        [[function_constant(FC_MUL_MM + 6)]];
constant bool  FC_mul_mm_narrow        [[function_constant(FC_MUL_MM + 7)]];
constant bool  FC_mul_mm_double_buffer [[function_constant(FC_MUL_MM + 8)]];
constant bool  FC_mul_mm_acc_f32       [[function_constant(FC_MUL_MM + 9)]];

// An activation panel bigger than this is assumed not to stay resident in L2. The wave64
// path runs on GCN/Vega, whose L2 is 2 MiB class (RX 580: 2 MiB); leave headroom for the
// weight panel and the threadgroup scratch that share it.
constant uint64_t LX_L2_BUDGET = 1024ull*1024ull;

// NLR rows and NCC columns of the output per lane. NCC 4 with 8 lanes per row block gives a
// wave 32 columns, which is what keeps the accumulator count equal to the 32-lane path.
template<
    typename S0, typename S0_4x4, typename S0_8x8,
    typename S1, typename S1_2x4, typename S1_8x8,
    typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &),
    typename T0, typename T0_4x4, typename T1, typename T1_2x4, short NLR = 4>
kernel void kernel_mul_mm_w64(
        constant ggml_metal_kargs_mul_mm & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        uint3  tgpg[[threadgroups_per_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    constexpr int NR0 = 64;
    constexpr short NCC = 4;

    const int NR1 = FC_mul_mm_narrow ? 32 : 64;

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int im = tgpig.z;

    // L2-friendly tile ordering, ported from the Vulkan cm2 mulmat scheduling
    // (llama.cpp #30215). Threadgroups are dispatched x-fastest, so the stock mapping sweeps every
    // column tile for one row tile and re-streams the whole activation matrix once per row tile.
    // Pulling the row tiles into the inner loop instead makes the reused object one column tile's
    // activation panel (NR1 vectors of K, one megabyte at K 4096) instead of the whole matrix, and
    // that panel does stay resident, so it is read once and reused by every row tile. The panel
    // size is therefore not the test: the reorder pays whenever there are more row tiles than
    // column tiles, and both orders cost the same when the panel overflows. Measured on q4_K
    // 12288x1228x4096: 2509 GFLOPS stock, 2753 GFLOPS reordered.
    int tg_x = tgpig.x;
    int tg_y = tgpig.y;
    {
        const uint blocks_x = tgpg.x; // column (activation) tiles
        const uint blocks_y = tgpg.y; // row (weight) tiles
        if (blocks_x > 1 && blocks_x < blocks_y) {
            const uint lin = tgpig.y*blocks_x + tgpig.x;
            tg_x = lin / blocks_y;
            tg_y = lin % blocks_y;
        }
    }

    const int r0 = tg_y*NR0;
    const int r1 = tg_x*NR1;

    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? (args.ne1 - r1) : NR1;

    // a thread shouldn't load data outside of the matrix
    const short lr0  = ((short)tiitg/NL0)       < nr0 ? ((short)tiitg/NL0)       : nr0 - 1;
    const short lr1  = ((short)tiitg/NL1)       < nr1 ? ((short)tiitg/NL1)       : nr1 - 1;
    const short lr1b = ((short)tiitg/NL1 + 32)  < nr1 ? ((short)tiitg/NL1 + 32)  : nr1 - 1;

    const short il0 = (tiitg % NL0);

    short il = il0;

    const int i12 = im % FC_mul_mm_ne12;
    const int i13 = im / FC_mul_mm_ne12;

    const uint64_t offset0 = (i12/FC_mul_mm_r2)*args.nb02 + (i13/FC_mul_mm_r3)*args.nb03;
    const short    offset1 = il0/nl;

    device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const T1 * y = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*(r1 + lr1)
        + args.nb10*iy);

    device const T1 * yb = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*(r1 + lr1b)
        + args.nb10*iy);

    // one wave, one tile: the row block is the low three bits of the lane and the column is the
    // high three, so a wave covers 8 rows by 8 columns of the 8x8 blocks it reads
    const short lane = tiitg % 64;
    const short lr   = lane % 8;
    const short lc   = lane / 8;

    // NLR 4 splits the wave grid two by two over rows and columns; NLR 8 gives each wave all
    // 64 rows and half the columns
    float acc[NLR][NCC];
    for (short i = 0; i < NLR; i++) { for (short j = 0; j < NCC; j++) { acc[i][j] = 0.f; } }

    // NLR 8 runs 128 threads: all load A, each loads two columns of B
    const bool loadA = NLR == 8 || tiitg < 128;

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        const short db = FC_mul_mm_double_buffer ? ((loop_k/NK) & 1) : 0;
        threadgroup S0 * sa = (threadgroup S0 *)(shmem + db*8192);
        threadgroup S1 * sb = (threadgroup S1 *)(shmem + db*8192 + 4096);

        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
            if (!FC_mul_mm_double_buffer) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // no need for dequantization
            for (short i = 0; loadA && i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                // row-major within the 8x8 block so (k, k+1) pairs sit contiguous for the
                // packed-half2 compute loop below
                *(sa + 64*ib + 8*lx + ly) = loop_k + 16*il + i < args.ne00 ? *((device T0 *) x + i) : 0;
            }
        } else {
            S0_4x4 temp_a;
            if (loadA) {
                dequantize_func(x, il, temp_a);
            }

            if (!FC_mul_mm_double_buffer) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            const short sy = (tiitg/NL0)/8;
            const short lx = (tiitg/NL0)%8;

            // the row-major 8x8 layout leaves each group of 4 contiguous, so the 16 scalar
            // stores collapse into 4 vector ones
            const short ib0 = 8*(2*il0    ) + sy;
            const short ib1 = 8*(2*il0 + 1) + sy;

            if (loadA) {
                *(threadgroup vec<S0, 4> *)(sa + 64*ib0 + 8*lx + 0) = temp_a[0];
                *(threadgroup vec<S0, 4> *)(sa + 64*ib0 + 8*lx + 4) = temp_a[1];
                *(threadgroup vec<S0, 4> *)(sa + 64*ib1 + 8*lx + 0) = temp_a[2];
                *(threadgroup vec<S0, 4> *)(sa + 64*ib1 + 8*lx + 4) = temp_a[3];
            }
        }

        {
            constexpr short ncb = 8;

            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;
            const short ly = (tiitg/NL1)%8;

            const short ib = ncb*sx + sy;

            if (FC_mul_mm_bc_inp) {
                for (short i = 0; i < 8; ++i) {
                    *(sb + 64*ib + 8*ly + i) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;
                }
                if (NLR == 8) {
                    for (short i = 0; i < 8; ++i) {
                        *(sb + 64*(ib + 4) + 8*ly + i) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) yb + i) : 0;
                    }
                }
            } else {
                *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)(*((device T1_2x4 *) y));
                if (NLR == 8) {
                    *(threadgroup S1_2x4 *)(sb + 64*(ib + 4) + 8*ly) = (S1_2x4)(*((device T1_2x4 *) yb));
                }
            }
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y  += NK;
        yb += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const S0 * lsma = (sa + (NLR == 8 ? 0 : 4*64*(sgitg % 2)));
        threadgroup const S1 * lsmb = (sb + 4*64*(NLR == 8 ? sgitg : sgitg/2));

        // packed rank-1 updates on (k, k+1) pairs, flushed to float once per NK. The
        // accumulator is half for weights; attention products overflow it, so those use float.
        if (FC_mul_mm_acc_f32) {
            vec<float,2> pacc[NLR][NCC];
            for (short rr = 0; rr < NLR; rr++) { for (short cc = 0; cc < NCC; cc++) { pacc[rr][cc] = 0; } }

            for (short ik = 0; ik < NK/8; ik++) {
                FOR_UNROLL (short k = 0; k < 8; k += 2) {
                    vec<S0,2> av[NLR];
                    vec<S1,2> bv[NCC];
                    for (short rr = 0; rr < NLR; rr++) {
                        av[rr] = *(threadgroup const vec<S0,2> *)(lsma + 64*rr + 8*lr + k);
                    }
                    for (short cc = 0; cc < NCC; cc++) {
                        bv[cc] = *(threadgroup const vec<S1,2> *)(lsmb + 64*cc + 8*lc + k);
                    }
                    for (short rr = 0; rr < NLR; rr++) {
                        for (short cc = 0; cc < NCC; cc++) {
                            // mul+add: no bfloat2 fma() overload; fast math contracts it anyway
                            pacc[rr][cc] = vec<float,2>(av[rr])*vec<float,2>(bv[cc]) + pacc[rr][cc];
                        }
                    }
                }

                lsma += 8*64;
                lsmb += 8*64;
            }

            for (short rr = 0; rr < NLR; rr++) {
                for (short cc = 0; cc < NCC; cc++) {
                    acc[rr][cc] += (float) pacc[rr][cc][0] + (float) pacc[rr][cc][1];
                }
            }
        } else {
            vec<S0,2> pacc[NLR][NCC];

            // the first block writes its product, so the accumulators need no zeroing pass
            FOR_UNROLL (short k = 0; k < 8; k += 2) {
                vec<S0,2> av[NLR];
                vec<S1,2> bv[NCC];
                for (short rr = 0; rr < NLR; rr++) {
                    av[rr] = *(threadgroup const vec<S0,2> *)(lsma + 64*rr + 8*lr + k);
                }
                for (short cc = 0; cc < NCC; cc++) {
                    bv[cc] = *(threadgroup const vec<S1,2> *)(lsmb + 64*cc + 8*lc + k);
                }
                for (short rr = 0; rr < NLR; rr++) {
                    for (short cc = 0; cc < NCC; cc++) {
                        const vec<S0,2> p = vec<S0,2>(av[rr])*vec<S0,2>(bv[cc]);
                        pacc[rr][cc] = k == 0 ? p : p + pacc[rr][cc];
                    }
                }
            }
            lsma += 8*64;
            lsmb += 8*64;

            for (short ik = 1; ik < NK/8; ik++) {
                // bv is the small side, so widening only it halves its reads for four extra
                // registers instead of NLR times four
                if (NLR == 8) {
                    FOR_UNROLL (short k = 0; k < 8; k += 4) {
                        vec<S1,4> bv[NCC];
                        for (short cc = 0; cc < NCC; cc++) {
                            bv[cc] = *(threadgroup const vec<S1,4> *)(lsmb + 64*cc + 8*lc + k);
                        }
                        FOR_UNROLL (short h = 0; h < 2; h++) {
                            vec<S0,2> av[NLR];
                            for (short rr = 0; rr < NLR; rr++) {
                                av[rr] = *(threadgroup const vec<S0,2> *)(lsma + 64*rr + 8*lr + k + 2*h);
                            }
                            for (short rr = 0; rr < NLR; rr++) {
                                for (short cc = 0; cc < NCC; cc++) {
                                    const vec<S0,2> b2 = h == 0 ? vec<S0,2>(bv[cc].xy) : vec<S0,2>(bv[cc].zw);
                                    pacc[rr][cc] = vec<S0,2>(av[rr])*b2 + pacc[rr][cc];
                                }
                            }
                        }
                    }
                } else {
                    FOR_UNROLL (short k = 0; k < 8; k += 2) {
                        vec<S0,2> av[NLR];
                        vec<S1,2> bv[NCC];
                        for (short rr = 0; rr < NLR; rr++) {
                            av[rr] = *(threadgroup const vec<S0,2> *)(lsma + 64*rr + 8*lr + k);
                        }
                        for (short cc = 0; cc < NCC; cc++) {
                            bv[cc] = *(threadgroup const vec<S1,2> *)(lsmb + 64*cc + 8*lc + k);
                        }
                        for (short rr = 0; rr < NLR; rr++) {
                            for (short cc = 0; cc < NCC; cc++) {
                                pacc[rr][cc] = vec<S0,2>(av[rr])*vec<S0,2>(bv[cc]) + pacc[rr][cc];
                            }
                        }
                    }
                }

                lsma += 8*64;
                lsmb += 8*64;
            }

            for (short rr = 0; rr < NLR; rr++) {
                for (short cc = 0; cc < NCC; cc++) {
                    acc[rr][cc] += (float) pacc[rr][cc][0] + (float) pacc[rr][cc][1];
                }
            }
        }
    }

    const int base_row = r0 + (NLR == 8 ? 0 : 32*(sgitg & 1));
    const int base_col = r1 + 32*(NLR == 8 ? sgitg : (sgitg >> 1));

    device float * Dbase = (device float *) dst + im*(int64_t)args.ne1*args.ne0;

    for (short rr = 0; rr < NLR; rr++) {
        for (short cc = 0; cc < NCC; cc++) {
            const int row = base_row + 8*rr + lr;
            const int col = base_col + lc + 8*cc;
            if (row < args.ne0 && col < args.ne1) {
                Dbase[row + (int64_t)col*args.ne0] = acc[rr][cc];
            }
        }
    }
}

typedef decltype(kernel_mul_mm_w64<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, float4x4, 1, dequantize_f32, float, float4x4, float, float2x4>) mul_mm_w64_t;

// The expert GEMM, same treatment as kernel_mul_mm_w64 above: one wave owns a tile instead of
// two virtual 32-lane halves, so a wave covers 32 columns and the column index falls out of the
// lane id. The dispatch already hands this kernel 128 threads flat, which is two whole waves.
template<typename S0, typename S0_4x4, typename S0_8x8, typename S1, typename S1_2x4, typename S1_8x8, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &), typename T0, typename T0_4x4, typename T1, typename T1_2x4, short NLR = 4>
kernel void kernel_mul_mm_id_w64(
        constant ggml_metal_kargs_mul_mm_id & args,
        device const char * src0,
        device const char * src1,
        device const char * htpe,
        device const char * hids,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = NLR == 8 ? 64 : 32;
    constexpr short NCC = 4;

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int im = tgpig.z; // expert
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    device const uint32_t * tpe_u32 = (device const uint32_t *) (htpe);
    device const int32_t  * ids_i32 = (device const int32_t  *) (hids);

    const int32_t neh1 = tpe_u32[im];

    if (r1 >= neh1) {
        return;
    }

    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (    neh1 - r1 < NR1) ? (    neh1 - r1) : NR1;

    // a thread shouldn't load data outside of the matrix
    const short lr0  = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1; // 0 .. 63
    const short lr1  = ((short)tiitg/NL1)      < nr1 ? ((short)tiitg/NL1)      : nr1 - 1;
    const short lr1b = ((short)tiitg/NL1 + 32) < nr1 ? ((short)tiitg/NL1 + 32) : nr1 - 1;

    const short il0 = (tiitg % NL0);

    short il = il0;

    const int id  = ids_i32[im*args.ne21 + r1 + lr1];
    const int idb = NLR == 8 ? ids_i32[im*args.ne21 + r1 + lr1b] : 0;

    // int, not short: the AMD Metal compiler miscompiles device address math
    // built from short indices with ids-derived dynamic bases (reads zeros/junk
    // for every token with id >= 256, i.e. token index >= 128)
    const int i11 = (id % args.ne20) % args.ne11;
    const int i12 = (id / args.ne20);
    const int i13 = 0;

    const int i11b = (idb % args.ne20) % args.ne11;
    const int i12b = (idb / args.ne20);

    const uint64_t offset0 = im*args.nb02 + i13*args.nb03;
    const short    offset1 = il0/nl;

    device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const T1 * y = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*i11
        + args.nb10*iy);

    device const T1 * yb = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12b
        + args.nb11*i11b
        + args.nb10*iy);

    const short lane = tiitg % 64;
    const short lr   = lane % 8; // row within an 8-row block of sa
    const short lc   = lane / 8; // column within the wave's 8-column group

    // an expert narrower than the tile leaves whole waves on padding columns; they still
    // load and hit the barriers, they just skip the products
    const bool sg_live = NLR != 8 || 32*sgitg < nr1;

    float acc[NLR][NCC];
    for (short i = 0; i < NLR; i++) { for (short j = 0; j < NCC; j++) { acc[i][j] = 0.f; } }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        // load data and store to threadgroup memory
        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // no need for dequantization
            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                // row-major within the 8x8 block so (k, k+1) pairs sit contiguous
                *(sa + 64*ib + 8*lx + ly) = loop_k + 16*il + i < args.ne00 ? (S0) *((device T0 *) x + i) : (S0) 0;
            }
        } else {
            S0_4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            const short sy = (tiitg/NL0)/8;
            const short lx = (tiitg/NL0)%8;

            const short ib0 = 8*(2*il0    ) + sy;
            const short ib1 = 8*(2*il0 + 1) + sy;

            *(threadgroup vec<S0, 4> *)(sa + 64*ib0 + 8*lx + 0) = temp_a[0];
            *(threadgroup vec<S0, 4> *)(sa + 64*ib0 + 8*lx + 4) = temp_a[1];
            *(threadgroup vec<S0, 4> *)(sa + 64*ib1 + 8*lx + 0) = temp_a[2];
            *(threadgroup vec<S0, 4> *)(sa + 64*ib1 + 8*lx + 4) = temp_a[3];
        }

        {
            constexpr short ncb = NLR == 8 ? 8 : 4;

            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;
            const short ly = (tiitg/NL1)%8;

            const short ib = ncb*sx + sy;

            const bool colb = NLR == 8 && (short)tiitg/NL1 + 32 < nr1;

            if (FC_mul_mm_bc_inp) {
                for (short i = 0; i < 8; ++i) {
                    *(sb + 64*ib + 8*ly + i) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;
                }
                if (colb) {
                    for (short i = 0; i < 8; ++i) {
                        *(sb + 64*(ib + 4) + 8*ly + i) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) yb + i) : 0;
                    }
                }
            } else {
                *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)(*((device T1_2x4 *) y));
                if (colb) {
                    *(threadgroup S1_2x4 *)(sb + 64*(ib + 4) + 8*ly) = (S1_2x4)(*((device T1_2x4 *) yb));
                }
            }
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y  += NK;
        yb += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const S0 * lsma = (sa + (NLR == 8 ? 0 : 4*64*(sgitg % 2)));
        threadgroup const S1 * lsmb = (sb + (NLR == 8 ? 4*64*sgitg : 0));

        constexpr short bstride = (NLR == 8 ? 8 : 4)*64;

        if (sg_live) {
            // packed rank-1 updates on (k, k+1) pairs; partials flush to float every
            // 8-deep slice when the accumulator cannot hold the products
            if (FC_mul_mm_acc_f32) {
                for (short ik = 0; ik < NK/8; ik++) {
                    vec<float,2> pacc[NLR][NCC];
                    FOR_UNROLL (short k = 0; k < 8; k += 2) {
                        vec<S0,2> av[NLR];
                        vec<S1,2> bv[NCC];
                        for (short rr = 0; rr < NLR; rr++) {
                            av[rr] = *(threadgroup const vec<S0,2> *)(lsma + 64*rr + 8*lr + k);
                        }
                        for (short cc = 0; cc < NCC; cc++) {
                            bv[cc] = *(threadgroup const vec<S1,2> *)(lsmb + 64*cc + 8*lc + k);
                        }
                        for (short rr = 0; rr < NLR; rr++) {
                            for (short cc = 0; cc < NCC; cc++) {
                                const vec<float,2> p = vec<float,2>(av[rr])*vec<float,2>(bv[cc]);
                                pacc[rr][cc] = k == 0 ? p : p + pacc[rr][cc];
                            }
                        }
                    }

                    for (short rr = 0; rr < NLR; rr++) {
                        for (short cc = 0; cc < NCC; cc++) {
                            acc[rr][cc] += (float) pacc[rr][cc][0] + (float) pacc[rr][cc][1];
                        }
                    }

                    lsma += 8*64;
                    lsmb += bstride;
                }
            } else {
                for (short ik = 0; ik < NK/8; ik++) {
                    vec<S0,2> pacc[NLR][NCC];
                    FOR_UNROLL (short k = 0; k < 8; k += 2) {
                        vec<S0,2> av[NLR];
                        vec<S1,2> bv[NCC];
                        for (short rr = 0; rr < NLR; rr++) {
                            av[rr] = *(threadgroup const vec<S0,2> *)(lsma + 64*rr + 8*lr + k);
                        }
                        for (short cc = 0; cc < NCC; cc++) {
                            bv[cc] = *(threadgroup const vec<S1,2> *)(lsmb + 64*cc + 8*lc + k);
                        }
                        for (short rr = 0; rr < NLR; rr++) {
                            for (short cc = 0; cc < NCC; cc++) {
                                const vec<S0,2> p = vec<S0,2>(av[rr])*vec<S0,2>(bv[cc]);
                                pacc[rr][cc] = k == 0 ? p : p + pacc[rr][cc];
                            }
                        }
                    }

                    for (short rr = 0; rr < NLR; rr++) {
                        for (short cc = 0; cc < NCC; cc++) {
                            acc[rr][cc] += (float) pacc[rr][cc][0] + (float) pacc[rr][cc][1];
                        }
                    }

                    lsma += 8*64;
                    lsmb += bstride;
                }
            }
        }
    }

    // the scatter reuses the whole threadgroup buffer, so a 64-column tile goes out in two
    // passes of 32
    for (short p = 0; p < (NLR == 8 ? 2 : 1); p++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float * temp_str = ((threadgroup float *) shmem) + (NLR == 8 ? 0 : 32*(sgitg & 1));

        if (sg_live && (NLR != 8 || sgitg == p)) {
            for (short rr = 0; rr < NLR; rr++) {
                for (short cc = 0; cc < NCC; cc++) {
                    temp_str[(8*rr + lr) + (lc + 8*cc)*NR0] = acc[rr][cc];
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short j = sgitg; j < 32 && j + 32*p < nr1; j += 2) {
            const int id = ids_i32[im*args.ne21 + r1 + 32*p + j];

            // int, not short: the AMD Metal compiler miscompiles device address math
            // built from short indices with ids-derived dynamic bases
            const int ide = id % args.ne20;
            const int idt = id / args.ne20;

            device float  * D  = (device float  *) dst + r0 + (int64_t) ide*args.ne0 + (int64_t) idt*args.ne1*args.ne0;
            device float4 * D4 = (device float4 *) D;

            threadgroup float  * C  = (threadgroup float  *) shmem + j*NR0;
            threadgroup float4 * C4 = (threadgroup float4 *) C;

            int i = lane;
            for (; i < nr0/4; i += 64) {
                *(D4 + i) = *(C4 + i);
            }

            i = (4*(nr0/4)) + lane;
            for (; i < nr0; i += 64) {
                *(D + i) = *(C + i);
            }
        }
    }
}

typedef decltype(kernel_mul_mm_id_w64<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, float4x4, 1, dequantize_f32, float, float4x4, float, float2x4>) mul_mm_id_w64_t;


template [[host_name("kernel_mul_mm_f16_f32_w_w64")]]     kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   float, float2x4, 8>;
template [[host_name("kernel_mul_mm_q5_0_f32_w_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_q5_1_f32_w_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_q8_0_f32_w_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_pq2_0_f32_w_w64")]]   kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_pq2_0,   8,     dequantize_pq2_0,   float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_ptq1_0_f32_w_w64")]]  kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_ptq1_0,  8,     dequantize_ptq1_0,  float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_q2_K_f32_w_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_q3_K_f32_w_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_q5_K_f32_w_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_q6_K_f32_w_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_q4_0_f32_w_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_q4_1_f32_w_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_mxfp4_f32_w_w64")]]   kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_iq4_nl_f32_w_w64")]]  kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_iq4_xs_f32_w_w64")]]  kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_iq3_s_f32_w_w64")]]   kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_iq2_s_f32_w_w64")]]   kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_iq3_xxs_f32_w_w64")]]   kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs,   QK_NL, dequantize_iq3_xxs,   float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_iq2_xs_f32_w_w64")]]   kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,   QK_NL, dequantize_iq2_xs,   float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_iq2_xxs_f32_w_w64")]]   kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs,   QK_NL, dequantize_iq2_xxs,   float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_iq1_m_f32_w_w64")]]   kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_iq1_s_f32_w_w64")]]   kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_q4_K_f32_w_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_f32_f32_w64")]]     kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_f16_f32_w64")]]     kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   float, float2x4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_bf16_f32_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, bfloat4x4,     1,     dequantize_bf16,    bfloat, bfloat4x4, float, float2x4>;
#endif
template [[host_name("kernel_mul_mm_bf16e_f32_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   bf16e4x4,      1,     dequantize_bf16e,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q1_0_f32_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q2_0_f32_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_0,    4,     dequantize_q2_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_pq2_0_f32_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_pq2_0, 8,     dequantize_pq2_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_ptq1_0_f32_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_ptq1_0, 8,     dequantize_ptq1_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q4_0_f32_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q4_1_f32_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q5_0_f32_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q5_1_f32_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q8_0_f32_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_mxfp4_f32_w64")]]   kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q2_K_f32_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q3_K_f32_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q4_K_f32_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q5_K_f32_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q6_K_f32_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq2_xxs_f32_w64")]] kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq2_xs_f32_w64")]]  kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq3_xxs_f32_w64")]] kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq3_s_f32_w64")]]   kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq2_s_f32_w64")]]   kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq1_s_f32_w64")]]   kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq1_m_f32_w64")]]   kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq4_nl_f32_w64")]]  kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq4_xs_f32_w64")]]  kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_tq2_0_f32_w64")]]   kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_tq2_0,   QK_NL, dequantize_tq2_0,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_f32_f16_w64")]]     kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_f16_f16_w64")]]     kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   half, half2x4>;
template [[host_name("kernel_mul_mm_q1_0_f16_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q2_0_f16_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_0,    4,     dequantize_q2_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_pq2_0_f16_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_pq2_0, 8,     dequantize_pq2_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_ptq1_0_f16_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_ptq1_0, 8,     dequantize_ptq1_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q4_0_f16_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q4_1_f16_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q5_0_f16_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q5_1_f16_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q8_0_f16_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_mxfp4_f16_w64")]]   kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q2_K_f16_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q3_K_f16_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q4_K_f16_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q5_K_f16_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q6_K_f16_w64")]]    kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq2_xxs_f16_w64")]] kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq2_xs_f16_w64")]]  kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq3_xxs_f16_w64")]] kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq3_s_f16_w64")]]   kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq2_s_f16_w64")]]   kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq1_s_f16_w64")]]   kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq1_m_f16_w64")]]   kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq4_nl_f16_w64")]]  kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq4_xs_f16_w64")]]  kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_tq2_0_f16_w64")]]   kernel mul_mm_w64_t kernel_mul_mm_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_tq2_0,   QK_NL, dequantize_tq2_0,   float,  float4x4,  half, half2x4>;
// end
template [[host_name("kernel_mul_mm_id_f32_f32_w64")]]     kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_f16_f32_w64")]]     kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   float, float2x4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_id_bf16_f32_w64")]]    kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, bfloat4x4,     1,     dequantize_bf16,    bfloat, bfloat4x4, float, float2x4>;
#endif
template [[host_name("kernel_mul_mm_id_bf16e_f32_w64")]]    kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   bf16e4x4,      1,     dequantize_bf16e,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q1_0_f32_w64")]]    kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q2_0_f32_w64")]]    kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_0,    4,     dequantize_q2_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_pq2_0_f32_w64")]]    kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_pq2_0, 8,     dequantize_pq2_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_ptq1_0_f32_w64")]]    kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_ptq1_0, 8,     dequantize_ptq1_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q4_0_f32_w64")]]    kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q4_1_f32_w64")]]    kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q5_0_f32_w64")]]    kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q5_1_f32_w64")]]    kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q8_0_f32_w64")]]    kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_mxfp4_f32_w64")]]   kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q2_K_f32_w64")]]    kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q3_K_f32_w64")]]    kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q4_K_f32_w64")]]    kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q5_K_f32_w64")]]    kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q6_K_f32_w64")]]    kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f32_w64")]] kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xs_f32_w64")]]  kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq3_xxs_f32_w64")]] kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq3_s_f32_w64")]]   kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq2_s_f32_w64")]]   kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq1_s_f32_w64")]]   kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq1_m_f32_w64")]]   kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq4_nl_f32_w64")]]  kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq4_xs_f32_w64")]]  kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_tq2_0_f32_w64")]]   kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_tq2_0,   QK_NL, dequantize_tq2_0,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_f16_f32_w_w64")]]    kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_q4_0_f32_w_w64")]]   kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_q4_1_f32_w_w64")]]   kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_q5_0_f32_w_w64")]]   kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_q5_1_f32_w_w64")]]   kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_q8_0_f32_w_w64")]]   kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_mxfp4_f32_w_w64")]]  kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_q2_K_f32_w_w64")]]   kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_q3_K_f32_w_w64")]]   kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_q4_K_f32_w_w64")]]   kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_q5_K_f32_w_w64")]]   kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_q6_K_f32_w_w64")]]   kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_iq3_s_f32_w_w64")]]  kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_iq2_s_f32_w_w64")]]  kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_iq3_xxs_f32_w_w64")]]  kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs,   QK_NL, dequantize_iq3_xxs,   float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_iq2_xs_f32_w_w64")]]  kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,   QK_NL, dequantize_iq2_xs,   float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f32_w_w64")]]  kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs,   QK_NL, dequantize_iq2_xxs,   float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_iq1_m_f32_w_w64")]]  kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_iq1_s_f32_w_w64")]]  kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_iq4_nl_f32_w_w64")]] kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_iq4_xs_f32_w_w64")]] kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_f32_f16_w64")]]     kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_f16_f16_w64")]]     kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   half, half2x4>;
template [[host_name("kernel_mul_mm_id_q1_0_f16_w64")]]    kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q2_0_f16_w64")]]    kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_0,    4,     dequantize_q2_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_pq2_0_f16_w64")]]    kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_pq2_0, 8,     dequantize_pq2_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_ptq1_0_f16_w64")]]    kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_ptq1_0, 8,     dequantize_ptq1_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q4_0_f16_w64")]]    kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q4_1_f16_w64")]]    kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q5_0_f16_w64")]]    kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q5_1_f16_w64")]]    kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q8_0_f16_w64")]]    kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_mxfp4_f16_w64")]]   kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q2_K_f16_w64")]]    kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q3_K_f16_w64")]]    kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q4_K_f16_w64")]]    kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q5_K_f16_w64")]]    kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q6_K_f16_w64")]]    kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f16_w64")]] kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xs_f16_w64")]]  kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq3_xxs_f16_w64")]] kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq3_s_f16_w64")]]   kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq2_s_f16_w64")]]   kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq1_s_f16_w64")]]   kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq1_m_f16_w64")]]   kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq4_nl_f16_w64")]]  kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq4_xs_f16_w64")]]  kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_tq2_0_f16_w64")]]   kernel mul_mm_id_w64_t kernel_mul_mm_id_w64<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_tq2_0,   QK_NL, dequantize_tq2_0,   float,  float4x4,  half, half2x4>;

// ---------------------------------------------------------------------------------------------
// q4_K / q6_K x f32 prefill GEMM, one wave per 64x64 tile.
//
// One simdgroup owns one tile: 8 rows per lane over 8 row groups and 8 columns per lane over 8
// column groups, so a lane keeps 64 accumulators and reads the weight vector once per 8 columns
// instead of once per 4. The next panel is dequantized into registers before the fma loop that
// consumes the current one and only written to the shared buffer afterwards, so the global loads
// and the nibble unpack of panel i+1 overlap the arithmetic of panel i without a second buffer.
//
// Measured standalone (12288x1088x4096, q4_K x f32): 2970 GFLOPS, against 2236 for the same tile
// with the loads left behind the barrier and 2030 for the two-wave 4-column tile.
//
// The staging buffers are KT*64 floats for A and KT*64 for B: 8 KiB, single buffered.
// ---------------------------------------------------------------------------------------------

#define MW_Q4K_BYTES 144
#define MW_Q6K_BYTES 210
#define MW_QK_K      256

#define MW_NCOLS 8
#define MW_KT    16
#define MW_TW    (MW_NCOLS*8)            // 64 columns per wave
#define MW_NV4   (MW_KT*MW_NCOLS/32)     // 4 float4 per lane on the B side
#define MW_KGRP  (MW_KT/4)               // 4 lanes share a B column
#define MW_CSTEP (MW_TW/MW_NV4)          // 16 columns apart
#define MW_BUF   (MW_KT*(64 + MW_TW))    // 2048 floats

// 16 bytes at an arbitrary byte offset: read the two aligned words that cover it and shift
static inline void mw_span16(device const uchar * p, uint al,
                             thread uint & w0, thread uint & w1,
                             thread uint & w2, thread uint & w3) {
    const device uint4 * q = (const device uint4 *) (p - al);
    const uint4 a = q[0], b = q[1];
    const ulong u0 = ((ulong) a.y << 32) | (ulong) a.x;
    const ulong u1 = ((ulong) a.w << 32) | (ulong) a.z;
    const ulong u2 = ((ulong) b.y << 32) | (ulong) b.x;
    const ulong u3 = ((ulong) b.w << 32) | (ulong) b.z;
    ulong lo, hi;
    if (al < 8u) {
        const uint s = al*8u;
        lo = s ? ((u0 >> s) | (u1 << (64u - s))) : u0;
        hi = s ? ((u1 >> s) | (u2 << (64u - s))) : u1;
    } else {
        const uint s = (al - 8u)*8u;
        lo = s ? ((u1 >> s) | (u2 << (64u - s))) : u1;
        hi = s ? ((u2 >> s) | (u3 << (64u - s))) : u2;
    }
    w0 = (uint) lo;         w1 = (uint) (lo >> 32);
    w2 = (uint) hi;         w3 = (uint) (hi >> 32);
}

// one 16-k panel of a q4_K row: nibbles into q[], the panel's (scale, min) into dsc/dbia
// q4_K panel split into issue and decode: the two 16-byte loads for panel i+1 are put on the bus
// before the fma loop of panel i and only turned into floats after it, so the wait sits behind a
// thousand fmas instead of in front of them. Measured 12288x1228x4096 q4_K, with the tile order
// already fixed: 2789 GFLOPS split, 2555 unsplit.
// fresh = the panel starts a 256-k super-block, so its 16-byte header has to be read; otherwise the
// header in registers is still the right one.  The header is a third of what a lane fetches per
// panel and it lives in the same 128-byte line as the value window, so this saves four bytes of
// instruction stream rather than four bytes of traffic -- measured +2.6% wide, +1.6% deep.
static inline void mw_issue_a4(device const char * wrow, int loop_k, bool fresh,
                               thread uint4 & h, thread uint4 & v) {
    const int bi = loop_k / MW_QK_K;
    const int j  = (loop_k % MW_QK_K) / 32;
    const int ko = loop_k % 32;
    device const uchar * blk = (device const uchar *) (wrow + (uint64_t) bi*MW_Q4K_BYTES);
    if (fresh) h = *(device const uint4 *) blk;
    v = *(device const uint4 *) (blk + 16 + 32*(j/2) + ko);
}

static inline void mw_decode_a4(uint4 h, uint4 v, int loop_k, bool rowok,
                                thread float & dsc, thread float & dbia, thread uchar * q) {
    const int j = (loop_k % MW_QK_K) / 32;

    const float d  = (float) as_type<half>((ushort) (h.x & 0xFFFFu));
    const float dm = (float) as_type<half>((ushort) (h.x >> 16));

    float sc, mn;
    if (j < 4) {
        const uint sj  = (h.y >> (8*j)) & 0xFFu;
        const uint sj4 = (h.z >> (8*j)) & 0xFFu;
        sc = (float) (sj  & 63u);
        mn = (float) (sj4 & 63u);
    } else {
        const int  o   = j - 4;
        const uint sj4 = (h.w >> (8*o)) & 0xFFu;
        const uint sjm = (h.y >> (8*o)) & 0xFFu;
        const uint sj  = (h.z >> (8*o)) & 0xFFu;
        sc = (float) ((sj4 & 0x0Fu) | ((sjm >> 6) << 4));
        mn = (float) ((sj4 >>    4) | ((sj  >> 6) << 4));
    }
    dsc  = rowok ? d*sc    : 0.f;
    dbia = rowok ? -dm*mn  : 0.f;

    const uint sh = (j & 1) ? 4u : 0u;
    v.x >>= sh; v.y >>= sh; v.z >>= sh; v.w >>= sh;
    q[ 0] = (uchar) (v.x & 0xFu);          q[ 1] = (uchar) ((v.x >>  8) & 0xFu);
    q[ 2] = (uchar) ((v.x >> 16) & 0xFu);  q[ 3] = (uchar) ((v.x >> 24) & 0xFu);
    q[ 4] = (uchar) (v.y & 0xFu);          q[ 5] = (uchar) ((v.y >>  8) & 0xFu);
    q[ 6] = (uchar) ((v.y >> 16) & 0xFu);  q[ 7] = (uchar) ((v.y >> 24) & 0xFu);
    q[ 8] = (uchar) (v.z & 0xFu);          q[ 9] = (uchar) ((v.z >>  8) & 0xFu);
    q[10] = (uchar) ((v.z >> 16) & 0xFu);  q[11] = (uchar) ((v.z >> 24) & 0xFu);
    q[12] = (uchar) (v.w & 0xFu);          q[13] = (uchar) ((v.w >>  8) & 0xFu);
    q[14] = (uchar) ((v.w >> 16) & 0xFu);  q[15] = (uchar) ((v.w >> 24) & 0xFu);
}

static inline void mw_load_a4(device const char * wrow, int loop_k, bool rowok,
                              thread float & dsc, thread float & dbia, thread uchar * q) {
    const int bi = loop_k / MW_QK_K;
    const int j  = (loop_k % MW_QK_K) / 32;   // 32-value group inside the super-block
    const int ko = loop_k % 32;               // half of that group
    device const uchar * blk = (device const uchar *) (wrow + (uint64_t) bi*MW_Q4K_BYTES);

    const uint4 h = *(device const uint4 *) blk;
    const float d  = (float) as_type<half>((ushort) (h.x & 0xFFFFu));
    const float dm = (float) as_type<half>((ushort) (h.x >> 16));

    float sc, mn;
    if (j < 4) {
        const uint sj  = (h.y >> (8*j)) & 0xFFu;
        const uint sj4 = (h.z >> (8*j)) & 0xFFu;
        sc = (float) (sj  & 63u);
        mn = (float) (sj4 & 63u);
    } else {
        const int  o   = j - 4;
        const uint sj4 = (h.w >> (8*o)) & 0xFFu;
        const uint sjm = (h.y >> (8*o)) & 0xFFu;
        const uint sj  = (h.z >> (8*o)) & 0xFFu;
        sc = (float) ((sj4 & 0x0Fu) | ((sjm >> 6) << 4));
        mn = (float) ((sj4 >>    4) | ((sj  >> 6) << 4));
    }
    dsc  = rowok ? d*sc    : 0.f;
    dbia = rowok ? -dm*mn  : 0.f;

    // 32 bytes carry 64 values: the low nibbles are the first 32, the high nibbles the next 32
    const int base = 16 + 32*(j/2) + ko;
    const uint sh  = (j & 1) ? 4u : 0u;

    uint4 v = *(device const uint4 *) (blk + base);
    v.x >>= sh; v.y >>= sh; v.z >>= sh; v.w >>= sh;
    q[ 0] = (uchar) (v.x & 0xFu);          q[ 1] = (uchar) ((v.x >>  8) & 0xFu);
    q[ 2] = (uchar) ((v.x >> 16) & 0xFu);  q[ 3] = (uchar) ((v.x >> 24) & 0xFu);
    q[ 4] = (uchar) (v.y & 0xFu);          q[ 5] = (uchar) ((v.y >>  8) & 0xFu);
    q[ 6] = (uchar) ((v.y >> 16) & 0xFu);  q[ 7] = (uchar) ((v.y >> 24) & 0xFu);
    q[ 8] = (uchar) (v.z & 0xFu);          q[ 9] = (uchar) ((v.z >>  8) & 0xFu);
    q[10] = (uchar) ((v.z >> 16) & 0xFu);  q[11] = (uchar) ((v.z >> 24) & 0xFu);
    q[12] = (uchar) (v.w & 0xFu);          q[13] = (uchar) ((v.w >>  8) & 0xFu);
    q[14] = (uchar) ((v.w >> 16) & 0xFu);  q[15] = (uchar) ((v.w >> 24) & 0xFu);
}

// the same panel of a q6_K row. Its 6-bit values live in two 128-byte halves that a 16-value
// span reaches through a byte-aligned window, and the value carries a +32 bias.
static inline void mw_load_a6(device const char * wrow, int loop_k, bool rowok,
                              thread float & dsc, thread float & dbia, thread uchar * q) {
    const int bi = loop_k / MW_QK_K;
    const int j  = (loop_k % MW_QK_K) / 32;
    const int ko = loop_k % 32;
    device const uchar * blk = (device const uchar *) (wrow + (uint64_t) bi*MW_Q6K_BYTES);

    const int gp = j >> 2;
    const int kk = ko;
    const int h  = ((j & 3) << 1) | (kk >> 4);
    const int qlo = 64*gp + 32*((h >> 1) & 1) + 16*(h & 1) + (kk & 15);
    const int qho = 128 + 32*gp + 16*(h & 1) + (kk & 15);
    const int qs  = 2*(h >> 1);
    const int nib = (h & 4) ? 4 : 0;

    const float d  = (float) *(device const half *) (blk + 208);
    const float sc = (float) *(device const char *) (blk + 192 + 8*gp + h);
    dsc  = rowok ? d*sc    : 0.f;
    dbia = rowok ? -32.f*dsc : 0.f;

    uint l0, l1, l2, l3, h0, h1, h2, h3;
    const uint al = (uint) ((uint64_t) blk & 15u);
    mw_span16(blk + qlo, al, l0, l1, l2, l3);
    mw_span16(blk + qho, al, h0, h1, h2, h3);
    const uint lw[4] = { l0, l1, l2, l3 };
    const uint hw[4] = { h0, h1, h2, h3 };
    #pragma unroll
    for (int i = 0; i < MW_KT; i++) {
        const uint qlb = (lw[i >> 2] >> (8*(i & 3))) & 0xFFu;
        const uint qhb = (hw[i >> 2] >> (8*(i & 3))) & 0xFFu;
        q[i] = (uchar) (((qlb >> nib) & 0xFu) | (((qhb >> qs) & 3u) << 4));
    }
}

template<int QT>
static inline void mw_load_a(device const char * wrow, int loop_k, bool rowok,
                             thread float & dsc, thread float & dbia, thread uchar * q) {
    if (QT == 1) mw_load_a6(wrow, loop_k, rowok, dsc, dbia, q);
    else         mw_load_a4(wrow, loop_k, rowok, dsc, dbia, q);
}

static inline void mw_store_a(threadgroup float * sa, const thread uchar * q, int arow,
                              float dsc, float dbia) {
    #pragma unroll
    for (int i = 0; i < MW_KT; i++) sa[i*64 + arow] = dsc*(float) q[i] + dbia;
}

static inline void mw_load_b(device const char * abase, uint64_t nb11, int r1, int loop_k,
                             int bcol0, int bkg, int ne1, thread float4 * bq) {
    #pragma unroll
    for (int i = 0; i < MW_NV4; i++) {
        const int col  = r1 + bcol0 + MW_CSTEP*i;
        const int acol = min(col, ne1 - 1);
        const float z  = col < ne1 ? 1.f : 0.f;
        device const float * src = (device const float *) (abase + nb11*(uint64_t) acol) + loop_k + bkg;
        bq[i] = *(device const float4 *) src * z;
    }
}

static inline void mw_store_b(threadgroup float * sb, const thread float4 * bq, int bkg,
                              int bcol0) {
    #pragma unroll
    for (int i = 0; i < MW_NV4; i++) {
        const int col = bcol0 + MW_CSTEP*i;
        sb[(bkg + 0)*MW_TW + col] = bq[i].x;
        sb[(bkg + 1)*MW_TW + col] = bq[i].y;
        sb[(bkg + 2)*MW_TW + col] = bq[i].z;
        sb[(bkg + 3)*MW_TW + col] = bq[i].w;
    }
}

template<int QT>
static inline void mw_tile(
        const constant ggml_metal_kargs_mul_mm & args,
        device const char * src0,
        device const char * src1,
        device char * dst,
        threadgroup float * shmem,
        uint3 tgpig,
        uint3 tgpg,
        int tiitg) {

    const int ne00 = args.ne00;
    const int ne0  = args.ne0;
    const int ne1  = args.ne1;

    // same tile order as kernel_mul_mm_w64: row tiles in the inner loop, so one column tile's
    // activation panel is the thing that stays resident.
    int tg_x = tgpig.x;
    int tg_y = tgpig.y;
    {
        const uint blocks_x = tgpg.x;
        const uint blocks_y = tgpg.y;
        if (blocks_x > 1 && blocks_x < blocks_y) {
            const uint lin = tgpig.y*blocks_x + tgpig.x;
            tg_x = lin / blocks_y;
            tg_y = lin % blocks_y;
        }
    }

    const int r0 = tg_y*64;
    const int r1 = tg_x*MW_TW;

    const int im = tgpig.z;
    const int i12 = im % args.ne12;
    const int i13 = im / args.ne12;

    const uint64_t off0 = (uint64_t) (i12/args.r2)*args.nb02 + (uint64_t) (i13/args.r3)*args.nb03;
    device const char * wbase = src0 + off0;
    device const char * abase = src1 + (uint64_t) args.nb13*i13 + (uint64_t) args.nb12*i12;

    const int lane  = tiitg;
    const int row0  = 8*(lane >> 3);
    const int col0  = MW_NCOLS*(lane & 7);
    const int bkg   = 4*(lane & (MW_KGRP - 1));
    const int bcol0 = lane / MW_KGRP;

    const int  grow  = min(r0 + lane, ne0 - 1);
    const bool rowok = (r0 + lane) < ne0;
    device const char * wrow = wbase + (uint64_t) grow*args.nb01;

    threadgroup float * sa = shmem;
    threadgroup float * sb = sa + MW_KT*64;

    vec<float, 8> acc[MW_NCOLS];
    #pragma unroll
    for (int c = 0; c < MW_NCOLS; c++) acc[c] = 0.f;

    float dsc, dbia;
    uchar qa[MW_KT];
    float4 bq[MW_NV4];
    const int nb = ne00 / MW_KT;

    // q6_K keeps the one-piece load: its panel needs two byte-aligned windows plus the int8 scale,
    // which does not fit the same issue/decode split without costing more than it saves.
    constexpr bool SP = (QT == 0);
    uint4 rawH, rawV;

    if (SP) mw_issue_a4(wrow, 0, true, rawH, rawV);
    else    mw_load_a<QT>(wrow, 0, rowok, dsc, dbia, qa);
    mw_load_b(abase, args.nb11, r1, 0, bcol0, bkg, ne1, bq);
    if (SP) mw_decode_a4(rawH, rawV, 0, rowok, dsc, dbia, qa);
    mw_store_a(sa, qa, lane, dsc, dbia);
    mw_store_b(sb, bq, bkg, bcol0);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = 0; i < nb; i++) {
        const bool more = (i + 1) < nb;

        // the next panel's loads are issued before the fma loop below and decoded after it, so the
        // memory wait lands behind the arithmetic instead of in front of it
        if (more) {
            if (SP) mw_issue_a4(wrow, (i + 1)*MW_KT, (((i + 1) & (MW_QK_K/MW_KT - 1)) == 0), rawH, rawV);
            else    mw_load_a<QT>(wrow, (i + 1)*MW_KT, rowok, dsc, dbia, qa);
            mw_load_b(abase, args.nb11, r1, (i + 1)*MW_KT, bcol0, bkg, ne1, bq);
        }

        #pragma unroll
        for (int k = 0; k < MW_KT; k++) {
            const vec<float, 8> av = *(threadgroup const vec<float, 8> *) (sa + k*64 + row0);
            #pragma unroll
            for (int g = 0; g < MW_NCOLS/4; g++) {
                const float4 b = *(threadgroup const float4 *) (sb + k*MW_TW + col0 + 4*g);
                acc[4*g + 0] += av*b.x;
                acc[4*g + 1] += av*b.y;
                acc[4*g + 2] += av*b.z;
                acc[4*g + 3] += av*b.w;
            }
        }

        if (more) {
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (SP) mw_decode_a4(rawH, rawV, (i + 1)*MW_KT, rowok, dsc, dbia, qa);
            mw_store_a(sa, qa, lane, dsc, dbia);
            mw_store_b(sb, bq, bkg, bcol0);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    device float * D = (device float *) dst + (int64_t) im*ne1*ne0 + r0;
    #pragma unroll
    for (int c = 0; c < MW_NCOLS; c++) {
        const int col = r1 + col0 + c;
        if (col >= ne1) continue;
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            const int row = row0 + r;
            if (r0 + row < ne0) D[row + (int64_t) col*ne0] = acc[c][r];
        }
    }
}

#define MW_KERNEL(NAME, QT)                                                         \
kernel void NAME(                                                                   \
        constant ggml_metal_kargs_mul_mm & args,                                    \
        device const char * src0,                                                   \
        device const char * src1,                                                   \
        device       char * dst,                                                    \
        threadgroup  char * shmem [[threadgroup(0)]],                               \
        uint3  tgpig[[threadgroup_position_in_grid]],                               \
        uint3  tgpg[[threadgroups_per_grid]],                                       \
        ushort tiitg[[thread_index_in_threadgroup]]) {                              \
    if (tiitg >= 64) return;                                                        \
    mw_tile<QT>(args, src0, src1, dst, (threadgroup float *) shmem,                  \
                tgpig, tgpg, (int) tiitg);                                          \
}

MW_KERNEL(kernel_mul_mm_mw,  0)
MW_KERNEL(kernel_mul_mm_mw6, 1)
