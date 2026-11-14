#import "ggml-metal-context.h"

#import "ggml-impl.h"
#import "ggml-backend-impl.h"

#import "ggml-metal-impl.h"
#import "ggml-metal-common.h"
#import "ggml-metal-ops.h"
#import "ggml-metal-fusion.h"


#import <Foundation/Foundation.h>

#import <Metal/Metal.h>

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// max number of MTLCommandBuffer used to submit a graph for processing
#define GGML_METAL_MAX_COMMAND_BUFFERS 8
#define GGML_METAL_MAX_PENDING_WAITS   8
#define GGML_METAL_MAX_PREFETCH_STAGES 8

#include <stdatomic.h>
#include <time.h>

// TOSH_CB_PROFILE=1: per-command-buffer GPU accounting, bucketed by graph size.
// Off by default, and it attaches no completion handler when off.
#define TOSH_CBP_MAX_BUCKETS 24

struct tosh_cbp_bucket {
    _Atomic int      n_nodes;    // 0 = free slot
    _Atomic uint64_t n_graphs;
    _Atomic uint64_t n_cbs;
    _Atomic uint64_t gpu_ns;     // sum of GPUEndTime - GPUStartTime
    _Atomic uint64_t ker_ns;     // sum of kernelEndTime - kernelStartTime
    _Atomic uint64_t enc_ns;     // CPU time inside ggml_metal_graph_compute
    _Atomic uint64_t span_ns;    // per graph: max(GPUEndTime) - min(GPUStartTime)
    _Atomic uint64_t gpu_min_ns; // busiest/idlest single graph, to expose the warmup
    _Atomic uint64_t gpu_max_ns;
};

// CPU side of synchronize, kept apart from the encode so a blocked wait is not read
// as scheduler waste
static _Atomic uint64_t g_cbp_wait_ns;   // blocked in waitUntilCompleted
static _Atomic uint64_t g_cbp_drain_ns;  // blocked draining staged uploads
static _Atomic uint64_t g_cbp_sync_n;
static _Atomic uint64_t g_cbp_ops_ns;    // encoding the ops of a command buffer
static _Atomic uint64_t g_cbp_commit_ns; // the commit call itself
static _Atomic uint64_t g_cbp_create_ns; // creating and enqueuing the buffers of a graph

static struct tosh_cbp_bucket g_cbp[TOSH_CBP_MAX_BUCKETS];
static _Atomic int g_cbp_state = 0; // 0 unknown, 1 off, 2 on

static bool tosh_cbp_on(void) {
    int st = atomic_load_explicit(&g_cbp_state, memory_order_relaxed);
    if (st == 0) {
        const char * e = TOSH_ENV("TOSH_CB_PROFILE");
        st = (e != NULL && strcmp(e, "0") != 0) ? 2 : 1;
        atomic_store_explicit(&g_cbp_state, st, memory_order_relaxed);
    }
    return st == 2;
}

static struct tosh_cbp_bucket * tosh_cbp_bucket_for(int n_nodes) {
    for (int i = 0; i < TOSH_CBP_MAX_BUCKETS; ++i) {
        int cur = atomic_load_explicit(&g_cbp[i].n_nodes, memory_order_relaxed);
        if (cur == n_nodes) {
            return &g_cbp[i];
        }
        if (cur == 0) {
            int expected = 0;
            if (atomic_compare_exchange_strong(&g_cbp[i].n_nodes, &expected, n_nodes)) {
                return &g_cbp[i];
            }
            if (atomic_load_explicit(&g_cbp[i].n_nodes, memory_order_relaxed) == n_nodes) {
                return &g_cbp[i];
            }
        }
    }
    return NULL;
}

// per-graph GPU timeline, filled by the completion handlers of that graph
struct tosh_cbp_graph {
    struct tosh_cbp_bucket * bucket;
    _Atomic uint64_t t_first_ns;  // min GPUStartTime, ns
    _Atomic uint64_t t_last_ns;   // max GPUEndTime, ns
    _Atomic uint64_t gpu_ns;      // this graph's own GPU busy time
    _Atomic int      n_left;      // command buffers still running
};

static void tosh_cbp_attach(id<MTLCommandBuffer> cb, struct tosh_cbp_graph * g) {
    if (g == NULL) {
        return;
    }
    atomic_fetch_add_explicit(&g->n_left, 1, memory_order_relaxed);
    [cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
        const double t0 = [b GPUStartTime];
        const double t1 = [b GPUEndTime];
        const double k0 = [b kernelStartTime];
        const double k1 = [b kernelEndTime];

        atomic_fetch_add_explicit(&g->bucket->n_cbs, 1, memory_order_relaxed);
        if (t1 > t0) {
            const uint64_t d = (uint64_t)((t1 - t0)*1e9);
            atomic_fetch_add_explicit(&g->bucket->gpu_ns, d, memory_order_relaxed);
            atomic_fetch_add_explicit(&g->gpu_ns,         d, memory_order_relaxed);
        }
        if (k1 > k0) {
            atomic_fetch_add_explicit(&g->bucket->ker_ns, (uint64_t)((k1 - k0)*1e9), memory_order_relaxed);
        }

        const uint64_t s_ns = (uint64_t)(t0*1e9);
        const uint64_t e_ns = (uint64_t)(t1*1e9);
        uint64_t cur = atomic_load_explicit(&g->t_first_ns, memory_order_relaxed);
        while (s_ns < cur || cur == 0) {
            if (atomic_compare_exchange_weak(&g->t_first_ns, &cur, s_ns)) break;
        }
        cur = atomic_load_explicit(&g->t_last_ns, memory_order_relaxed);
        while (e_ns > cur) {
            if (atomic_compare_exchange_weak(&g->t_last_ns, &cur, e_ns)) break;
        }

        if (atomic_fetch_sub_explicit(&g->n_left, 1, memory_order_acq_rel) == 1) {
            const uint64_t a = atomic_load_explicit(&g->t_first_ns, memory_order_relaxed);
            const uint64_t z = atomic_load_explicit(&g->t_last_ns,  memory_order_relaxed);
            if (z > a) {
                atomic_fetch_add_explicit(&g->bucket->span_ns, z - a, memory_order_relaxed);
            }
            const uint64_t own = atomic_load_explicit(&g->gpu_ns, memory_order_relaxed);
            uint64_t m = atomic_load_explicit(&g->bucket->gpu_min_ns, memory_order_relaxed);
            while (m == 0 || own < m) {
                if (atomic_compare_exchange_weak(&g->bucket->gpu_min_ns, &m, own)) break;
            }
            m = atomic_load_explicit(&g->bucket->gpu_max_ns, memory_order_relaxed);
            while (own > m) {
                if (atomic_compare_exchange_weak(&g->bucket->gpu_max_ns, &m, own)) break;
            }
            free(g);
        }
    }];
}

// definidos en ggml-metal-device.m
extern _Atomic uint64_t g_tosh_d2h_ns, g_tosh_d2h_bytes, g_tosh_d2h_n;
extern _Atomic uint64_t g_tosh_h2d_ns, g_tosh_h2d_bytes, g_tosh_h2d_n;

static void tosh_cbp_dump_dev(ggml_metal_device_t dev) {
    if (!tosh_cbp_on()) {
        return;
    }
    {
        const uint64_t dn = atomic_load_explicit(&g_tosh_d2h_n, memory_order_relaxed);
        const uint64_t hn = atomic_load_explicit(&g_tosh_h2d_n, memory_order_relaxed);
        fprintf(stderr, "TOSH_CBP  readback: %llu lecturas, %.2f MiB, %.1f ms | subidas: %llu, %.2f MiB, %.1f ms\n",
                (unsigned long long) dn,
                atomic_load_explicit(&g_tosh_d2h_bytes, memory_order_relaxed)/1048576.0,
                atomic_load_explicit(&g_tosh_d2h_ns,    memory_order_relaxed)/1e6,
                (unsigned long long) hn,
                atomic_load_explicit(&g_tosh_h2d_bytes, memory_order_relaxed)/1048576.0,
                atomic_load_explicit(&g_tosh_h2d_ns,    memory_order_relaxed)/1e6);
        atomic_store_explicit(&g_tosh_d2h_n, 0, memory_order_relaxed);
        atomic_store_explicit(&g_tosh_d2h_bytes, 0, memory_order_relaxed);
        atomic_store_explicit(&g_tosh_d2h_ns, 0, memory_order_relaxed);
        atomic_store_explicit(&g_tosh_h2d_n, 0, memory_order_relaxed);
        atomic_store_explicit(&g_tosh_h2d_bytes, 0, memory_order_relaxed);
        atomic_store_explicit(&g_tosh_h2d_ns, 0, memory_order_relaxed);
    }
    {
        const uint64_t n = atomic_load_explicit(&g_cbp_sync_n, memory_order_relaxed);
        fprintf(stderr, "TOSH_CBP  pid=%d  syncs=%llu  cpu_wait=%.1f  upload_drain=%.1f  "
                "encode_ops=%.1f  commit=%.1f  create=%.1f  (ms totales)\n",
                getpid(), (unsigned long long) n,
                atomic_load_explicit(&g_cbp_wait_ns,   memory_order_relaxed)/1e6,
                atomic_load_explicit(&g_cbp_drain_ns,  memory_order_relaxed)/1e6,
                atomic_load_explicit(&g_cbp_ops_ns,    memory_order_relaxed)/1e6,
                atomic_load_explicit(&g_cbp_commit_ns, memory_order_relaxed)/1e6,
                atomic_load_explicit(&g_cbp_create_ns, memory_order_relaxed)/1e6);
        atomic_store_explicit(&g_cbp_wait_ns,   0, memory_order_relaxed);
        atomic_store_explicit(&g_cbp_drain_ns,  0, memory_order_relaxed);
        atomic_store_explicit(&g_cbp_sync_n,    0, memory_order_relaxed);
        atomic_store_explicit(&g_cbp_ops_ns,    0, memory_order_relaxed);
        atomic_store_explicit(&g_cbp_commit_ns, 0, memory_order_relaxed);
        atomic_store_explicit(&g_cbp_create_ns, 0, memory_order_relaxed);
    }
    fprintf(stderr, "TOSH_CBP  %8s %8s %8s %12s %12s %12s %12s %12s %12s\n",
            "n_nodes", "graphs", "cbs", "gpu_us/g", "gpu_us_min", "gpu_us_max", "kern_us/g", "span_us/g", "encode_us/g");
    for (int i = 0; i < TOSH_CBP_MAX_BUCKETS; ++i) {
        const int nn = atomic_load_explicit(&g_cbp[i].n_nodes, memory_order_relaxed);
        if (nn == 0) {
            continue;
        }
        const uint64_t ng = atomic_load_explicit(&g_cbp[i].n_graphs, memory_order_relaxed);
        if (ng == 0) {
            continue;
        }
        fprintf(stderr, "TOSH_CBP  %8d %8llu %8llu %12.1f %12.1f %12.1f %12.1f %12.1f %12.1f\n", nn,
                (unsigned long long) ng,
                (unsigned long long) atomic_load_explicit(&g_cbp[i].n_cbs, memory_order_relaxed),
                atomic_load_explicit(&g_cbp[i].gpu_ns,     memory_order_relaxed)/1000.0/ng,
                atomic_load_explicit(&g_cbp[i].gpu_min_ns, memory_order_relaxed)/1000.0,
                atomic_load_explicit(&g_cbp[i].gpu_max_ns, memory_order_relaxed)/1000.0,
                atomic_load_explicit(&g_cbp[i].ker_ns,     memory_order_relaxed)/1000.0/ng,
                atomic_load_explicit(&g_cbp[i].span_ns,    memory_order_relaxed)/1000.0/ng,
                atomic_load_explicit(&g_cbp[i].enc_ns,     memory_order_relaxed)/1000.0/ng);

        // reset so each dump covers only the graphs since the previous one
        atomic_store_explicit(&g_cbp[i].n_graphs,   0, memory_order_relaxed);
        atomic_store_explicit(&g_cbp[i].n_cbs,      0, memory_order_relaxed);
        atomic_store_explicit(&g_cbp[i].gpu_ns,     0, memory_order_relaxed);
        atomic_store_explicit(&g_cbp[i].ker_ns,     0, memory_order_relaxed);
        atomic_store_explicit(&g_cbp[i].span_ns,    0, memory_order_relaxed);
        atomic_store_explicit(&g_cbp[i].enc_ns,     0, memory_order_relaxed);
        atomic_store_explicit(&g_cbp[i].gpu_min_ns, 0, memory_order_relaxed);
        atomic_store_explicit(&g_cbp[i].gpu_max_ns, 0, memory_order_relaxed);
    }
    fflush(stderr);
}

struct ggml_metal_command_buffer {
    id<MTLCommandBuffer> obj;
};

struct ggml_metal {
    char name[128];

    ggml_metal_device_t  dev;
    ggml_metal_library_t lib;

    // shared device queue, or an owned one for secondary (prefetch) contexts
    // so their uploads overlap the primary context's compute
    id<MTLCommandQueue> queue;
    bool owns_queue;
    bool is_prefetch;

    // Bounded host-visible source ring for expert uploads. Keeping one no-copy
    // MTLBuffer per CPU tensor can pin many GiB of host pages on discrete GPUs.
    id<MTLBuffer>        prefetch_stages[GGML_METAL_MAX_PREFETCH_STAGES];
    id<MTLCommandBuffer> prefetch_stage_cmds[GGML_METAL_MAX_PREFETCH_STAGES];
    int prefetch_stage_count;
    int prefetch_stage_cur;

    ggml_metal_event_t ev_cpy; // for async copies
    ggml_metal_event_t ev_sync; // destination completion signal

    id<MTLCounterSampleBuffer> ct_buf;   // trace only, TOSH_MGPU_CTIME
    id<MTLSharedEvent>         ev_wait_dummy;   // benchmark only, TOSH_MGPU_WAIT_SATISFIED
    _Atomic uint64_t           ct_idx;

    // Synchronization and resources shared by both cross-device copy paths.
    struct ggml_metal_xdev_link {
        ggml_metal_device_t  src_dev;
        void *               host;
        size_t               cap;
        id<MTLBuffer>        wrap_src, wrap_dst;
        // Retaining the source keeps its cached remote view valid.
        id<MTLBuffer>        peer_src, peer_view;
        id<MTLSharedEvent>   ev_ready, ev_done;
        uint64_t             seq;
        bool                 peer_logged;
    } xlinks[4];
    int n_xlinks;

    dispatch_queue_t d_queue;

    // additional, inference-time compiled pipelines
    ggml_metal_pipelines_t pipelines_ext;

    bool use_concurrency;
    bool use_graph_optimize;

    int debug_graph;

    struct ggml_metal_fusion_info * finfo;

    // capture state
    int capture_compute;
    bool capture_started;

    id<MTLCaptureScope> capture_scope;

    // command buffer state
    int n_cb;           // number of extra threads used to submit the command buffers
    int n_nodes_0;      // number of nodes submitted by the main thread
    int n_nodes_1;      // remaining number of nodes submitted by the n_cb threads
    int n_nodes_per_cb;

    struct ggml_cgraph * gf;

    // the callback given to the thread pool
    void (^encode_async)(size_t ith);

    // n_cb command buffers + 1 used by the main thread
    struct ggml_metal_command_buffer cmd_bufs[GGML_METAL_MAX_COMMAND_BUFFERS + 1];

    // extra command buffers for things like getting, setting and copying tensors
    NSMutableArray * cmd_bufs_ext;

    // buffers to release after async Metal operations complete
    // if Metal released them, it would do so on a Metal-internal thread without an autorelease pool, which could cause leaks
    NSMutableArray * buf_refs;

    // the last command buffer queued into the Metal queue with operations relevant to the current Metal backend
    id<MTLCommandBuffer> cmd_buf_last;

    // abort ggml_metal_graph_compute if callback returns true
    ggml_abort_callback abort_callback;
    void *              abort_callback_data;

    // error state - set when a command buffer fails during synchronize
    // once set, graph_compute will return GGML_STATUS_FAILED until the backend is recreated
    bool has_error;

    ggml_metal_event_t ev_wait_pending[GGML_METAL_MAX_PENDING_WAITS];
    int                n_ev_wait_pending;

    // a peer copy reads this context's memory from the other GPU's queue, so nothing
    // it submits next may overwrite the source until that read reports done
    struct ggml_metal_peer_wait {
        id<MTLSharedEvent> ev;
        uint64_t           val;
    } peer_waits[GGML_METAL_MAX_PENDING_WAITS];
    int n_peer_waits;

    // destination command buffer a cross-device copy left open, so the reduction that follows
    // it is encoded into the same one instead of opening another
    id<MTLCommandBuffer> cmd_buf_hold;
};

// TOSH_MGPU_DEFER_WAITS folds a cross-device wait into the next command buffer this context
// submits instead of spending a whole one on it. Opt-in: on a single card both contexts share
// the device queue, so it measures slightly negative there and only 2 real GPUs can settle it.
static bool ggml_metal_defer_waits(void) {
    static int val = -1;
    if (val < 0) {
        const char * v = TOSH_ENV("TOSH_MGPU_DEFER_WAITS");
        val = (v && v[0] == '1') ? 1 : 0;
    }
    return val;
}

static _Atomic uint64_t g_tr_cmd_all     = 0; // command buffers created through the tracked paths
static _Atomic uint64_t g_tr_graphs      = 0; // graph submissions, one per subgraph per card
static _Atomic uint64_t g_tr_merge_reuse = 0; // subgraphs that continued the collective's buffer
static _Atomic uint64_t g_tr_merge_waits = 0; // waits that had queued in between, and would have
                                              // been dropped before this was fixed
static _Atomic uint64_t g_tr_signals     = 0; // cross-device event signals encoded
static _Atomic uint64_t g_tr_waits       = 0; // cross-device event waits encoded


// TOSH_MGPU_TRACE: where the collective's time and bytes go. The counters are only touched
// when it is on, so a release run pays one cached branch per crossing.
static _Atomic uint64_t g_tr_reduce      = 0; // exchange_reduce calls
static _Atomic uint64_t g_tr_copy        = 0; // plain cross-device copies
static _Atomic uint64_t g_tr_bytes_peer  = 0; // crossed through a remote buffer view
static _Atomic uint64_t g_tr_bytes_host  = 0; // crossed through the shared host block
static _Atomic uint64_t g_tr_cmd_bufs    = 0; // command buffers the collective created
static _Atomic uint64_t g_tr_rv_new      = 0; // remote views built
static _Atomic uint64_t g_tr_rv_hit      = 0; // remote views reused
static _Atomic uint64_t g_tr_encode_ns   = 0; // CPU time inside the collective
static _Atomic uint64_t g_tr_gpu_ns      = 0; // GPU execution of the collective's buffers
static _Atomic uint64_t g_tr_sched_ns    = 0; // committed to running: queueing plus event waits

static bool tosh_mgpu_trace(void) {
    static int on = -1;
    if (on < 0) {
        const char * v = TOSH_ENV("TOSH_MGPU_TRACE");
        on = (v && v[0] == '1') ? 1 : 0;
    }
    return on == 1;
}

static uint64_t tosh_now_ns(void) {
    return clock_gettime_nsec_np(CLOCK_MONOTONIC);
}

// Separates three things the old instrumentation conflated: when the host committed each
// side, when each GPU started the collective, and the signed order of both. The collective
// loop is serial on the CPU, so a GPU-start difference is not evidence of compute imbalance
// until the commit difference is subtracted from it.
#define TOSH_SKEW_SLOTS 8192
static struct {
    _Atomic uint64_t seq[2];
    _Atomic uint64_t commit_ns[2];
    _Atomic uint64_t gpu_ns[2];
} g_skew[TOSH_SKEW_SLOTS];

static _Atomic uint64_t g_sk_n;
static _Atomic uint64_t g_sk_gpu_hist[8], g_sk_com_hist[8];
static _Atomic uint64_t g_sk_gpu_sum, g_sk_com_sum, g_sk_gpu_max;
static _Atomic uint64_t g_sk_gpu_first0, g_sk_com_first0;   // how often peerIndex 0 leads

static int tosh_skew_bucket(double us) {
    return us < 1 ? 0 : us < 2 ? 1 : us < 5 ? 2 : us < 10 ? 3 :
           us < 20 ? 4 : us < 50 ? 5 : us < 200 ? 6 : 7;
}

static void tosh_skew_commit(uint64_t seq, int side, uint64_t t_ns) {
    const int slot = (int) (seq % TOSH_SKEW_SLOTS);
    atomic_store(&g_skew[slot].gpu_ns[side], 0);
    atomic_store(&g_skew[slot].commit_ns[side], t_ns);
    atomic_store(&g_skew[slot].seq[side], seq);
}

static void tosh_skew_started(uint64_t seq, int side, double gpu_start) {
    const int slot = (int) (seq % TOSH_SKEW_SLOTS);
    atomic_store(&g_skew[slot].gpu_ns[side], (uint64_t) (gpu_start*1e9));

    const int other = side ^ 1;
    if (atomic_load(&g_skew[slot].seq[other]) != seq) {
        return;
    }
    const uint64_t g0 = atomic_load(&g_skew[slot].gpu_ns[0]);
    const uint64_t g1 = atomic_load(&g_skew[slot].gpu_ns[1]);
    const uint64_t c0 = atomic_load(&g_skew[slot].commit_ns[0]);
    const uint64_t c1 = atomic_load(&g_skew[slot].commit_ns[1]);
    if (g0 == 0 || g1 == 0 || c0 == 0 || c1 == 0) {
        return;                                  // the other side has not started yet
    }
    // only the side that completes second gets here with both present
    if (side != 1 && atomic_load(&g_skew[slot].gpu_ns[other]) == 0) {
        return;
    }
    static _Atomic uint64_t claimed[TOSH_SKEW_SLOTS];
    uint64_t was = atomic_exchange(&claimed[slot], seq);
    if (was == seq) {
        return;                                  // already accounted
    }

    const uint64_t dg = g0 > g1 ? g0 - g1 : g1 - g0;
    const uint64_t dc = c0 > c1 ? c0 - c1 : c1 - c0;
    atomic_fetch_add(&g_sk_n, 1);
    atomic_fetch_add(&g_sk_gpu_sum, dg);
    atomic_fetch_add(&g_sk_com_sum, dc);
    if (g0 <= g1) { atomic_fetch_add(&g_sk_gpu_first0, 1); }
    if (c0 <= c1) { atomic_fetch_add(&g_sk_com_first0, 1); }
    uint64_t m = atomic_load(&g_sk_gpu_max);
    while (dg > m && !atomic_compare_exchange_weak(&g_sk_gpu_max, &m, dg)) { }
    atomic_fetch_add(&g_sk_gpu_hist[tosh_skew_bucket(dg/1000.0)], 1);
    atomic_fetch_add(&g_sk_com_hist[tosh_skew_bucket(dc/1000.0)], 1);
}

static void tosh_mgpu_trace_report(void);
static bool ggml_metal_collective_nowait(void);

// The subgraph command buffer is timed start and end separately, because a difference in
// completion can come from starting late or from running longer and those need different
// answers. Note what this buffer contains: ggml_metal_encode_pending_waits runs at its top,
// so its duration is graph compute plus any event wait encoded there, not pure compute.
static _Atomic uint64_t g_gr_ord[2];
static struct {
    _Atomic uint64_t ord[2];
    _Atomic uint64_t start_ns[2];
    _Atomic uint64_t end_ns[2];
} g_gr[TOSH_SKEW_SLOTS];

static _Atomic uint64_t g_gr_n;
static _Atomic uint64_t g_gr_hs[8], g_gr_hd[8], g_gr_he[8];      // |start|, |duration|, |end| deltas
static _Atomic uint64_t g_gr_ss, g_gr_sd, g_gr_se;               // their sums, ns
static _Atomic uint64_t g_gr_sign_start, g_gr_sign_dur;          // sign agreement with end delta
static _Atomic uint64_t g_gr_start0_first, g_gr_dur0_less;       // is either side systematically ahead
static _Atomic uint64_t g_gr_resid;                              // |end - (start + dur)| residual
static _Atomic uint64_t g_dur_n[2], g_dur_sum[2], g_dur_h[2][8]; // per-side duration

static int tosh_dur_bucket(double us) {
    return us <  40 ? 0 : us <  80 ? 1 : us < 120 ? 2 : us < 160 ? 3 :
           us < 200 ? 4 : us < 300 ? 5 : us < 500 ? 6 : 7;
}

static void tosh_trace_graph_end(id<MTLCommandBuffer> cmd_buf, int side) {
    if (!tosh_mgpu_trace()) {
        return;
    }
    static _Atomic int reg = 0;                              // a single GPU runs no collective
    int exp = 0;
    if (atomic_compare_exchange_strong(&reg, &exp, 1)) { atexit(tosh_mgpu_trace_report); }
    const uint64_t ord = atomic_fetch_add(&g_gr_ord[side], 1);
    [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        if (cb.GPUStartTime <= 0 || cb.GPUEndTime <= 0) {
            return;
        }
        const int slot = (int) (ord % TOSH_SKEW_SLOTS);
        const uint64_t st = (uint64_t) (cb.GPUStartTime*1e9);
        const uint64_t en = (uint64_t) (cb.GPUEndTime*1e9);

        atomic_fetch_add(&g_dur_n[side], 1);
        atomic_fetch_add(&g_dur_sum[side], en - st);
        atomic_fetch_add(&g_dur_h[side][tosh_dur_bucket((en - st)/1000.0)], 1);

        atomic_store(&g_gr[slot].start_ns[side], st);
        atomic_store(&g_gr[slot].end_ns[side], en);
        atomic_store(&g_gr[slot].ord[side], ord + 1);
        if (atomic_load(&g_gr[slot].ord[side ^ 1]) != ord + 1) {
            return;
        }
        const int64_t s0 = (int64_t) atomic_load(&g_gr[slot].start_ns[0]);
        const int64_t s1 = (int64_t) atomic_load(&g_gr[slot].start_ns[1]);
        const int64_t e0 = (int64_t) atomic_load(&g_gr[slot].end_ns[0]);
        const int64_t e1 = (int64_t) atomic_load(&g_gr[slot].end_ns[1]);
        if (s0 == 0 || s1 == 0 || e0 == 0 || e1 == 0) {
            return;
        }
        static _Atomic uint64_t claimed[TOSH_SKEW_SLOTS];
        uint64_t was = atomic_exchange(&claimed[slot], ord + 1);
        if (was == ord + 1) {
            return;
        }

        const int64_t ds = s0 - s1;
        const int64_t dd = (e0 - s0) - (e1 - s1);
        const int64_t de = e0 - e1;
        const uint64_t as = ds < 0 ? -ds : ds, ad = dd < 0 ? -dd : dd, ae = de < 0 ? -de : de;
        const int64_t r = de - (ds + dd);

        atomic_fetch_add(&g_gr_n, 1);
        atomic_fetch_add(&g_gr_ss, as);
        atomic_fetch_add(&g_gr_sd, ad);
        atomic_fetch_add(&g_gr_se, ae);
        atomic_fetch_add(&g_gr_resid, (uint64_t) (r < 0 ? -r : r));
        if (ds <= 0) { atomic_fetch_add(&g_gr_start0_first, 1); }
        if (dd <= 0) { atomic_fetch_add(&g_gr_dur0_less, 1); }
        if ((ds >= 0) == (de >= 0)) { atomic_fetch_add(&g_gr_sign_start, 1); }
        if ((dd >= 0) == (de >= 0)) { atomic_fetch_add(&g_gr_sign_dur, 1); }
        atomic_fetch_add(&g_gr_hs[tosh_skew_bucket(as/1000.0)], 1);
        atomic_fetch_add(&g_gr_hd[tosh_skew_bucket(ad/1000.0)], 1);
        atomic_fetch_add(&g_gr_he[tosh_skew_bucket(ae/1000.0)], 1);
    }];
}

static void tosh_trace_exchange(id<MTLCommandBuffer> cmd_buf, uint64_t seq, int side) {
    if (!tosh_mgpu_trace()) {
        return;
    }
    tosh_skew_commit(seq, side, clock_gettime_nsec_np(CLOCK_MONOTONIC));
    [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        if (cb.GPUStartTime > 0) {
            tosh_skew_started(seq, side, cb.GPUStartTime);
        }
    }];
}

// Steady-state decode only. Every counter is snapshotted the first time a collective runs on a
// single-row batch, which is the first generated token, so nothing from load, warm-up or the
// prompt is amortised into the per-token figures.
extern _Atomic uint64_t * ggml_backend_meta_pass_counter(void);

static _Atomic int      g_snap_taken = 0;
static uint64_t g_snap_reduce, g_snap_cmd, g_snap_graphs, g_snap_sig, g_snap_wait,
                g_snap_peer_b, g_snap_host_b, g_snap_passes;

static void tosh_decode_snapshot(int64_t batch) {
    if (batch != 1 || atomic_load(&g_snap_taken)) {
        return;
    }
    int expected = 0;
    if (!atomic_compare_exchange_strong(&g_snap_taken, &expected, 1)) {
        return;
    }
    g_snap_reduce = atomic_load(&g_tr_reduce);
    g_snap_cmd    = atomic_load(&g_tr_cmd_all);
    g_snap_graphs = atomic_load(&g_tr_graphs);
    g_snap_sig    = atomic_load(&g_tr_signals);
    g_snap_wait   = atomic_load(&g_tr_waits);
    g_snap_peer_b = atomic_load(&g_tr_bytes_peer);
    g_snap_host_b = atomic_load(&g_tr_bytes_host);
    g_snap_passes = atomic_load(ggml_backend_meta_pass_counter());
}

// TOSH_MGPU_CTIME=1, trace only. GPU timestamps inside the staged exchange, the path batch-1
// decode actually takes. Four samples, all from one device, so no cross-GPU clock correlation
// is involved: S0/S1 bracket the publish blit, S2/S3 the staging read. S1 to S2 is a
// WAIT-WINDOW, not a pure event wait: it spans the signal, the peer wait and the encoder switch.
#define TOSH_CT_SLOTS 4096
#define TOSH_CT_PER   4

static bool ggml_metal_ctime(void) {
    static int v = -1;
    if (v < 0) { const char * s = TOSH_ENV("TOSH_MGPU_CTIME"); v = (s && s[0] == '1') ? 1 : 0; }
    return v == 1;
}

static bool ggml_metal_ctime_barrier(void) {
    static int v = -1;
    if (v < 0) { const char * s = TOSH_ENV("TOSH_MGPU_CTIME_BARRIER"); v = (s && s[0] == '1') ? 1 : 0; }
    return v == 1;
}

static _Atomic uint64_t g_ct_n, g_ct_pub, g_ct_wait, g_ct_read, g_ct_fail;
static _Atomic uint64_t g_ct_hw[8], g_ct_hp[8], g_ct_hr[8];
static double g_ct_ns_per_tick = 0.0;

static int tosh_ct_bucket(double us) {
    return us < 1 ? 0 : us < 5 ? 1 : us < 10 ? 2 : us < 20 ? 3 :
           us < 50 ? 4 : us < 100 ? 5 : us < 200 ? 6 : 7;
}

static id<MTLCounterSampleBuffer> tosh_ct_buffer(ggml_metal_t ctx) {
    if (!ggml_metal_ctime()) { return nil; }
    if (ctx->ct_buf != nil)  { return ctx->ct_buf; }
    if (@available(macOS 10.15, *)) {
        id<MTLDevice> dev = ggml_metal_device_get_obj(ctx->dev);
        id<MTLCounterSet> ts = nil;
        for (id<MTLCounterSet> cs in dev.counterSets) {
            if ([cs.name isEqualToString:MTLCommonCounterSetTimestamp]) { ts = cs; break; }
        }
        if (ts == nil) { return nil; }
        MTLCounterSampleBufferDescriptor * d = [MTLCounterSampleBufferDescriptor new];
        d.counterSet = ts; d.sampleCount = TOSH_CT_SLOTS*TOSH_CT_PER; d.storageMode = MTLStorageModeShared;
        NSError * err = nil;
        ctx->ct_buf = [dev newCounterSampleBufferWithDescriptor:d error:&err];
        if (ctx->ct_buf == nil) {
            GGML_LOG_WARN("%s: counter sample buffer unavailable\n", __func__);
            return nil;
        }
        if (g_ct_ns_per_tick == 0.0) {
            g_ct_ns_per_tick = 1.0;
            MTLTimestamp c0 = 0, g0 = 0, c1 = 0, g1 = 0;
            [dev sampleTimestamps:&c0 gpuTimestamp:&g0];
            usleep(20000);
            [dev sampleTimestamps:&c1 gpuTimestamp:&g1];
            if (g1 > g0) { g_ct_ns_per_tick = (double) (c1 - c0)/(double) (g1 - g0); }
            GGML_LOG_INFO("%s: GPU timestamp sampling on, %.4f ns per tick\n", __func__, g_ct_ns_per_tick);
        }
    }
    return ctx->ct_buf;
}

static void tosh_ct_record(id<MTLCounterSampleBuffer> cs, uint64_t base) {
    if (@available(macOS 10.15, *)) {
        NSData * d = [cs resolveCounterRange:NSMakeRange(base, TOSH_CT_PER)];
        if (d == nil || d.length < TOSH_CT_PER*sizeof(MTLCounterResultTimestamp)) {
            atomic_fetch_add(&g_ct_fail, 1); return;
        }
        const MTLCounterResultTimestamp * t = (const MTLCounterResultTimestamp *) d.bytes;
        for (int i = 0; i < TOSH_CT_PER; i++) {
            if (t[i].timestamp == MTLCounterErrorValue) { atomic_fetch_add(&g_ct_fail, 1); return; }
        }
        const double k = g_ct_ns_per_tick;
        const double pub  = (double) (t[1].timestamp - t[0].timestamp)*k;
        const double wait = (double) (t[2].timestamp - t[1].timestamp)*k;
        const double read = (double) (t[3].timestamp - t[2].timestamp)*k;
        if (pub < 0 || wait < 0 || read < 0) { atomic_fetch_add(&g_ct_fail, 1); return; }
        atomic_fetch_add(&g_ct_n, 1);
        atomic_fetch_add(&g_ct_pub,  (uint64_t) pub);
        atomic_fetch_add(&g_ct_wait, (uint64_t) wait);
        atomic_fetch_add(&g_ct_read, (uint64_t) read);
        atomic_fetch_add(&g_ct_hp[tosh_ct_bucket(pub /1000.0)], 1);
        atomic_fetch_add(&g_ct_hw[tosh_ct_bucket(wait/1000.0)], 1);
        atomic_fetch_add(&g_ct_hr[tosh_ct_bucket(read/1000.0)], 1);
    }
}

// TOSH_MGPU_WAIT_SATISFIED=1, benchmark only. The staged collective keeps every encoder, every
// signal, the same command buffer and the same sampling points; the only change is that the
// peer-ready wait names a private event that is already signalled, so it can never stall. The
// read then observes whatever the partner left behind and the output is wrong by construction.
// Subtracting this run's central window from the normal one isolates the real peer stall
// without adding an encoder to the interval being measured.
// The same trick for the other wait in the path, the one guarding reuse of the shared host
// block. TOSH_MGPU_BP_SATISFIED=1, benchmark only, same warning applies.
static bool ggml_metal_bp_satisfied(void) {
    static int v = -1;
    if (v < 0) {
        const char * s = TOSH_ENV("TOSH_MGPU_BP_SATISFIED");
        v = (s && s[0] == '1') ? 1 : 0;
        if (v) {
            fprintf(stderr, "ggml_metal: WARNING TOSH_MGPU_BP_SATISFIED=1, the collective does not "
                            "wait before reusing the staging block. Benchmark only.\n");
        }
    }
    return v == 1;
}

        const char * s = TOSH_ENV("TOSH_MGPU_TP2_FUSED_BOUNDARY");
static bool ggml_metal_wait_satisfied(void) {
    static int v = -1;
    if (v < 0) {
        const char * s = TOSH_ENV("TOSH_MGPU_WAIT_SATISFIED");
        v = (s && s[0] == '1') ? 1 : 0;
        if (v) {
            fprintf(stderr, "ggml_metal: WARNING TOSH_MGPU_WAIT_SATISFIED=1, the collective never "
                            "waits for the peer and the output is meaningless. Benchmark only.\n");
        }
    }
    return v == 1;
}

static id<MTLSharedEvent> tosh_wait_dummy(ggml_metal_t ctx) {
    if (ctx->ev_wait_dummy == nil) {
        ctx->ev_wait_dummy = [ggml_metal_device_get_obj(ctx->dev) newSharedEvent];
        ctx->ev_wait_dummy.signaledValue = 1;   // satisfied before anything is encoded
    }
    return ctx->ev_wait_dummy;
}

static void tosh_mgpu_trace_report(void) {
    const uint64_t reduce = atomic_load(&g_tr_reduce);
    const uint64_t peer   = atomic_load(&g_tr_bytes_peer);
    const uint64_t host   = atomic_load(&g_tr_bytes_host);

    fprintf(stderr, "TOSH_MGPU_TRACE\n");
    fprintf(stderr, "  allreduce exchanges   %llu\n", (unsigned long long) reduce);
    fprintf(stderr, "  plain xdev copies     %llu\n", (unsigned long long) atomic_load(&g_tr_copy));
    fprintf(stderr, "  peer-direct bytes     %llu\n", (unsigned long long) peer);
    fprintf(stderr, "  host-staged bytes     %llu\n", (unsigned long long) host);
    fprintf(stderr, "  peer share            %.1f%%\n",
            peer + host ? 100.0*(double) peer/(double)(peer + host) : 0.0);
    fprintf(stderr, "  collective cmd bufs   %llu\n", (unsigned long long) atomic_load(&g_tr_cmd_bufs));
    fprintf(stderr, "  tracked cmd bufs      %llu\n", (unsigned long long) atomic_load(&g_tr_cmd_all));
    fprintf(stderr, "  merge reuse / waits   %llu/%llu\n",
            (unsigned long long) atomic_load(&g_tr_merge_reuse),
            (unsigned long long) atomic_load(&g_tr_merge_waits));
    fprintf(stderr, "  graph submissions     %llu\n", (unsigned long long) atomic_load(&g_tr_graphs));
    {
        const uint64_t n = atomic_load(&g_sk_n);
        if (n > 0) {
            static const char * lbl[8] = {"<1","<2","<5","<10","<20","<50","<200",">=200"};
            const double nn = (double) n;
            fprintf(stderr, "  skew n=%llu   commit mean %.1f us   GPU-start mean %.1f us   GPU max %.1f us\n",
                    (unsigned long long) n, atomic_load(&g_sk_com_sum)/1000.0/nn,
                    atomic_load(&g_sk_gpu_sum)/1000.0/nn, atomic_load(&g_sk_gpu_max)/1000.0);
            fprintf(stderr, "    side0 leads: commit %.1f%%   GPU %.1f%%\n",
                    100.0*atomic_load(&g_sk_com_first0)/nn, 100.0*atomic_load(&g_sk_gpu_first0)/nn);
            fprintf(stderr, "    commit us   ");
            for (int i = 0; i < 8; i++) { fprintf(stderr, "%s %.1f%%  ", lbl[i], 100.0*atomic_load(&g_sk_com_hist[i])/nn); }
            fprintf(stderr, "\n    GPUstart us ");
            for (int i = 0; i < 8; i++) { fprintf(stderr, "%s %.1f%%  ", lbl[i], 100.0*atomic_load(&g_sk_gpu_hist[i])/nn); }
            fprintf(stderr, "\n");
        }
    }
    {
        const uint64_t n = atomic_load(&g_gr_n);
        if (n > 0) {
            static const char * lbl[8] = {"<1","<2","<5","<10","<20","<50","<200",">=200"};
            const double nn = (double) n;
            fprintf(stderr, "  subgraph deltas n=%llu  |start| %.1f us  |duration| %.1f us  |end| %.1f us  residual %.3f us\n",
                    (unsigned long long) n, atomic_load(&g_gr_ss)/1000.0/nn,
                    atomic_load(&g_gr_sd)/1000.0/nn, atomic_load(&g_gr_se)/1000.0/nn,
                    atomic_load(&g_gr_resid)/1000.0/nn);
            fprintf(stderr, "    sign of end delta agrees with start %.1f%%, with duration %.1f%%\n",
                    100.0*atomic_load(&g_gr_sign_start)/nn, 100.0*atomic_load(&g_gr_sign_dur)/nn);
            fprintf(stderr, "    side0 starts first %.1f%%   side0 shorter %.1f%%\n",
                    100.0*atomic_load(&g_gr_start0_first)/nn, 100.0*atomic_load(&g_gr_dur0_less)/nn);
            const char * nm[3] = {"|start|  ", "|duration|", "|end|    "};
            _Atomic uint64_t * hh[3] = {g_gr_hs, g_gr_hd, g_gr_he};
            for (int k = 0; k < 3; k++) {
                fprintf(stderr, "    %s us ", nm[k]);
                for (int i = 0; i < 8; i++) { fprintf(stderr, "%s %.1f%%  ", lbl[i], 100.0*atomic_load(&hh[k][i])/nn); }
                fprintf(stderr, "\n");
            }
        }
        static const char * dlb2[8] = {"<40","<80","<120","<160","<200","<300","<500",">=500"};
        for (int sd = 0; sd < 2; sd++) {
            const uint64_t dn = atomic_load(&g_dur_n[sd]);
            if (!dn) continue;
            fprintf(stderr, "  side%d subgraph duration mean %.1f us  n=%llu  ", sd,
                    atomic_load(&g_dur_sum[sd])/1000.0/(double) dn, (unsigned long long) dn);
            for (int i = 0; i < 8; i++) { fprintf(stderr, "%s %.1f%%  ", dlb2[i], 100.0*atomic_load(&g_dur_h[sd][i])/(double) dn); }
            fprintf(stderr, "\n");
        }
    }
    {
        const uint64_t n = atomic_load(&g_ct_n);
        if (n > 0 || atomic_load(&g_ct_fail) > 0) {
            static const char * lbl[8] = {"<1","<5","<10","<20","<50","<100","<200",">=200"};
            const double nn = (double) (n ? n : 1);
            fprintf(stderr, "  staged timeline n=%llu (failed %llu)  publish %.1f us  WAIT-WINDOW %.1f us  read %.1f us\n",
                    (unsigned long long) n, (unsigned long long) atomic_load(&g_ct_fail),
                    atomic_load(&g_ct_pub)/1000.0/nn, atomic_load(&g_ct_wait)/1000.0/nn,
                    atomic_load(&g_ct_read)/1000.0/nn);
            const char * rn[3] = {"publish    ", "wait-window", "read       "};
            _Atomic uint64_t * hh[3] = {g_ct_hp, g_ct_hw, g_ct_hr};
            for (int k = 0; k < 3; k++) {
                fprintf(stderr, "    %s us ", rn[k]);
                for (int i = 0; i < 8; i++) { fprintf(stderr, "%s %.1f%%  ", lbl[i], 100.0*atomic_load(&hh[k][i])/nn); }
                fprintf(stderr, "\n");
            }
        }
    }
    fprintf(stderr, "  remote views new/hit  %llu/%llu\n",
            (unsigned long long) atomic_load(&g_tr_rv_new),
            (unsigned long long) atomic_load(&g_tr_rv_hit));
    fprintf(stderr, "  encode CPU            %.1f ms\n", atomic_load(&g_tr_encode_ns)/1e6);
    fprintf(stderr, "  collective GPU        %.1f ms\n", atomic_load(&g_tr_gpu_ns)/1e6);
    fprintf(stderr, "  commit to GPU start   %.1f ms\n", atomic_load(&g_tr_sched_ns)/1e6);
    // decode-only, per card per token
    if (atomic_load(&g_snap_taken)) {
        const double t = (double) (atomic_load(ggml_backend_meta_pass_counter()) - g_snap_passes);
        if (t > 0) {
            fprintf(stderr, "  --- steady-state decode, per card per token (%.0f passes) ---\n", t);
            fprintf(stderr, "    collectives         %.2f\n", (atomic_load(&g_tr_reduce) - g_snap_reduce)/t);
            fprintf(stderr, "    tracked cmd bufs    %.2f\n", (atomic_load(&g_tr_cmd_all) - g_snap_cmd)/t);
            fprintf(stderr, "    graph submissions   %.2f\n", (atomic_load(&g_tr_graphs) - g_snap_graphs)/t);
            fprintf(stderr, "    event signals       %.2f\n", (atomic_load(&g_tr_signals) - g_snap_sig)/t);
            fprintf(stderr, "    event waits         %.2f\n", (atomic_load(&g_tr_waits) - g_snap_wait)/t);
            fprintf(stderr, "    peer bytes          %.0f\n", (atomic_load(&g_tr_bytes_peer) - g_snap_peer_b)/t);
            fprintf(stderr, "    host-staged bytes   %.0f\n", (atomic_load(&g_tr_bytes_host) - g_snap_host_b)/t);
        }
    }

    if (reduce) {
        fprintf(stderr, "  per exchange          %.1f us GPU, %.1f us to start, %llu B\n",
                atomic_load(&g_tr_gpu_ns)/1e3/(double) reduce,
                atomic_load(&g_tr_sched_ns)/1e3/(double) reduce,
                (unsigned long long) ((peer + host)/reduce));
    }
}

// Metal reports both when the buffer was handed over and when the GPU actually ran it; the
// gap is the queueing and the event waits, which is the number the collective is judged on.
static void ggml_metal_encode_pending_waits(ggml_metal_t ctx, id<MTLCommandBuffer> cmd_buf) {
    for (int i = 0; i < ctx->n_ev_wait_pending; ++i) {
        ggml_metal_event_encode_wait(ctx->ev_wait_pending[i], (ggml_metal_cmd_buf_t) cmd_buf);
    }
    ctx->n_ev_wait_pending = 0;

    for (int i = 0; i < ctx->n_peer_waits; ++i) {
        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_waits, 1);
        [cmd_buf encodeWaitForEvent:ctx->peer_waits[i].ev value:ctx->peer_waits[i].val];
    }
    ctx->n_peer_waits = 0;
}

static void ggml_metal_cmd_buf_submit(ggml_metal_t ctx, id<MTLCommandBuffer> cmd_buf) {
    [cmd_buf commit];
    [ctx->cmd_bufs_ext addObject:cmd_buf];
    ctx->cmd_buf_last = cmd_buf;
    [cmd_buf retain];
}
static void ggml_metal_cmd_buf_hold_flush(ggml_metal_t ctx) {
    if (ctx->cmd_buf_hold == nil) {
        return;
    }

    id<MTLCommandBuffer> cmd_buf = ctx->cmd_buf_hold;
    ctx->cmd_buf_hold = nil;
    ggml_metal_cmd_buf_submit(ctx, cmd_buf);
    [cmd_buf release];
}

static id<MTLCommandBuffer> ggml_metal_cmd_buf_new(ggml_metal_t ctx) {
    if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_cmd_all, 1);
    id<MTLCommandBuffer> cmd_buf = [ctx->queue commandBuffer];
    ggml_metal_encode_pending_waits(ctx, cmd_buf);

    return cmd_buf;
}

// Continues the buffer a copy left open, if there is one. Only the reduction that follows the
// copy may do this: a buffer that stays open is a buffer that has not started running.
static id<MTLCommandBuffer> ggml_metal_cmd_buf_continue(ggml_metal_t ctx) {
    if (ctx->cmd_buf_hold != nil) {
        id<MTLCommandBuffer> cmd_buf = ctx->cmd_buf_hold;
        ctx->cmd_buf_hold = nil;
        return [cmd_buf autorelease];
    }

    return ggml_metal_cmd_buf_new(ctx);
}

static void ggml_metal_cmd_buf_end(ggml_metal_t ctx, id<MTLCommandBuffer> cmd_buf, bool hold) {
    if (hold) {
        ctx->cmd_buf_hold = [cmd_buf retain];
    } else {
        ggml_metal_cmd_buf_submit(ctx, cmd_buf);
    }
}

// folded into the next command buffer this context submits, so the back-edge costs
// no command buffer of its own
static void ggml_metal_peer_wait_add(ggml_metal_t ctx, id<MTLSharedEvent> ev, uint64_t val) {
    for (int i = 0; i < ctx->n_peer_waits; ++i) {
        if (ctx->peer_waits[i].ev == ev) {
            ctx->peer_waits[i].val = MAX(ctx->peer_waits[i].val, val);
            return;
        }
    }

    if (ctx->n_peer_waits < GGML_METAL_MAX_PENDING_WAITS) {
        ctx->peer_waits[ctx->n_peer_waits].ev  = ev;
        ctx->peer_waits[ctx->n_peer_waits].val = val;
        ctx->n_peer_waits++;
        return;
    }

    @autoreleasepool {
        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_cmd_all, 1);
        id<MTLCommandBuffer> cmd_buf = [ctx->queue commandBuffer];
        ggml_metal_encode_pending_waits(ctx, cmd_buf);
        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_waits, 1);
        [cmd_buf encodeWaitForEvent:ev value:val];
        ggml_metal_cmd_buf_submit(ctx, cmd_buf);
    }
}

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev, bool prefetch) {
    GGML_LOG_INFO("%s: allocating\n", __func__);

    @autoreleasepool {
#if TARGET_OS_OSX && !GGML_METAL_NDEBUG
        // Show all the Metal device instances in the system
        NSArray * devices = MTLCopyAllDevices();
        for (id<MTLDevice> device in devices) {
            GGML_LOG_INFO("%s: found device: %s\n", __func__, [[device name] UTF8String]);
        }
        [devices release]; // since it was created by a *Copy* C method
#endif

        // init context
        ggml_metal_t res = calloc(1, sizeof(struct ggml_metal));

        id<MTLDevice> device = ggml_metal_device_get_obj(dev);

        GGML_LOG_INFO("%s: picking default device: %s\n", __func__, [[device name] UTF8String]);

        // TODO: would it be better to have one queue for the backend and one queue for the device?
        //       the graph encoders and async ops would use the backend queue while the sync ops would use the device queue?
        //res->queue = [device newCommandQueue]; [TAG_QUEUE_PER_BACKEND]
        res->queue = prefetch ? ggml_metal_device_acquire_queue(dev, &res->owns_queue)
                              : ggml_metal_device_get_queue(dev);
        if (res->queue == nil) {
            GGML_LOG_ERROR("%s: error: failed to create command queue\n", __func__);
            free(res);
            return NULL;
        }
        res->is_prefetch = prefetch;
        if (prefetch) {
            const char * val = TOSH_ENV("GGML_METAL_PREFETCH_STAGE_COUNT");
            const int requested = val ? atoi(val) : 1;
            res->prefetch_stage_count = MAX(1, MIN(requested, GGML_METAL_MAX_PREFETCH_STAGES));
        }

        res->dev = dev;
        res->lib = ggml_metal_device_get_library(dev);
        if (res->lib == NULL) {
            GGML_LOG_WARN("%s: the device does not have a precompiled Metal library - this is unexpected\n", __func__);
            GGML_LOG_WARN("%s: will try to compile it on the fly\n", __func__);

            res->lib = ggml_metal_library_init(dev);
            if (res->lib == NULL) {
                GGML_LOG_ERROR("%s: error: failed to initialize the Metal library\n", __func__);

                free(res);

                return NULL;
            }
        }

        res->ev_cpy  = ggml_metal_device_event_init(dev);
        res->ev_sync = ggml_metal_device_event_init(dev);

        const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(dev);

        snprintf(res->name, sizeof(res->name), "%s", props_dev->name);

        res->d_queue = dispatch_queue_create("ggml-metal", DISPATCH_QUEUE_CONCURRENT);

        // discrete GPUs (AMD) corrupt their output with a concurrent encoder: the driver does
        // not honour the memory barrier. GGML_METAL_CONCURRENCY_ENABLE forces it back on to test.
        res->use_concurrency = props_dev->has_unified_memory
            ? TOSH_ENV("GGML_METAL_CONCURRENCY_DISABLE") == nil
            : TOSH_ENV("GGML_METAL_CONCURRENCY_ENABLE")  != nil;

        {
            const char * val = TOSH_ENV("GGML_METAL_GRAPH_DEBUG");
            res->debug_graph = val ? atoi(val) : 0;
        }

        res->use_graph_optimize = true;

        if (TOSH_ENV("GGML_METAL_GRAPH_OPTIMIZE_DISABLE") != NULL) {
            res->use_graph_optimize = false;
        }

        res->finfo = ggml_metal_device_get_fusion_info(dev);
        if (ggml_metal_fusion_info_stats(res->finfo)) {
            ggml_metal_fusion_info_labels_init(res->finfo);
            res->n_cb = 0;
        }

        GGML_LOG_INFO("%s: use fusion         = %s\n", __func__, ggml_metal_fusion_info_enabled(res->finfo) ? "true" : "false");
        GGML_LOG_INFO("%s: use concurrency    = %s\n", __func__, res->use_concurrency    ? "true" : "false");
        GGML_LOG_INFO("%s: use graph optimize = %s\n", __func__, res->use_graph_optimize ? "true" : "false");

        res->capture_compute = 0;
        res->capture_started = false;
        res->capture_scope = nil;

        {
            const char * val = TOSH_ENV("GGML_METAL_CAPTURE_COMPUTE");
            if (val) {
                res->capture_compute = atoi(val);
            }
        }

        res->has_error = false;

        res->gf = nil;
        res->encode_async = nil;
        for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
            res->cmd_bufs[i].obj = nil;
        }

        res->cmd_bufs_ext = [[NSMutableArray alloc] init];
        res->buf_refs     = [[NSMutableArray alloc] init];

        res->cmd_buf_last = nil;

        res->pipelines_ext = ggml_metal_pipelines_init();

        return res;
    }
}

void ggml_metal_free(ggml_metal_t ctx) {
    tosh_cbp_dump_dev(ctx->dev);

    GGML_LOG_INFO("%s: deallocating\n", __func__);

    ggml_metal_cmd_buf_hold_flush(ctx);

    for (int i = 0; i < ctx->prefetch_stage_count; ++i) {
        if (ctx->prefetch_stage_cmds[i]) {
            [ctx->prefetch_stage_cmds[i] waitUntilCompleted];
            [ctx->prefetch_stage_cmds[i] release];
        }
        [ctx->prefetch_stages[i] release];
    }

    for (int i = 0; i < ctx->n_xlinks; ++i) {
        if (ctx->xlinks[i].seq > 0) {
            [ctx->xlinks[i].ev_done waitUntilSignaledValue:ctx->xlinks[i].seq timeoutMS:10000];
        }
        [ctx->xlinks[i].wrap_src release];
        [ctx->xlinks[i].wrap_dst release];
        [ctx->xlinks[i].peer_view release];
        [ctx->xlinks[i].peer_src release];
        [ctx->xlinks[i].ev_ready release];
        [ctx->xlinks[i].ev_done release];
        free(ctx->xlinks[i].host);
    }

    for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        if (ctx->cmd_bufs[i].obj) {
            [ctx->cmd_bufs[i].obj release];
        }
    }

    for (int i = 0; i < (int) ctx->cmd_bufs_ext.count; ++i) {
        if (ctx->cmd_bufs_ext[i]) {
            [ctx->cmd_bufs_ext[i] release];
        }
    }

    [ctx->cmd_bufs_ext removeAllObjects];
    [ctx->cmd_bufs_ext release];

    @autoreleasepool {
        [ctx->buf_refs removeAllObjects];
        [ctx->buf_refs release];
    }

    if (ctx->pipelines_ext) {
        ggml_metal_pipelines_free(ctx->pipelines_ext);
        ctx->pipelines_ext = nil;
    }

    if (ggml_metal_fusion_info_debug(ctx->finfo) > 0) {
        GGML_LOG_DEBUG("%s: fusion stats:\n", __func__);

        const int n_fusions = ggml_metal_fusion_info_n_fusions(ctx->finfo);
        for (int i = 0; i < n_fusions; i++) {
            const uint64_t count = ggml_metal_fusion_info_count(ctx->finfo, i);
            if (count == 0) {
                continue;
            }

            // note: cannot use ggml_log here
            GGML_LOG_DEBUG("%s: - %s: %" PRIu64 "\n", __func__, ggml_metal_fusion_info_label(ctx->finfo, i), count);
        }
    }

    Block_release(ctx->encode_async);

    ggml_metal_device_release_queue(ctx->dev, ctx->queue, ctx->owns_queue);

    dispatch_release(ctx->d_queue);

    ggml_metal_device_event_free(ctx->dev, ctx->ev_cpy);
    ggml_metal_device_event_free(ctx->dev, ctx->ev_sync);

    free(ctx);
}

const char * ggml_metal_get_name(ggml_metal_t ctx) {
    return ctx->name;
}

void ggml_metal_synchronize(ggml_metal_t ctx) {
    // nothing completes while a command buffer is still held open
    ggml_metal_cmd_buf_hold_flush(ctx);

    // pending waits guard memory a peer blit can still be reading, so they must run and complete before synchronize returns
    if (ctx->n_ev_wait_pending > 0 || ctx->n_peer_waits > 0) {
        @autoreleasepool {
            if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_cmd_all, 1);
            id<MTLCommandBuffer> cmd_buf = [ctx->queue commandBuffer];
            ggml_metal_encode_pending_waits(ctx, cmd_buf);
            ggml_metal_cmd_buf_submit(ctx, cmd_buf);
        }
    }

    // wait for any backend operations to finish
    const bool cbp_on = tosh_cbp_on();
    uint64_t cbp_t = cbp_on ? clock_gettime_nsec_np(CLOCK_UPTIME_RAW) : 0;
    if (ctx->cmd_buf_last) {
        [ctx->cmd_buf_last waitUntilCompleted];
        ctx->cmd_buf_last = nil;
    }
    if (cbp_on) {
        const uint64_t t1 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        atomic_fetch_add_explicit(&g_cbp_wait_ns, t1 - cbp_t, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_cbp_sync_n, 1, memory_order_relaxed);
        cbp_t = t1;
    }

    // in-flight staged uploads count as pending backend operations
    ggml_metal_device_upload_drain(ctx->dev);
    if (cbp_on) {
        atomic_fetch_add_explicit(&g_cbp_drain_ns,
                clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - cbp_t, memory_order_relaxed);
    }

    // check status of all command buffers
    {
        const int n_cb = ctx->n_cb;

        for (int cb_idx = 0; cb_idx <= n_cb; ++cb_idx) {
            id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;
            if (!cmd_buf) {
                continue;
            }

            MTLCommandBufferStatus status = [cmd_buf status];
            if (status != MTLCommandBufferStatusCompleted) {
                GGML_LOG_ERROR("%s: error: command buffer %d failed with status %d\n", __func__, cb_idx, (int) status);
                if (status == MTLCommandBufferStatusError) {
                    GGML_LOG_ERROR("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                }
                ctx->has_error = true;
                return;
            }
        }
    }

    // release any completed extra command buffers
    if (ctx->cmd_bufs_ext.count > 0) {
        for (size_t i = 0; i < ctx->cmd_bufs_ext.count; ++i) {
            id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs_ext[i];
    // release any completed extra command buffers
    if (ctx->cmd_bufs_ext.count > 0) {
        for (size_t i = 0; i < ctx->cmd_bufs_ext.count; ++i) {
            id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs_ext[i];

            MTLCommandBufferStatus status = [cmd_buf status];
            if (status != MTLCommandBufferStatusCompleted) {
                GGML_LOG_ERROR("%s: error: command buffer %d failed with status %d\n", __func__, (int) i, (int) status);
                if (status == MTLCommandBufferStatusError) {
                    GGML_LOG_ERROR("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                }

                // release this and all remaining command buffers before returning
                for (size_t j = i; j < ctx->cmd_bufs_ext.count; ++j) {
                    [ctx->cmd_bufs_ext[j] release];
                }
                [ctx->cmd_bufs_ext removeAllObjects];

                ctx->has_error = true;
                return;
            }

            [cmd_buf release];
        }

        [ctx->cmd_bufs_ext removeAllObjects];
    }

    @autoreleasepool {
        [ctx->buf_refs removeAllObjects];
    }

    for (int i = 0; i < ctx->prefetch_stage_count; ++i) {
        if (ctx->prefetch_stage_cmds[i]) {
            [ctx->prefetch_stage_cmds[i] release];
            ctx->prefetch_stage_cmds[i] = nil;
        }
    }
}

static struct ggml_metal_buffer_id ggml_metal_get_buffer_id(const struct ggml_tensor * t) {
    if (!t) {
        return (struct ggml_metal_buffer_id) { nil, 0 };
    }

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    return ggml_metal_buffer_get_id(buffer->context, t);
}

void ggml_metal_read_batch_begin(ggml_metal_t ctx) {
    ggml_metal_device_read_batch_begin(ctx->dev);
}

void ggml_metal_read_batch_end(ggml_metal_t ctx) {
    ggml_metal_device_read_batch_end(ctx->dev);
}
    @autoreleasepool {
        ggml_metal_prof_note(tensor->name, size, false);
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);

        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(tensor);
        if (bid_dst.metal == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        bid_dst.offs += offset;

        // Expert prefetch keeps the zero-copy source path while it fits the
        // device budget, then falls back to a bounded shared staging ring.
        if (ctx->is_prefetch) {
            int stage = -1;
            size_t offs_src = 0;
            id<MTLBuffer> buf_src = ggml_metal_device_wrap_host(ctx->dev, data, size, &offs_src);

            if (buf_src == nil) {
                stage = ctx->prefetch_stage_cur;
                ctx->prefetch_stage_cur = (ctx->prefetch_stage_cur + 1) % ctx->prefetch_stage_count;

                if (ctx->prefetch_stage_cmds[stage]) {
                    [ctx->prefetch_stage_cmds[stage] waitUntilCompleted];
                    [ctx->prefetch_stage_cmds[stage] release];
                    ctx->prefetch_stage_cmds[stage] = nil;
                }

                buf_src = ctx->prefetch_stages[stage];
                if (buf_src == nil || buf_src.length < size) {
                    [buf_src release];
                    buf_src = [device newBufferWithLength:size
                                                  options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];
                    ctx->prefetch_stages[stage] = buf_src;
                }

                if (buf_src != nil) {
                    memcpy(buf_src.contents, data, size);
                    offs_src = 0;
                }
            }

            if (buf_src != nil) {
                if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_cmd_all, 1);
                id<MTLCommandBuffer> cmd_buf = [ctx->queue commandBuffer];
                ggml_metal_encode_pending_waits(ctx, cmd_buf);
                id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

                [encoder copyFromBuffer:buf_src
                           sourceOffset:offs_src
                               toBuffer:bid_dst.metal
                      destinationOffset:bid_dst.offs
                                   size:size];

                [encoder endEncoding];
                [cmd_buf commit];

                [ctx->cmd_bufs_ext addObject:cmd_buf];
                ctx->cmd_buf_last = cmd_buf;
                if (stage >= 0) {
                    ctx->prefetch_stage_cmds[stage] = [cmd_buf retain];
                }

                [cmd_buf retain];

                return;
            }
        }

        // private buffers route small transfers through the device staging
        // buffer (synchronous) instead of allocating a Metal buffer per call
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        if (!ctx->owns_queue && !ggml_metal_buffer_is_shared(buffer->context)
            && ggml_metal_device_stage_set(ctx->dev, bid_dst, data, size)) {
            return;
        }

        // wrap the source data into a Metal buffer
        id<MTLBuffer> buf_src = [device newBufferWithBytes:data
                                                    length:size
                                                   options:MTLResourceStorageModeShared];

        GGML_ASSERT(buf_src);

        // queue the copy operation into the queue of the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ctx->queue;
        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_cmd_all, 1);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        ggml_metal_encode_pending_waits(ctx, cmd_buf);
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:buf_src
                   sourceOffset:0
                       toBuffer:bid_dst.metal
              destinationOffset:bid_dst.offs
                           size:size];

        [encoder endEncoding];
        [cmd_buf commit];

        [ctx->buf_refs addObject:buf_src];
        [buf_src release];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

void ggml_metal_get_tensor_async(ggml_metal_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    if (size == 0) {
        return;
    }

    @autoreleasepool {
        ggml_metal_prof_note(tensor->name, size, true);
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);

        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(tensor);
        if (bid_src.metal == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        bid_src.offs += offset;

        // private buffers route small transfers through the device staging
        // buffer (synchronous) instead of allocating a Metal buffer per call
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        if (!ggml_metal_buffer_is_shared(buffer->context)
            && ggml_metal_device_stage_get(ctx->dev, bid_src, data, size)) {
            return;
        }

        id<MTLCommandQueue> queue = ctx->queue;

        id<MTLBuffer> buf_dst = [device newBufferWithBytesNoCopy:data
                                                          length:size
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];

        if (buf_dst != nil) {
            // queue the copy operation into the queue of the Metal context
            // this will be queued at the end, after any currently ongoing GPU operations
            if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_cmd_all, 1);
            id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
            ggml_metal_encode_pending_waits(ctx, cmd_buf);
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder copyFromBuffer:bid_src.metal
                       sourceOffset:bid_src.offs
                           toBuffer:buf_dst
                  destinationOffset:0
                               size:size];

            [encoder endEncoding];
            [cmd_buf commit];

            [ctx->buf_refs addObject:buf_dst];
            [buf_dst release];

            // do not wait here for completion
            //[cmd_buf waitUntilCompleted];

            // instead, remember a reference to the command buffer and wait for it later if needed
            [ctx->cmd_bufs_ext addObject:cmd_buf];
            ctx->cmd_buf_last = cmd_buf;

            [cmd_buf retain];

            return;
        }

        // newBufferWithBytesNoCopy requires page-aligned data, and some drivers cap the size of
        // host-visible allocations; copy through a small staging buffer in chunks instead. The
        // staging path is synchronous (commit + waitUntilCompleted) so callers that would later
        // wait on cmd_buf_last still observe a completed transfer.
        size_t stage_size = MIN(size, (size_t) 8*1024*1024);
        id<MTLBuffer> buf_stage = nil;
        while (stage_size > 0) {
            buf_stage = [device newBufferWithLength:stage_size options:MTLResourceStorageModeShared];
            if (buf_stage != nil) {
                break;
            }
            stage_size /= 2;
        }
        if (buf_stage == nil) {
            GGML_LOG_ERROR("%s: failed to allocate staging buffer, size = %zu\n", __func__, size);
        }
        GGML_ASSERT(buf_stage);

        for (size_t done = 0; done < size;) {
            const size_t n = MIN(stage_size, size - done);

            if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_cmd_all, 1);
            id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
            ggml_metal_encode_pending_waits(ctx, cmd_buf);

            {
                id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

                [encoder copyFromBuffer:bid_src.metal
                           sourceOffset:bid_src.offs + done
                               toBuffer:buf_stage
                      destinationOffset:0
                                   size:n];

                [encoder endEncoding];
            }

            [cmd_buf commit];
            [cmd_buf waitUntilCompleted];

            memcpy((char *) data + done, buf_stage.contents, n);

            done += n;
        }
    }
}

static struct ggml_metal_xdev_link * ggml_metal_xdev_link_get(
        ggml_metal_t ctx_src, ggml_metal_t ctx_dst) {
    struct ggml_metal_xdev_link * link = NULL;
    for (int i = 0; i < ctx_dst->n_xlinks; i++) {
        if (ctx_dst->xlinks[i].src_dev == ctx_src->dev) {
            link = &ctx_dst->xlinks[i];
            break;
        }
    }
    if (link == NULL) {
        if (ctx_dst->n_xlinks == 4) {
            return NULL;
        }
        link = &ctx_dst->xlinks[ctx_dst->n_xlinks++];
        id<MTLDevice> dev_src = ggml_metal_device_get_obj(ctx_src->dev);
        id<MTLDevice> dev_dst = ggml_metal_device_get_obj(ctx_dst->dev);
        link->ev_ready = [dev_src newSharedEvent];
        link->ev_done  = [dev_dst newSharedEvent];
            [link->ev_ready release];
            [link->ev_done release];
            memset(link, 0, sizeof(*link));
            ctx_dst->n_xlinks--;
            return NULL;
        }
    }

    return link;
}

static bool ggml_metal_xdev_link_reserve(ggml_metal_t ctx_src, ggml_metal_t ctx_dst,
                                        struct ggml_metal_xdev_link * link, size_t size) {
    if (link->cap == 0) {
        GGML_LOG_INFO("%s: cross-device hand-off via shared events and host staging (TOSH_MGPU_EVENTS)\n", __func__);
    }

    if (link->cap < size) {
        if (link->seq > 0 && ![link->ev_done waitUntilSignaledValue:link->seq timeoutMS:10000]) {
            return false;
        [link->wrap_src release];
        [link->wrap_dst release];
        free(link->host);
        link->cap = MAX(size, (size_t) 1*1024*1024);
        if (posix_memalign(&link->host, (size_t) sysconf(_SC_PAGESIZE), link->cap) != 0) {
            link->host = NULL;
            link->cap  = 0;
            return false;
        }
        id<MTLDevice> dev_src = ggml_metal_device_get_obj(ctx_src->dev);
        id<MTLDevice> dev_dst = ggml_metal_device_get_obj(ctx_dst->dev);
        link->wrap_src = [dev_src newBufferWithBytesNoCopy:link->host
                              length:link->cap options:MTLResourceStorageModeShared deallocator:nil];
        link->wrap_dst = [dev_dst newBufferWithBytesNoCopy:link->host
                              length:link->cap options:MTLResourceStorageModeShared deallocator:nil];
        if (link->wrap_src == nil || link->wrap_dst == nil) {
            // leaving cap set would make the next call skip this block and hand out the nil wraps
            [link->wrap_src release];
            [link->wrap_dst release];
            link->wrap_src = nil;
            link->wrap_dst = nil;
            free(link->host);
            link->host = NULL;
            link->cap  = 0;
            return false;
        }
    }
    return true;
// TOSH_MGPU_NOWAIT=1, benchmark only: the collective still publishes, still signals and still
// reduces, but never waits for the partner. The output is wrong by construction. Subtracting
// this from the normal run measures the stall at the wait inside the real collective, instead
// of inferring it from whole-command-buffer completion timestamps.
static bool ggml_metal_collective_nowait(void) {
    static int v = -1;
    if (v < 0) {
        const char * s = TOSH_ENV("TOSH_MGPU_NOWAIT");
        v = (s && s[0] == '1') ? 1 : 0;
        if (v) {
            fprintf(stderr, "ggml_metal: WARNING TOSH_MGPU_NOWAIT=1, the collective does not wait "
                            "for the peer and the output is meaningless. Benchmark only.\n");
        }
    }
    return v == 1;
}

// TOSH_MGPU_PEER_SELF runs the peer hand-off between two logical devices backed by the
// same GPU (GGML_METAL_DEVICE_LIST=0,0), which needs no bridge: the queues are still
// separate, so the cross-device ordering is testable on a single card.
static bool ggml_metal_peer_self(void) {
    static int val = -1;
    if (val < 0) {
        const char * v = TOSH_ENV("TOSH_MGPU_PEER_SELF");
        val = (v && v[0] == '1') ? 1 : 0;
    }
    return val;
}

// Direct copy between GPUs in the same Metal peer group.
static bool ggml_metal_cpy_xdev_peer(ggml_metal_t ctx_src, ggml_metal_t ctx_dst,
                                     const struct ggml_tensor * src, struct ggml_tensor * dst) {
#if TARGET_OS_OSX
    if (@available(macOS 10.15, *)) {
        id<MTLDevice> dev_src = ggml_metal_device_get_obj(ctx_src->dev);
        id<MTLDevice> dev_dst = ggml_metal_device_get_obj(ctx_dst->dev);
        const bool same_dev = dev_src == dev_dst && ggml_metal_peer_self();
        const uint64_t peer_group = dev_src.peerGroupID;
        if (!same_dev && (peer_group == 0 || peer_group != dev_dst.peerGroupID)) {
            return false;
        }

        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(src);
        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(dst);
        if (bid_src.metal == nil || bid_dst.metal == nil) {
            return false;
        }
        id<MTLBuffer> metal_src = (id<MTLBuffer>) bid_src.metal;
        id<MTLBuffer> metal_dst = (id<MTLBuffer>) bid_dst.metal;

        const size_t size = ggml_nbytes(src);
        if (bid_src.offs > metal_src.length || size > metal_src.length - bid_src.offs ||
            bid_dst.offs > metal_dst.length || size > metal_dst.length - bid_dst.offs) {
            return false;
        }

        struct ggml_metal_xdev_link * link = ggml_metal_xdev_link_get(ctx_src, ctx_dst);
        if (link == NULL) {
            return false;
        }
        if (link->peer_src != metal_src) {
            // Wait before replacing a remote view still used by the last blit.
            if (link->peer_view != nil && link->seq > 0 &&
                ![link->ev_done waitUntilSignaledValue:link->seq timeoutMS:10000]) {
                return false;
            }

            [link->peer_view release];
            [link->peer_src release];
            link->peer_view = same_dev ? [metal_src retain]
                                       : [metal_src newRemoteBufferViewForDevice:dev_dst];
            link->peer_src  = link->peer_view != nil ? [metal_src retain] : nil;
            if (link->peer_view == nil) {
                return false;

        const uint64_t seq = ++link->seq;
        @autoreleasepool {
            if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_cmd_all, 1);
            id<MTLCommandBuffer> cmd_src = [ctx_src->queue commandBuffer];
            ggml_metal_encode_pending_waits(ctx_src, cmd_src);
            if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_signals, 1);
        [cmd_src encodeSignalEvent:link->ev_ready value:seq];
            [cmd_src commit];
            [ctx_src->cmd_bufs_ext addObject:cmd_src];
            ctx_src->cmd_buf_last = cmd_src;
            [cmd_src retain];

            if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_cmd_all, 1);
            id<MTLCommandBuffer> cmd_dst = [ctx_dst->queue commandBuffer];
            ggml_metal_encode_pending_waits(ctx_dst, cmd_dst);
            if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_waits, 1);
        [cmd_dst encodeWaitForEvent:link->ev_ready value:seq];
            id<MTLBlitCommandEncoder> encoder = [cmd_dst blitCommandEncoder];
            [encoder copyFromBuffer:link->peer_view sourceOffset:bid_src.offs
                           toBuffer:metal_dst destinationOffset:bid_dst.offs size:size];
            [encoder endEncoding];
            if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_signals, 1);
        [cmd_dst encodeSignalEvent:link->ev_done value:seq];
            [cmd_dst commit];
            [ctx_dst->cmd_bufs_ext addObject:cmd_dst];
            ctx_dst->cmd_buf_last = cmd_dst;
            [cmd_dst retain];
        }

        ggml_metal_peer_wait_add(ctx_src, link->ev_done, seq);

        if (!link->peer_logged) {
            // fprintf, not GGML_LOG_INFO, so the confirmation shows at the default
            // log level like the SIMD-width line. Deduped by peer group across contexts:
            // a tool that reloads the model per run rebuilds the link every time.
            static uint64_t logged[8];
            static int      n_logged = 0;

            bool seen = false;
            for (int i = 0; i < n_logged; ++i) {
                seen = seen || logged[i] == peer_group;
            }

            if (!seen) {
                fprintf(stderr, same_dev
                            ? "ggml_metal: peer transfer self-test: %s -> %s (no bridge, peer group %llu)\n"
                            : "ggml_metal: Infinity Fabric peer transfer enabled: %s -> %s (peer group %llu)\n",
                        dev_src.name.UTF8String, dev_dst.name.UTF8String,
                        (unsigned long long) peer_group);
                if (n_logged < (int) (sizeof(logged)/sizeof(logged[0]))) {
                    logged[n_logged++] = peer_group;
                }
            }

            link->peer_logged = true;
        }
        return true;
    }
#else
    GGML_UNUSED(ctx_src);
    GGML_UNUSED(ctx_dst);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
#endif
    return false;
}

// event-chained hand-off: both blits share one host block and order through
// MTLSharedEvents, so neither GPU drains and the CPU never blocks
static _Atomic uint64_t g_xdev_ev = 0, g_xdev_peer = 0, g_xdev_gen = 0;

static void tosh_xdev_report(void) {
    fprintf(stderr, "TOSH_XDEV  events=%llu  peer=%llu  generic=%llu\n",
            (unsigned long long) atomic_load(&g_xdev_ev),
            (unsigned long long) atomic_load(&g_xdev_peer),
            (unsigned long long) atomic_load(&g_xdev_gen));
}

static bool ggml_metal_cpy_xdev_events(ggml_metal_t ctx_src, ggml_metal_t ctx_dst,
                                       const struct ggml_tensor * src, struct ggml_tensor * dst,
                                       bool hold_dst) {
    struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(src);
    struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(dst);
    if (bid_src.metal == nil || bid_dst.metal == nil) {
        return false;
    }

    const size_t size = ggml_nbytes(src);

    struct ggml_metal_xdev_link * link = ggml_metal_xdev_link_get(ctx_src, ctx_dst);
    if (link == NULL) {
        return false;
    }
    if (!ggml_metal_xdev_link_reserve(ctx_src, ctx_dst, link, size)) {
    const uint64_t seq = ++link->seq;

    @autoreleasepool {
        // reuse the host block only after the destination consumed the last hand-off
        id<MTLCommandBuffer> cmd_src = ggml_metal_cmd_buf_continue(ctx_src);
        if (seq > 1) {
            if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_waits, 1);
        [cmd_src encodeWaitForEvent:link->ev_done value:seq - 1];
        }
        id<MTLBlitCommandEncoder> enc_src = [cmd_src blitCommandEncoder];
        [enc_src copyFromBuffer:bid_src.metal sourceOffset:bid_src.offs
                       toBuffer:link->wrap_src destinationOffset:0 size:size];
        [enc_src endEncoding];
        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_signals, 1);
        [cmd_src encodeSignalEvent:link->ev_ready value:seq];
        ggml_metal_cmd_buf_end(ctx_src, cmd_src, false);

        id<MTLCommandBuffer> cmd_dst = ggml_metal_cmd_buf_new(ctx_dst);
        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_waits, 1);
        [cmd_dst encodeWaitForEvent:link->ev_ready value:seq];
        id<MTLBlitCommandEncoder> enc_dst = [cmd_dst blitCommandEncoder];
        [enc_dst copyFromBuffer:link->wrap_dst sourceOffset:0
                       toBuffer:bid_dst.metal destinationOffset:bid_dst.offs size:size];
        [enc_dst endEncoding];
        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_signals, 1);
        [cmd_dst encodeSignalEvent:link->ev_done value:seq];
        // the signal lands when the blit does, so holding the buffer open does not keep the
        // source waiting on whatever gets encoded into it afterwards
        ggml_metal_cmd_buf_end(ctx_dst, cmd_dst, hold_dst);
    }

    return true;
}

bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    return ggml_metal_cpy_tensor_async_ex(ctx_src, ctx_dst, src, dst, /*prefer_events =*/ false, /*hold_dst =*/ false);
}

bool ggml_metal_cpy_tensor_async_ex(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, const struct ggml_tensor * src, struct ggml_tensor * dst, bool prefer_events, bool hold_dst) {
    // A Metal blit can't reach a buffer owned by another device, so a multi-GPU layer
    // hand-off goes through host memory (GGML_METAL_CROSS_STAGING_DISABLE uses the generic copy).
    if (ctx_src->dev != ctx_dst->dev) {
        if (TOSH_ENV("GGML_METAL_CROSS_STAGING_DISABLE") != NULL) {
            return false;
        }

        // both buffers must be private for the cross-device blits
        static int use_events = -1;
        if (use_events < 0) {
            const char * v = TOSH_ENV("TOSH_MGPU_EVENTS");
            use_events = (v && v[0] == '1') ? 1 : 0;
        }
        // Infinity Fabric peer copy is opt-in until validated on bridged hardware:
        // a success that silently returns wrong data has no fallback.
        static int use_peer = -1;
        if (use_peer < 0) {
            const char * v = TOSH_ENV("TOSH_MGPU_PEER");
            use_peer = (v && v[0] == '1') ? 1 : 0;
        }
        ggml_backend_buffer_t ebsrc = src->view_src ? src->view_src->buffer : src->buffer;
        ggml_backend_buffer_t ebdst = dst->view_src ? dst->view_src->buffer : dst->buffer;
        // which of the three hand-offs actually runs is not visible otherwise, and the flag
        // alone does not tell: TOSH_MGPU_XDEV_COUNT reports the tally at exit
        static int count = -1;
        if (count < 0) {
            count = TOSH_ENV("TOSH_MGPU_XDEV_COUNT") != NULL;
            if (count) {
                atexit(tosh_xdev_report);
            }
        }

        if (!ggml_metal_buffer_is_shared(ebsrc->context) &&
            !ggml_metal_buffer_is_shared(ebdst->context)) {
            if (prefer_events && use_events && ggml_metal_cpy_xdev_events(ctx_src, ctx_dst, src, dst, hold_dst)) {
                atomic_fetch_add(&g_xdev_ev, 1);
                return true;
            }
            if (use_peer && ggml_metal_cpy_xdev_peer(ctx_src, ctx_dst, src, dst)) {
                atomic_fetch_add(&g_xdev_peer, 1);
                return true;
            }
            if (use_events && ggml_metal_cpy_xdev_events(ctx_src, ctx_dst, src, dst, hold_dst)) {
                atomic_fetch_add(&g_xdev_ev, 1);
                return true;
            }
        }
        atomic_fetch_add(&g_xdev_gen, 1);
        // drain both GPUs: the source before reading it, the destination before
        // overwriting an input its in-flight graph may still be consuming
        ggml_metal_synchronize(ctx_src);
        ggml_metal_synchronize(ctx_dst);

        ggml_backend_buffer_t bsrc = src->view_src ? src->view_src->buffer : src->buffer;
        ggml_backend_buffer_t bdst = dst->view_src ? dst->view_src->buffer : dst->buffer;
        ggml_metal_buffer_t buf_src = (ggml_metal_buffer_t) bsrc->context;
        ggml_metal_buffer_t buf_dst = (ggml_metal_buffer_t) bdst->context;

        const size_t size = ggml_nbytes(src);
        void * stage = malloc(size);
        if (!stage) {
            return false;
        }
        ggml_metal_buffer_get_tensor(buf_src, src, stage, 0, size);
        ggml_metal_buffer_set_tensor(buf_dst, dst, stage, 0, size);
        free(stage);
        return true;
    }

    @autoreleasepool {
        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(src);
        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(dst);

        if (bid_src.metal == nil || bid_dst.metal == nil) {
            return false;
        }

        id<MTLCommandQueue> dst_queue = ctx_dst->queue;
        id<MTLCommandBuffer> sync_cmd_buf = [dst_queue commandBuffer];

        ggml_metal_event_encode_signal(ctx_dst->ev_sync, sync_cmd_buf);

        [sync_cmd_buf commit];

        [ctx_dst->cmd_bufs_ext addObject:sync_cmd_buf];
        ctx_dst->cmd_buf_last = sync_cmd_buf;

        [sync_cmd_buf retain];

        // queue the copy operation into the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ctx_src->queue;
        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_cmd_all, 1);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        ggml_metal_encode_pending_waits(ctx_src, cmd_buf);
        ggml_metal_event_encode_wait(ctx_dst->ev_sync, cmd_buf);
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:bid_src.metal
                   sourceOffset:bid_src.offs
                       toBuffer:bid_dst.metal
              destinationOffset:bid_dst.offs
                           size:ggml_nbytes(src)];

        [encoder endEncoding];

        ggml_metal_event_t ev_cpy = ggml_metal_get_ev_cpy(ctx_src);
        ggml_metal_event_encode_signal(ev_cpy, cmd_buf);

        [cmd_buf commit];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx_src->cmd_bufs_ext addObject:cmd_buf];
        ctx_src->cmd_buf_last = cmd_buf;

        [cmd_buf retain];

        ggml_metal_event_wait(ctx_dst, ev_cpy);

        return true;
    }
}

bool ggml_metal_add_inplace_supported(ggml_metal_t ctx, const struct ggml_tensor * acc, const struct ggml_tensor * add) {
    if (ctx->has_error) {
        return false;
    }

    if (acc->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32) {
        return false;
    }

    if (!ggml_are_same_shape(acc, add) || !ggml_is_contiguous(acc) || !ggml_is_contiguous(add)) {
        return false;
    }

    if (ggml_metal_get_buffer_id(acc).metal == nil || ggml_metal_get_buffer_id(add).metal == nil) {
        return false;
    }

    return ggml_metal_library_get_pipeline_bin_one(ctx->lib, GGML_OP_ADD).pipeline != NULL;
}
    if (!ggml_metal_add_inplace_supported(ctx, acc, add)) {
        // a copy may be holding its buffer open for this add, and nothing else would submit it
        ggml_metal_cmd_buf_hold_flush(ctx);
        return false;
    }

    struct ggml_metal_buffer_id bid_acc = ggml_metal_get_buffer_id(acc);
    struct ggml_metal_buffer_id bid_add = ggml_metal_get_buffer_id(add);

    struct ggml_metal_pipeline_with_params pipeline = ggml_metal_library_get_pipeline_bin_one(ctx->lib, GGML_OP_ADD);

    ggml_metal_kargs_bin args = {
        /*.ne00 =*/ (int32_t) acc->ne[0],
        /*.ne01 =*/ (int32_t) acc->ne[1],
        /*.ne02 =*/ (int32_t) acc->ne[2],
        /*.ne03 =*/ (int32_t) acc->ne[3],
        /*.nb00 =*/ acc->nb[0],
        /*.nb01 =*/ acc->nb[1],
        /*.nb02 =*/ acc->nb[2],
        /*.nb03 =*/ acc->nb[3],
        /*.ne10 =*/ (int32_t) add->ne[0],
        /*.ne11 =*/ (int32_t) add->ne[1],
        /*.ne12 =*/ (int32_t) add->ne[2],
        /*.ne13 =*/ (int32_t) add->ne[3],
        /*.nb10 =*/ add->nb[0],
        /*.nb11 =*/ add->nb[1],
        /*.nb12 =*/ add->nb[2],
        /*.nb13 =*/ add->nb[3],
        /*.ne0  =*/ (int32_t) acc->ne[0],
        /*.ne1  =*/ (int32_t) acc->ne[1],
        /*.ne2  =*/ (int32_t) acc->ne[2],
        /*.ne3  =*/ (int32_t) acc->ne[3],
        /*.nb0  =*/ acc->nb[0],
        /*.nb1  =*/ acc->nb[1],
        /*.nb2  =*/ acc->nb[2],
        /*.nb3  =*/ acc->nb[3],
        /*.offs =*/ 0,
        /*.o1   =*/ { bid_add.offs },
    };

    // the kernel adds the fused offsets to the src1 base itself
    bid_add.offs = 0;

    if (pipeline.c4) {
        args.ne00 /= 4;
        args.ne10 /= 4;
        args.ne0  /= 4;
    }

        ggml_metal_encoder_t enc = ggml_metal_encoder_init((ggml_metal_cmd_buf_t) cmd_buf, false);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_acc, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_add, 2);
        ggml_metal_encoder_set_buffer  (enc, bid_acc, 3);

        if (pipeline.cnt) {
            ggml_metal_encoder_dispatch_threadgroups(enc, args.ne0, (int) ggml_nrows(acc), 1, 1, 1, 1);
        } else {
            const int nth_max = MIN(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

            int nth = 1;
            while (2*nth < args.ne0 && nth < nth_max) {
                nth *= 2;
            }

            ggml_metal_encoder_dispatch_threadgroups(enc, args.ne01, args.ne02, args.ne03, nth, 1, 1);
        }

        ggml_metal_encoder_end_encoding(enc);
        ggml_metal_encoder_free(enc);

    if (v < 0) { const char * s = TOSH_ENV("TOSH_MGPU_FB_ADD_RB"); v = (s && s[0] == '0') ? 0 : 1; }
// rows in the exchanged tensor: one for a generated token, the whole prompt for a prefill
static int64_t tosh_batch_of(const struct ggml_tensor * t) {
    if (t == NULL || t->ne[0] <= 0) { return 1; }
    return ggml_nelements(t)/t->ne[0];
}

        const char * s = TOSH_ENV("TOSH_MGPU_PEER_MIN_BATCH");
// Leaves the exchange's command buffer open so whatever the graph encodes next lands in it,
// one submission fewer. On its own this is neutral, which is what the first measurement said;
// it only pays together with the merge that makes the next subgraph continue the held buffer,
// and then only when the batch is small. A token of one row crosses 72 times and each crossing
// carries its own pair of command buffers, so halving them is worth ~3% of decode; a prompt
// amortises the submissions over the whole batch and loses ~2% instead, hence the batch gate.
// TOSH_MGPU_HOLD=1 forces it on, =0 off, unset follows the batch.
static bool ggml_metal_collective_hold_batch(int64_t batch) {
    static int v = -2;
    if (v == -2) {
        const char * s = TOSH_ENV("TOSH_MGPU_HOLD");
        v = s ? (s[0] == '1' ? 1 : 0) : -1;
    if (v >= 0) {
        return v == 1;
    }
    return batch < ggml_metal_peer_min_batch();
        const char * v = TOSH_ENV("TOSH_MGPU_PEER");
        if (seq > 2 && !ggml_metal_collective_nowait()) {
            if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_waits, 1);
        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_signals, 1);
        if (!ggml_metal_collective_nowait()) {
        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_waits, 1);
        }
            if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_signals, 1);
        [cmd_buf encodeSignalEvent:link_in->ev_x_done value:seq];
        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_signals, 1);
        tosh_trace_exchange(cmd_buf, seq, (int) (([(id<MTLDevice>) ggml_metal_device_get_obj(ctx_self->dev) peerIndex]) & 1));
        ggml_metal_cmd_buf_end(ctx_self, cmd_buf, ggml_metal_collective_hold_batch(tosh_batch_of(t_self)));
        if (seq > 2 && !ggml_metal_collective_nowait()) {
            if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_waits, 1);
        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_signals, 1);
        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_waits, 1);
        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_signals, 1);
        tosh_trace_exchange(cmd_buf, seq, (int) (([(id<MTLDevice>) ggml_metal_device_get_obj(ctx_self->dev) peerIndex]) & 1));
        ggml_metal_cmd_buf_end(ctx_self, cmd_buf, ggml_metal_collective_hold_batch(tosh_batch_of(t_self)));
        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_signals, 1);
        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_waits, 1);

            MTLCommandBufferStatus status = [cmd_buf status];
            if (status != MTLCommandBufferStatusCompleted) {
                GGML_LOG_ERROR("%s: error: command buffer %d failed with status %d\n", __func__, (int) i, (int) status);
                if (status == MTLCommandBufferStatusError) {
                    GGML_LOG_ERROR("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                }

        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_signals, 1);
        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_waits, 1);

                ctx->has_error = true;
                return;
            }

            [cmd_buf release];
        }
        tosh_trace_exchange(cmd_buf, seq, (int) (([(id<MTLDevice>) ggml_metal_device_get_obj(ctx_self->dev) peerIndex]) & 1));
        ggml_metal_cmd_buf_end(ctx_self, cmd_buf, ggml_metal_collective_hold_batch(tosh_batch_of(t_self)));

        [ctx->cmd_bufs_ext removeAllObjects];
        const char * s = TOSH_ENV("TOSH_MGPU_ONESHOT");
    }

    @autoreleasepool {
        [ctx->buf_refs removeAllObjects];
    }
}

static struct ggml_metal_buffer_id ggml_metal_get_buffer_id(const struct ggml_tensor * t) {
    if (!t) {
        return (struct ggml_metal_buffer_id) { nil, 0 };
    }

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    return ggml_metal_buffer_get_id(buffer->context, t);
}

void ggml_metal_set_tensor_async(ggml_metal_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    @autoreleasepool {
        // wrap the source data into a Metal buffer
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
        id<MTLBuffer> buf_src = [device newBufferWithBytes:data
                                                    length:size
                                                   options:MTLResourceStorageModeShared];

        GGML_ASSERT(buf_src);

        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(tensor);
        if (bid_dst.metal == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        bid_dst.offs += offset;

        // queue the copy operation into the queue of the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:buf_src
                   sourceOffset:0
                       toBuffer:bid_dst.metal
              destinationOffset:bid_dst.offs
                           size:size];

        [encoder endEncoding];
        [cmd_buf commit];

        [ctx->buf_refs addObject:buf_src];
        [buf_src release];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

void ggml_metal_get_tensor_async(ggml_metal_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    @autoreleasepool {
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
        id<MTLBuffer> buf_dst = [device newBufferWithBytesNoCopy:data
                                                          length:size
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];

        GGML_ASSERT(buf_dst);

        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(tensor);
        if (bid_src.metal == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        bid_src.offs += offset;

        // queue the copy operation into the queue of the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:bid_src.metal
                   sourceOffset:bid_src.offs
                       toBuffer:buf_dst
              destinationOffset:0
                           size:size];

        [encoder endEncoding];
        [cmd_buf commit];

        [ctx->buf_refs addObject:buf_dst];
        [buf_dst release];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    @autoreleasepool {
        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(src);
        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(dst);

        if (bid_src.metal == nil || bid_dst.metal == nil) {
            return false;
    if (tosh_mgpu_trace()) {
        tosh_decode_snapshot(t_self->ne[0] > 0 ? ggml_nelements(t_self)/t_self->ne[0] : 1);
    }

        const char * v = TOSH_ENV("TOSH_MGPU_PEER_FUSED");
        }

        id<MTLCommandQueue> dst_queue = ggml_metal_device_get_queue(ctx_dst->dev);
        id<MTLCommandBuffer> sync_cmd_buf = [dst_queue commandBuffer];

        ggml_metal_event_encode_signal(ctx_dst->ev_sync, sync_cmd_buf);

        [sync_cmd_buf commit];

        [ctx_dst->cmd_bufs_ext addObject:sync_cmd_buf];
        ctx_dst->cmd_buf_last = sync_cmd_buf;

        [sync_cmd_buf retain];

        // queue the copy operation into the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx_src->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        ggml_metal_event_encode_wait(ctx_dst->ev_sync, cmd_buf);
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_waits, 1);
            if (ggml_metal_bp_satisfied()) {
                [cmd_buf encodeWaitForEvent:tosh_wait_dummy(ctx_self) value:1];
            } else {
            }
        }

        id<MTLCounterSampleBuffer> cts = tosh_ct_buffer(ctx_self);
        uint64_t cbase = 0;
        const bool cbar = ggml_metal_ctime_barrier();
        if (cts != nil) {
            cbase = (atomic_fetch_add(&ctx_self->ct_idx, 1) % TOSH_CT_SLOTS)*TOSH_CT_PER;

        ggml_metal_event_t ev_cpy = ggml_metal_get_ev_cpy(ctx_src);
        ggml_metal_event_encode_signal(ev_cpy, cmd_buf);
        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_signals, 1);

        [cmd_buf commit];
        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_waits, 1);
        if (ggml_metal_wait_satisfied()) {
            [cmd_buf encodeWaitForEvent:tosh_wait_dummy(ctx_self) value:1];
        } else {
            [cmd_buf encodeWaitForEvent:link_in->ev_x_ready value:seq_in];
        }

        if (cts != nil) {
            id<MTLCounterSampleBuffer> csr = cts;
            const uint64_t cb2 = cbase;
            [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> cb) { (void) cb; tosh_ct_record(csr, cb2); }];
        }

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx_src->cmd_bufs_ext addObject:cmd_buf];
        ctx_src->cmd_buf_last = cmd_buf;
        tosh_trace_exchange(cmd_buf, seq, (int) (([(id<MTLDevice>) ggml_metal_device_get_obj(ctx_self->dev) peerIndex]) & 1));
        ggml_metal_cmd_buf_end(ctx_self, cmd_buf, ggml_metal_collective_hold_batch(tosh_batch_of(t_self)));

        [cmd_buf retain];

        ggml_metal_event_wait(ctx_dst, ev_cpy);

// TOSH_MGPU_MERGE=1: a tensor split cuts the graph at every partial node, so a token costs one
// command buffer for each collective plus one for each subgraph that follows it. With this on,
// the collective leaves its buffer open and the next subgraph encodes into it instead of
// submitting its own, which halves the command buffers a split token creates.
static bool ggml_metal_merge_graph(void) {
    static int v = -1;
    if (v < 0) {
        const char * s = TOSH_ENV("TOSH_MGPU_MERGE");
        // on by default: it is inert unless a collective left a buffer held, which only the
        // batch gate above does, so a prompt never sees it
        v = s ? (s[0] == '1' ? 1 : 0) : 1;
    }
    return v == 1;
}

enum ggml_status ggml_metal_graph_compute(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    if (tosh_mgpu_trace()) { atomic_fetch_add(&g_tr_graphs, 1); }
    // command buffers run in commit order, so a held one has to go in before this graph
    if (!ggml_metal_merge_graph()) {
        ggml_metal_cmd_buf_hold_flush(ctx);
    }

    if (ctx->has_error) {
        GGML_LOG_ERROR("%s: backend is in error state from a previous command buffer failure - recreate the backend to recover\n", __func__);
        return GGML_STATUS_FAILED;
    }

    if (gf->n_nodes == 0) {
        return GGML_STATUS_SUCCESS;
    }

    // number of nodes encoded by the main thread (empirically determined)
    // the split was tuned on unified memory; TOSH_N_MAIN overrides it to A/B the share here
    static int n_main_env = -2;
    if (n_main_env == -2) {
        const char * v = TOSH_ENV("TOSH_N_MAIN");
        n_main_env = v ? atoi(v) : -1;
    }
    const int n_main = n_main_env >= 0 ? n_main_env : MAX(64, 0.1*gf->n_nodes);

    // number of threads in addition to the main thread
    const int n_cb = ctx->n_cb;

    // keep the memory wired
    ggml_metal_device_rsets_keep_alive(ctx->dev);

    // submit the ggml compute graph to the GPU by creating command buffers and encoding the ops in them
    // the first n_nodes_0 are encoded and submitted for processing directly by the calling thread
    // while these nodes are processing, we start n_cb threads to enqueue the rest of the nodes
    // each thread creates it's own command buffer and enqueues the ops in parallel
    //
    // tests on M1 Pro and M2 Ultra using LLaMA models, show that optimal values for n_cb are 1 or 2

    struct tosh_cbp_graph * cbp = NULL;
    uint64_t cbp_t0 = 0;
    if (tosh_cbp_on()) {
        struct tosh_cbp_bucket * b = tosh_cbp_bucket_for(gf->n_nodes);
        if (b != NULL) {
            cbp = (struct tosh_cbp_graph *) calloc(1, sizeof(struct tosh_cbp_graph));
            if (cbp != NULL) {
                cbp->bucket = b;
                atomic_store_explicit(&cbp->n_left, 1, memory_order_relaxed); // released below
                atomic_fetch_add_explicit(&b->n_graphs, 1, memory_order_relaxed);
                cbp_t0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
            }
        }
    }

    @autoreleasepool {
        ctx->gf = gf;

        if (ctx->n_cb == 0) {
            // single-threaded encoding: the whole graph is encoded by one command buffer
            ctx->n_nodes_0      = gf->n_nodes;
            ctx->n_nodes_1      = 0;
            ctx->n_nodes_per_cb = 0;
        } else {
            ctx->n_nodes_0      = MIN(n_main, gf->n_nodes);
            ctx->n_nodes_1      = gf->n_nodes - ctx->n_nodes_0;

            ctx->n_nodes_per_cb = (ctx->n_nodes_1 + ctx->n_cb - 1) / ctx->n_cb;
        }

        if (ctx->capture_compute >= 0) {
            ctx->capture_compute--;
        }

        const bool use_capture = ctx->capture_compute == 0;
        if (use_capture) {
            // make sure all previous computations have finished before starting the capture
            if (ctx->cmd_buf_last) {
                [ctx->cmd_buf_last waitUntilCompleted];
                ctx->cmd_buf_last = nil;
            }

            if (!ctx->capture_started) {
                NSString * path = [NSString stringWithFormat:@"/tmp/perf-metal-%d.gputrace", getpid()];

                GGML_LOG_WARN("%s: capturing graph in %s\n", __func__, [path UTF8String]);

                // create capture scope
                id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
                ctx->capture_scope = [[MTLCaptureManager sharedCaptureManager] newCaptureScopeWithDevice:device];

                MTLCaptureDescriptor * descriptor = [MTLCaptureDescriptor new];
                descriptor.captureObject = ctx->capture_scope;
                descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
                descriptor.outputURL = [NSURL fileURLWithPath:path];

                NSError * error = nil;
                if (![[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:descriptor error:&error]) {
                    GGML_LOG_ERROR("%s: error: unable to start capture '%s' (did you set METAL_CAPTURE_ENABLED=1 ?)\n", __func__, [[error localizedDescription] UTF8String]);
                } else {
                    [ctx->capture_scope beginScope];
                    ctx->capture_started = true;
                }
            }
        }

        // short-hand
        id<MTLCommandQueue> queue = ctx->queue;

        const uint64_t cbp_tc = cbp != NULL ? clock_gettime_nsec_np(CLOCK_UPTIME_RAW) : 0;

        // the main thread commits the first few commands immediately
        // cmd_buf[n_cb]
        {
            // continue the collective's buffer when it left one open, otherwise start a new one
            id<MTLCommandBuffer> cmd_buf = nil;
            if (ggml_metal_merge_graph() && ctx->cmd_buf_hold != nil) {
                cmd_buf = ctx->cmd_buf_hold;
                ctx->cmd_buf_hold = nil;
                // the collective is already encoded in here; anything that queued a wait since
                // then has to land between it and this subgraph, not be dropped
                if (tosh_mgpu_trace()) {
                    if (ctx->n_ev_wait_pending > 0 || ctx->n_peer_waits > 0) {
                        atomic_fetch_add(&g_tr_merge_waits, ctx->n_ev_wait_pending + ctx->n_peer_waits);
                    }
                    atomic_fetch_add(&g_tr_merge_reuse, 1);
                }
            } else {
                if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_cmd_all, 1);
                cmd_buf = [queue commandBufferWithUnretainedReferences];
                [cmd_buf retain];
            }
            ggml_metal_encode_pending_waits(ctx, cmd_buf);
            tosh_cbp_attach(cmd_buf, cbp);

            if (ctx->cmd_bufs[n_cb].obj) {
                [ctx->cmd_bufs[n_cb].obj release];
            }
            ctx->cmd_bufs[n_cb].obj = cmd_buf;

            tosh_trace_graph_end(cmd_buf, (int) (([(id<MTLDevice>) ggml_metal_device_get_obj(ctx->dev) peerIndex]) & 1));

            [cmd_buf enqueue];

            ctx->encode_async(n_cb);
        }

        // remember the command buffer for the next iteration
        ctx->cmd_buf_last = ctx->cmd_bufs[n_cb].obj;

        // the main thread already encoded every node: the remaining command buffers would be
        // created, dispatched and committed empty. Tensor split hits this on every subgraph.
        const bool encoded_by_main = ctx->n_nodes_1 == 0 && !use_capture;

        if (!encoded_by_main) {
        // prepare the rest of the command buffers asynchronously (optional)
        // cmd_buf[0.. n_cb)
        for (int cb_idx = 0; cb_idx < n_cb; ++cb_idx) {
            if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_cmd_all, 1);
            id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
            ggml_metal_encode_pending_waits(ctx, cmd_buf);
            tosh_cbp_attach(cmd_buf, cbp);
            [cmd_buf retain];

            if (ctx->cmd_bufs[cb_idx].obj) {
                [ctx->cmd_bufs[cb_idx].obj release];
            }
            ctx->cmd_bufs[cb_idx].obj = cmd_buf;

            // always enqueue the first two command buffers
            // enqueue all of the command buffers if we don't need to abort
            if (cb_idx < 2 || ctx->abort_callback == NULL) {
                [cmd_buf enqueue];

                // update the pointer to the last queued command buffer
                // this is needed to implement synchronize()
                ctx->cmd_buf_last = cmd_buf;
            }
        }

        if (cbp != NULL) {
            atomic_fetch_add_explicit(&g_cbp_create_ns,
                    clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - cbp_tc, memory_order_relaxed);
        }

        dispatch_apply(n_cb, ctx->d_queue, ctx->encode_async);
        }

        // for debugging: block until graph is computed
        //[ctx->cmd_buf_last waitUntilCompleted];

        // enter here only when capturing in order to wait for all computation to finish
        // otherwise, we leave the graph to compute asynchronously
        if (use_capture && ctx->capture_started) {
            // wait for completion and check status of each command buffer
            // needed to detect if the device ran out-of-memory for example (#1881)
            {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[n_cb].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, n_cb, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }

                    return GGML_STATUS_FAILED;
                }
            }

            for (int i = 0; i < n_cb; ++i) {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[i].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, i, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }

                    return GGML_STATUS_FAILED;
                }

                id<MTLCommandBuffer> next_buffer = (i + 1 < n_cb ? ctx->cmd_bufs[i + 1].obj : nil);
                if (!next_buffer) {
                    continue;
                }

                const bool next_queued = ([next_buffer status] != MTLCommandBufferStatusNotEnqueued);
                if (next_queued) {
                    continue;
                }

                if (ctx->abort_callback && ctx->abort_callback(ctx->abort_callback_data)) {
                    GGML_LOG_INFO("%s: command buffer %d aborted", __func__, i);
                    return GGML_STATUS_ABORTED;
                }

                [next_buffer commit];
            }

            [ctx->capture_scope endScope];
            [[MTLCaptureManager sharedCaptureManager] stopCapture];

            ctx->capture_started = false;
        }
    }

    if (cbp != NULL) {
        atomic_fetch_add_explicit(&cbp->bucket->enc_ns,
                clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - cbp_t0, memory_order_relaxed);
        // drop the reference held while encoding; the last completion handler frees it
        if (atomic_fetch_sub_explicit(&cbp->n_left, 1, memory_order_acq_rel) == 1) {
            free(cbp);
        }
    }

    return GGML_STATUS_SUCCESS;
}

// the reorder gains on dense decode but disturbs the expert chain on MoE, so this stays
// narrow on purpose
static bool ggml_metal_graph_is_moe_decode(const struct ggml_cgraph * gf, int max_tokens) {
    bool has_experts = false;

    for (int i = 0; i < gf->n_nodes; i++) {
        const struct ggml_tensor * node = gf->nodes[i];

        // mul_mat_id carries the experts in ne[1] and the tokens in ne[2]
        const int dim = node->op == GGML_OP_MUL_MAT ? 1 : (node->op == GGML_OP_MUL_MAT_ID ? 2 : -1);
        if (dim < 0) {
            continue;
        }

        if (node->src[1] && node->src[1]->ne[dim] > max_tokens) {
            return false;
        }

        has_experts |= node->op == GGML_OP_MUL_MAT_ID;
    }

    return has_experts;
}

void ggml_metal_graph_optimize(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    //const int64_t t_start = ggml_time_us();

    if (ctx->use_graph_optimize) {
        ggml_graph_optimize(gf);
        static int max_tokens = -1;
        if (max_tokens < 0) {
            const char * s = TOSH_ENV("GGML_METAL_GRAPH_OPTIMIZE_MIN_BATCH");
            max_tokens = s ? atoi(s) - 1 : 7;
        }
            ggml_graph_optimize(gf);
        }
    }

    //printf("%s: graph optimize took %.3f ms\n", __func__, (ggml_time_us() - t_start) / 1000.0);
}

void ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = ctx->queue;
        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_cmd_all, 1);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        ggml_metal_encode_pending_waits(ctx, cmd_buf);

        ggml_metal_event_encode_signal(ev, cmd_buf);

        [cmd_buf commit];

        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

void ggml_metal_event_wait(ggml_metal_t ctx, ggml_metal_event_t ev) {
    if (ggml_metal_defer_waits() && ctx->n_ev_wait_pending < GGML_METAL_MAX_PENDING_WAITS) {
        ctx->ev_wait_pending[ctx->n_ev_wait_pending++] = ev;
        return;
    }

    @autoreleasepool {
        id<MTLCommandQueue> queue = ctx->queue;
        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_cmd_all, 1);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        ggml_metal_encode_pending_waits(ctx, cmd_buf);

        ggml_metal_event_encode_wait(ev, cmd_buf);

        [cmd_buf commit];

        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx) {
    return ctx->ev_cpy;
}

void ggml_metal_set_n_cb(ggml_metal_t ctx, int n_cb) {
    // when fusion stats are collected the graph must be encoded by a single thread so the
    // counters are race-free; override whatever the caller requested
    if (ggml_metal_fusion_info_stats(ctx->finfo)) {
        n_cb = 0;
    }

    if (ctx->n_cb != n_cb) {
        ctx->n_cb = MIN(n_cb, GGML_METAL_MAX_COMMAND_BUFFERS);

        if (ctx->n_cb > 2) {
            GGML_LOG_WARN("%s: n_cb = %d, using n_cb > 2 is not recommended and can degrade the performance in some cases\n", __func__, n_cb);
        }
    }

    if (ctx->encode_async) {
        Block_release(ctx->encode_async);
    }

    ctx->encode_async = Block_copy(^(size_t iter) {
        const int cb_idx = iter;
        const int n_cb_l = ctx->n_cb;

        const int n_nodes_0 = ctx->n_nodes_0;
        const int n_nodes_1 = ctx->n_nodes_1;

        const int n_nodes_per_cb = ctx->n_nodes_per_cb;

        int idx_start = 0;
        int idx_end   = n_nodes_0;

        if (cb_idx < n_cb_l) {
            idx_start = n_nodes_0 + (                                         (cb_idx + 0) * n_nodes_per_cb);
            idx_end   = n_nodes_0 + (MIN((cb_idx == n_cb_l - 1) ? n_nodes_1 : (cb_idx + 1) * n_nodes_per_cb, n_nodes_1));
        }

        id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;

        ggml_metal_op_t ctx_op = ggml_metal_op_init(
            ctx->dev,
            cmd_buf,
            ctx->gf,
            ctx->finfo,
            idx_start,
            idx_end,
            ctx->use_concurrency,
            ctx->capture_compute == 0,
            ctx->debug_graph);

        const bool cbp_on = tosh_cbp_on();
        const uint64_t cbp_t0 = cbp_on ? clock_gettime_nsec_np(CLOCK_UPTIME_RAW) : 0;

        for (int idx = 0; idx < ggml_metal_op_n_nodes(ctx_op); ++idx) {
            const int res = ggml_metal_op_encode(ctx_op, idx);
            if (res == 0) {
                break;
            }

            idx += res - 1;
        }

        ggml_metal_op_free(ctx_op);

        const uint64_t cbp_t1 = cbp_on ? clock_gettime_nsec_np(CLOCK_UPTIME_RAW) : 0;
        if (cb_idx < 2 || ctx->abort_callback == NULL) {
            [cmd_buf commit];
        }
        if (cbp_on) {
            const uint64_t cbp_t2 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
            atomic_fetch_add_explicit(&g_cbp_ops_ns,    cbp_t1 - cbp_t0, memory_order_relaxed);
            atomic_fetch_add_explicit(&g_cbp_commit_ns, cbp_t2 - cbp_t1, memory_order_relaxed);
        }
    });
}

void ggml_metal_set_abort_callback(ggml_metal_t ctx, ggml_abort_callback abort_callback, void * user_data) {
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = user_data;
}

bool ggml_metal_supports_family(ggml_metal_t ctx, int family) {
    GGML_ASSERT(ctx->dev != nil);

    id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);

    return [device supportsFamily:(MTLGPUFamilyApple1 + family - 1)];
}

void ggml_metal_capture_next_compute(ggml_metal_t ctx) {
    ctx->capture_compute = 1;
}
