#include "tosh-plan.h"

#include "common.h"
#include "fit.h"
#include "log.h"

#include "gguf.h"
#include "llama.h"

#ifdef TOSH_ENABLE_DYNAMIC_MOE
#include "tosh-dmoe.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <regex>

#ifdef __APPLE__
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

static constexpr double MiB = 1048576.0;

// safety reserve: a share of the card, bounded on small and large cards
static constexpr double RESERVE_FRACTION = 0.08;
static constexpr double RESERVE_MIN_MIB  = 768.0;
static constexpr double RESERVE_MAX_MIB  = 1536.0;
// RAM left to the rest of the system once a plan's host memory is taken
static constexpr double HOST_MARGIN_MIB  = 2048.0;

namespace {

struct moe_info {
    bool   moe       = false;
    int    n_layer   = 0;
    int    n_moe     = 0;       // layers with an expert bank
    int    n_expert  = 0;
    int    n_used    = 0;
    int    n_ctx_train = 0;
    double bank_mib  = 0;       // every expert weight
    double max_tensor_mib = 0;  // one layer's largest expert tensor: a prefill slot
    double embd_mib  = 0;       // token embeddings, kept in host memory
};

struct host_mem {
    double ram_mib        = 0;
    double wired_mib      = 0;
    double wire_limit_mib = 0;
    double available_mib  = 0;   // what the system can hand out now without compressing or swapping
};

struct probe {
    bool   ok = false;
    double model = 0, context = 0, compute = 0;   // GPU, MiB
    double host_model = 0, host_other = 0;
};

double env_mib(const char * k, double d) {
    const char * v = getenv(k);
    return v ? atof(v) : d;
}

moe_info read_moe_info(const char * path) {
    moe_info mi;
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * g = gguf_init_from_file(path, gp);
    if (g == nullptr) {
        return mi;
    }
    auto get_u32 = [&](const std::string & k) -> int {
        const int64_t id = gguf_find_key(g, k.c_str());
        return id < 0 ? 0 : (int) gguf_get_val_u32(g, id);
    };
    const int64_t aid = gguf_find_key(g, "general.architecture");
    const std::string arch = aid < 0 ? "" : gguf_get_val_str(g, aid);
    mi.n_layer     = get_u32(arch + ".block_count");
    mi.n_expert    = get_u32(arch + ".expert_count");
    mi.n_used      = get_u32(arch + ".expert_used_count");
    mi.n_ctx_train = get_u32(arch + ".context_length");

    const std::regex bank("^blk\\.([0-9]+)\\.ffn_(up|down|gate|gate_up)_exps\\.weight$");
        }
        }
    gguf_free(g);
    return mi;
}

host_mem read_host_info() {
    host_mem h;
#ifdef __APPLE__
    uint64_t mem = 0; size_t len = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0) h.ram_mib = mem/MiB;
    uint64_t lim = 0; len = sizeof(lim);
    if (sysctlbyname("vm.user_wire_limit", &lim, &len, nullptr, 0) == 0) h.wire_limit_mib = lim/MiB;
    vm_statistics64_data_t vs;
    mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &vs, &cnt) == KERN_SUCCESS) {
        h.wired_mib = (double) vs.wire_count*vm_page_size/MiB;
        h.available_mib = (double) (vs.free_count + vs.inactive_count + vs.speculative_count + vs.purgeable_count)*vm_page_size/MiB;
    }
#endif
    const double sim = env_mib("TOSH_AUTO_SIM_RAM_MIB", 0);
    if (sim > 0) {
        h.wire_limit_mib = h.wire_limit_mib > 0 ? h.wire_limit_mib*sim/h.ram_mib : 0;
        h.available_mib = std::min(h.available_mib, sim);
        h.ram_mib = sim;
    }
    return h;
}

// keeps override patterns alive for the lifetime of the params that point at them
std::list<std::string> & pattern_store() {
    static std::list<std::string> s;
    return s;
}

const char * experts_pattern() {
    // weights only: expert biases are small and stay next to the ops that add them
    return "\\.ffn_(up|down|gate|gate_up)_exps\\.weight$";
}

probe run_probe(const common_params & params, int n_ctx, ggml_type kv, int ub, bool experts_on_host, int ncmoe) {
    probe p;
    llama_model_params   mp = common_model_params_to_llama(const_cast<common_params &>(params));
    llama_context_params cp = common_context_params_to_llama(params);
    mp.n_gpu_layers = 999;
    // repacked weights are not host memory the executor or the scheduler's uploads can read
    mp.use_extra_bufts = !(experts_on_host || ncmoe > 0) && mp.use_extra_bufts;
    std::vector<llama_model_tensor_buft_override> ov;
    if (experts_on_host) {
        ov.push_back({ experts_pattern(), ggml_backend_cpu_buffer_type() });
    }
    for (int i = 0; i < ncmoe; ++i) {
        pattern_store().push_back(llm_ffn_block_regex(i, LLM_FFN_EXPS_REGEX));
        ov.push_back({ pattern_store().back().c_str(), ggml_backend_cpu_buffer_type() });
    }
    ov.push_back({ nullptr, nullptr });
    mp.tensor_buft_overrides = ov.size() > 1 ? ov.data() : nullptr;
    cp.n_ctx    = n_ctx;
    cp.n_ubatch = ub;
    cp.n_batch  = std::max<uint32_t>(cp.n_batch, ub);
    cp.type_k   = kv;
    cp.type_v   = kv;
    try {
        std::vector<ggml_backend_dev_t> devs;
        uint32_t ngl = 0, ctx_train = 0, n_expert = 0;
        const auto mem = common_get_device_memory_data(params.model.path.c_str(), &mp, &cp, devs, ngl, ctx_train, n_expert,
                                                       GGML_LOG_LEVEL_ERROR);
        if (mem.size() < 2) {
            return p;
        }
        p.model   = mem[0].model/MiB;
        p.context = mem[0].context/MiB;
        p.compute = mem[0].compute/MiB;
        p.host_model = mem.back().model/MiB;
        p.host_other = (mem.back().context + mem.back().compute)/MiB;
        p.ok = true;
    } catch (const std::exception & e) {
        LOG_WRN("%s: memory probe failed: %s\n", __func__, e.what());
    }
    return p;
}

bool read_vram(double & free_mib, double & total_mib) {
    free_mib = total_mib = 0;
#ifdef TOSH_ENABLE_DYNAMIC_MOE
    size_t f = 0, t = 0;
    // free before this process allocated anything; the probes allocate nothing
    if (tosh_dmoe_vram(&f, &t) || tosh_vram_read(&f, &t)) {
        free_mib = f/MiB; total_mib = t/MiB;
    }
#endif
    if (total_mib <= 0) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (dev == nullptr) return false;
        size_t f = 0, t = 0;
        ggml_backend_dev_memory(dev, &f, &t);
        free_mib = f/MiB; total_mib = t/MiB;
    }
    const double sim = env_mib("TOSH_AUTO_SIM_VRAM_MIB", 0);
    if (sim > 0) {
        // what other processes hold on the real card is kept as the simulated card's baseline
        free_mib = std::max(0.0, sim - (total_mib - free_mib));
        total_mib = sim;
    }
    return total_mib > 0;
}

const char * kv_name(ggml_type t) { return ggml_type_name(t); }

} // namespace

const char * tosh_plan_state_name(tosh_plan_state s) {
    switch (s) {
        case TOSH_PLAN_NOT_PLANNED:            return "NOT_PLANNED";
        case TOSH_PLAN_FULL_GPU_OPTIMAL:       return "FULL_GPU_OPTIMAL";
        case TOSH_PLAN_DMOE_CONTEXT_OPTIMAL:   return "DMOE_CONTEXT_OPTIMAL";
        case TOSH_PLAN_DMOE_CAPACITY_REQUIRED: return "DMOE_CAPACITY_REQUIRED";
        case TOSH_PLAN_LEGACY_OFFLOAD_BETTER:  return "LEGACY_OFFLOAD_BETTER";
        case TOSH_PLAN_UNSUPPORTED:            return "UNSUPPORTED";
    }
    return "?";
}

static std::string jstr(const std::string & s) {
    std::string o = "\"";
    for (char c : s) o += (c == '"' || c == '\\') ? '\'' : c;
    return o + "\"";
}

std::string tosh_plan_json(const tosh_plan & p) {
    std::string o = "{";
    o += "\"state\": " + jstr(tosh_plan_state_name(p.state));
    o += ", \"mode\": " + jstr(p.mode);
    o += ", \"reason\": " + jstr(p.reason);
    o += ", \"fallback\": " + jstr(p.fallback);
    o += ", \"kv\": " + jstr(kv_name(p.kv));
    o += string_format(", \"n_ctx\": %d, \"ubatch\": %d, \"ncmoe\": %d", p.n_ctx, p.ubatch, p.ncmoe);
    o += string_format(", \"reserve_mib\": %.0f, \"arena_mib\": %.0f, \"min_arena_mib\": %.0f", p.reserve_mib, p.arena_mib, p.min_arena_mib);
    o += string_format(", \"projected_private_mib\": %.0f, \"projected_free_mib\": %.0f", p.private_mib, p.free_mib);
    o += string_format(", \"vram_total_mib\": %.0f, \"vram_free_mib\": %.0f", p.vram_total_mib, p.vram_free_mib);
    o += string_format(", \"host_required_mib\": %.0f, \"host_ram_mib\": %.0f, \"host_available_mib\": %.0f, \"host_reserve_mib\": %.0f, \"bank_mib\": %.0f",
                       p.host_required_mib, p.host_ram_mib, p.host_available_mib, p.host_reserve_mib, p.bank_mib);
    o += ", \"mlock\": " + jstr(p.mlock) + ", \"dispersion\": " + jstr(p.dispersion);
    o += ", \"candidates\": [";
    for (size_t i = 0; i < p.candidates.size(); ++i) {
        const auto & c = p.candidates[i];
        o += i ? ", {" : "{";
        o += "\"name\": " + jstr(c.name) + ", \"mode\": " + jstr(c.mode) + ", \"kv\": " + jstr(kv_name(c.kv)) + ", \"valid\": " + (c.valid ? "true" : "false") + ", \"reason\": " + jstr(c.reason);
        o += string_format(", \"ubatch\": %d, \"ncmoe\": %d, \"private_mib\": %.0f, \"free_mib\": %.0f, \"arena_mib\": %.0f, \"kv_mib\": %.0f, \"compute_mib\": %.0f}",
                           c.ubatch, c.ncmoe, c.private_mib, c.free_mib, c.arena_mib, c.kv_mib, c.compute_mib);
    }
    return o + "]}";
}

static void plan_log(const tosh_plan & p) {
            p.n_ctx, p.vram_total_mib/1024, p.vram_free_mib/1024, p.reserve_mib/1024);
    for (const auto & c : p.candidates) {
                c.private_mib, c.free_mib, c.arena_mib, c.valid ? "valid" : "rejected", c.reason.c_str());
    }
}

static void add_override(common_params & params, const char * pattern) {
    auto & ov = params.tensor_buft_overrides;
    for (auto & o : ov) {
        if (o.pattern == nullptr) {
            o = { pattern, ggml_backend_cpu_buffer_type() };
            break;
        }
    }
    if (ov.empty() || ov.back().pattern != nullptr) {
        ov.push_back({ nullptr, nullptr });
    }
}

bool tosh_plan_apply(common_params & params, tosh_plan * out) {
    const char * mode_env = getenv("TOSH_AUTO");
    if (mode_env == nullptr || strcmp(mode_env, "0") == 0 || strcmp(mode_env, "off") == 0) {
        return false;
    }
    if (getenv("TOSH_DMOE_CACHE_MIB") != nullptr || getenv("TOSH_DMOE_RESIDENCY") != nullptr) {
        LOG_WRN("%s: TOSH_DMOE_* set by hand, the auto plan steps aside\n", __func__);
        return false;
    }
    std::string force = (strcmp(mode_env, "1") == 0 || strcmp(mode_env, "auto") == 0) ? "" : mode_env;
    // nodmoe: auto without mixed execution, the retry after the expert bank could not be locked
    const bool no_dmoe = force == "nodmoe";
    if (no_dmoe) force.clear();

    tosh_plan p;
    const moe_info mi = read_moe_info(params.model.path.c_str());
    if (!mi.moe) {
        if (out) *out = p;
        return false;
    }

    p.n_ctx = params.n_ctx > 0 ? params.n_ctx : std::min(mi.n_ctx_train, 32768);
    p.bank_mib = mi.bank_mib;
    const double per_expert = mi.bank_mib/std::max(1, mi.n_moe*mi.n_expert);
    // below one token's worth of experts the arena mostly misses and the host does the work
    p.min_arena_mib = std::max(256.0, env_mib("TOSH_AUTO_MIN_ARENA_MIB", mi.n_moe*mi.n_used*per_expert));
    // each chunk touches most experts when every token picks a large share of them
    const bool high = mi.n_expert > 0 && (double) mi.n_used/mi.n_expert > 1.0/16;
    p.dispersion = high ? "high" : "normal";


    // cache types to consider
    std::vector<ggml_type> kvs;
    const std::string kv_env = getenv("TOSH_AUTO_KV") ? getenv("TOSH_AUTO_KV") : "auto";
    if (kv_env == "auto") {
        kvs = { GGML_TYPE_F16, GGML_TYPE_Q8_0 };
    } else {
        const ggml_type t = kv_env == "f16" ? GGML_TYPE_F16 : kv_env == "q8_0" ? GGML_TYPE_Q8_0 :
                            kv_env == "turbo4" ? GGML_TYPE_TURBO4_0 : GGML_TYPE_COUNT;
        if (t == GGML_TYPE_COUNT) {
            LOG_WRN("%s: unknown TOSH_AUTO_KV=%s, using auto\n", __func__, kv_env.c_str());
            kvs = { GGML_TYPE_F16, GGML_TYPE_Q8_0 };
        } else {
            kvs = { t };
        }
    }

    // measured, not estimated: each configuration is loaded without allocating. The probe with the
    // experts in host memory places ops as the real run does; the full model is that plus the bank,
    // since a probe of the full model puts the expert ops on the CPU and overstates compute.
    const int ubs[3] = { 512, 1024, 2048 };
    const int ub_full = params.n_ubatch;
    probe dm[4][4];
    bool dmoe_built = false;
#ifdef TOSH_ENABLE_DYNAMIC_MOE
    dmoe_built = true;
#endif
    std::vector<std::pair<ggml_type, probe>> full;
    for (size_t k = 0; k < kvs.size(); ++k) {
        for (int u = 0; u < 3; ++u) dm[k][u] = run_probe(params, p.n_ctx, kvs[k], ubs[u], true, 0);
        int uf = -1;
        for (int u = 0; u < 3; ++u) if (ubs[u] == ub_full) uf = u;
        const probe pf = uf >= 0 ? dm[k][uf] : (dm[k][3] = run_probe(params, p.n_ctx, kvs[k], ub_full, true, 0));
        probe f = pf;
        f.model += mi.bank_mib;
        f.host_model = std::max(0.0, f.host_model - mi.bank_mib);
        full.push_back({ kvs[k], f });
    }
    if (!read_vram(p.vram_free_mib, p.vram_total_mib)) {
        LOG_WRN("%s: no GPU memory reading, planning off\n", __func__);
        return false;
    }
    const double host_dmoe   = mi.bank_mib + host_engine;
    std::string dmoe_block;
    if (!dmoe_built) {
        dmoe_block = "engine built without Dynamic MoE";
    } else if (hi.ram_mib > 0 && host_dmoe + p.host_reserve_mib > hi.ram_mib) {
        dmoe_block = string_format("host RAM: %.1f GiB needed + %.1f GiB reserve > %.1f GiB",
                                   host_dmoe/1024, p.host_reserve_mib/1024, hi.ram_mib/1024);
    } else if (hi.available_mib > 0 && host_dmoe + HOST_MARGIN_MIB > hi.available_mib) {
        dmoe_block = string_format("host RAM: %.1f GiB needed, %.1f GiB available now", host_dmoe/1024, hi.available_mib/1024);
    }
    p.mlock = dmoe_block.rfind("DMOE_HOST_BANK_NOT_LOCKABLE", 0) == 0 ? "not_lockable" : "required";

            // the weights kept in RAM and the staging are locked too
            bounded_warm = std::min({ mi.bank_mib, budget, p.lockable_mib - host_engine - staging_mib });
    const double slots    = 3.0*mi.max_tensor_mib;   // prefill uploads
    const double overhead = env_mib("TOSH_DMOE_OVERHEAD_MIB", 64);
    const double weights_full = full[0].second.ok ? full[0].second.model : mi.bank_mib;
    const bool weights_fit = weights_full + 512.0 <= p.vram_free_mib;

    auto evaluate = [&](double R) {
        p.candidates.clear();
        p.reserve_mib = R;
        for (auto & [kv, pr] : full) {
            tosh_plan_candidate c;
            c.name = std::string("full_gpu+") + kv_name(kv); c.mode = "full_gpu"; c.kv = kv; c.ubatch = ub_full;
            c.private_mib = pr.model + pr.context + pr.compute;
            c.free_mib    = p.vram_free_mib - c.private_mib;
            c.kv_mib = pr.context; c.compute_mib = pr.compute;
            c.valid  = pr.ok && c.free_mib >= R;
            c.reason = !pr.ok ? "probe failed" : c.valid ? "fits with the reserve" :
            p.candidates.push_back(c);
        }
        for (size_t k = 0; k < kvs.size() && dmoe_built; ++k) {
            for (int u = 0; u < 3; ++u) {
                const probe & pr = dm[k][u];
                tosh_plan_candidate c;
                c.name = std::string("dmoe+") + kv_name(kvs[k]); c.mode = "dmoe"; c.kv = kvs[k]; c.ubatch = ubs[u];
                const double base = pr.model + pr.context + pr.compute;
                c.arena_mib   = std::max(0.0, p.vram_free_mib - base - slots - overhead - R);
                c.private_mib = base + slots + overhead + c.arena_mib;
                c.free_mib    = p.vram_free_mib - c.private_mib;
                c.kv_mib = pr.context; c.compute_mib = pr.compute;
                p.candidates.push_back(c);
            }
        }
        // legacy: the fewest layers of experts on the host that leave the reserve
        for (auto & [kv, pr] : full) {
            tosh_plan_candidate c;
            c.name = std::string("legacy+") + kv_name(kv); c.mode = "legacy_offload"; c.kv = kv; c.ubatch = ub_full;
            const double per_layer = mi.bank_mib/std::max(1, mi.n_moe);
            const double full_priv = pr.model + pr.context + pr.compute;
            for (int n = 1; n <= mi.n_layer && pr.ok; ++n) {
                const double priv = full_priv - n*per_layer + slots;
                if (p.vram_free_mib - priv >= R) {
                    c.ncmoe = n; c.private_mib = priv; c.free_mib = p.vram_free_mib - priv; c.valid = true;
                    break;
                }
            }
            c.kv_mib = pr.context; c.compute_mib = pr.compute;
                string_format("%d layers of experts on the host", c.ncmoe);
            c.valid = c.valid && host_ok;
            p.candidates.push_back(c);
        }
    };

    auto find = [&](const std::string & mode, ggml_type kv, int ub) -> tosh_plan_candidate * {
        for (auto & c : p.candidates) if (c.mode == mode && c.kv == kv && (ub == 0 || c.ubatch == ub)) return &c;
        return nullptr;
    };

    // the ubatch for a DMoE candidate: larger ones speed prefill but take compute memory from the arena
    auto pick_ub = [&](ggml_type kv) -> tosh_plan_candidate * {
        tosh_plan_candidate * c512 = find("dmoe", kv, 512);
        if (c512 == nullptr || !c512->valid) return nullptr;
        const double a = c512->arena_mib;
        tosh_plan_candidate * c1k = find("dmoe", kv, 1024);
        tosh_plan_candidate * c2k = find("dmoe", kv, 2048);
        // what a larger ubatch takes from the arena is paid in decode misses, as a share of the bank
        const double cost = 0.10*mi.bank_mib;
        if (high && c2k && c2k->valid && a - c2k->arena_mib <= cost) return c2k;
        if (p.n_ctx <= 32768 && c1k && c1k->valid && a - c1k->arena_mib <= cost) return c1k;
        return c512;
    };

    const tosh_plan_candidate * sel = nullptr;
    // VRAM kept free after every known allocation, for the desktop and other apps; the plan
    // already counts weights, KV, compute, prefill slots and the driver margin
    // an MTP head adds its own context and buffers after the plan; past the reserve the driver pages
    // GPU memory into wired RAM, which slows every token and can use up the host
    const auto & spec_types = params.speculative.types;
    const bool mtp = std::find(spec_types.begin(), spec_types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != spec_types.end();
        evaluate(R);
        const tosh_plan_candidate * f16 = find("full_gpu", kvs[0], 0);
        const tosh_plan_candidate * q8  = kvs.size() > 1 ? find("full_gpu", kvs[1], 0) : nullptr;
        if (force.empty() || force == "full") {
            if (f16 && f16->valid) { sel = f16; p.reason = "the model and the requested context fit with the reserve"; }
            else if (q8 && q8->valid) { sel = q8; p.reason = "q8_0 cache makes the full model fit with the reserve"; }
        }
        if (sel == nullptr && !no_dmoe && (force.empty() || force == "dmoe")) {
            tosh_plan_candidate * a = pick_ub(kvs[0]);
            tosh_plan_candidate * b = kvs.size() > 1 ? pick_ub(kvs[1]) : nullptr;
            // the q8_0 cache is worth it when its savings buy a material share of the bank
            if (a && b && a->arena_mib < 0.9*mi.bank_mib && b->arena_mib - a->arena_mib >= 0.10*mi.bank_mib) {
                sel = b; p.reason = "q8_0 cache gives the expert arena a material share of the bank";
            } else if (a) {
                sel = a; p.reason = "the requested context and the reserve fit; experts take what is left";
            } else if (b) {
                sel = b; p.reason = "only a q8_0 cache leaves a useful expert arena";
            }
        }
        if (sel == nullptr && (force.empty() || force == "legacy")) {
            const tosh_plan_candidate * l16 = find("legacy_offload", kvs[0], 0);
            const tosh_plan_candidate * l8  = kvs.size() > 1 ? find("legacy_offload", kvs[1], 0) : nullptr;
            if (l16 && l16->valid) sel = l16; else if (l8 && l8->valid) sel = l8;
            if (sel) p.reason = "mixed execution is not available here; expert layers go to the host";
        }
        if (sel) {
            break;
        }
    }

    if (sel == nullptr) {
        p.state  = TOSH_PLAN_UNSUPPORTED;
        p.mode   = "none";
        p.reason = "no configuration fits the requested context with the minimum reserve";
    } else {
        p.mode = sel->mode; p.kv = sel->kv; p.ubatch = sel->ubatch; p.ncmoe = sel->ncmoe;
        p.arena_mib = sel->arena_mib; p.private_mib = sel->private_mib; p.free_mib = sel->free_mib;
        if (sel->mode == "full_gpu") {
            p.state = TOSH_PLAN_FULL_GPU_OPTIMAL;
            p.host_required_mib = host_engine;
            p.mlock = "not_required";
        } else if (sel->mode == "dmoe") {
            p.state = weights_fit ? TOSH_PLAN_DMOE_CONTEXT_OPTIMAL : TOSH_PLAN_DMOE_CAPACITY_REQUIRED;
            p.host_required_mib = host_dmoe;
        } else {
            p.state = TOSH_PLAN_LEGACY_OFFLOAD_BETTER;
            p.host_required_mib = mi.bank_mib*sel->ncmoe/std::max(1, mi.n_moe) + host_engine;
            p.mlock = "not_required";
            if (!dmoe_block.empty()) p.fallback = dmoe_block;
        }
    }
    }
    if (out) *out = p;
        return false;
    }

    // apply: the plan owns memory from here
    params.fit_params = false;
    params.n_gpu_layers = 999;
    params.cache_type_k = p.kv;
    params.cache_type_v = p.kv;
    params.n_ctx = p.n_ctx;
    params.n_ubatch = p.ubatch;
    params.n_batch  = std::max(params.n_batch, p.ubatch);
    if (p.mode != "full_gpu") {
        params.no_extra_bufts = true;
    }
        add_override(params, experts_pattern());
        params.load_mode = LLAMA_LOAD_MODE_MLOCK;
        setenv("TOSH_DMOE_CACHE_MIB", "auto", 1);
        setenv("TOSH_DMOE_RESERVE_MIB", string_format("%.0f", p.reserve_mib).c_str(), 1);
        setenv("TOSH_DMOE_MIN_ARENA_MIB", string_format("%.0f", p.min_arena_mib).c_str(), 1);
        setenv("TOSH_DMOE_RARE_ROWS", "auto", 0);
        setenv("TOSH_DMOE_RARE_TAIL", "0", 0);
        setenv("GGML_SCHED_PREFETCH_EXPERTS", "1", 0);
        setenv("GGML_CPU_NO_REPACK", "1", 0);
        // the host executor computes the missed experts on every core; the backend's own threads find little
        // CPU work here (verify windows and prompts included) and only compete with it for the cores
        const int cpu_threads = (int) env_mib("TOSH_DMOE_LLAMA_THREADS", 2);
        if (cpu_threads > 0) {
            for (auto * cp : { &params.cpuparams, &params.cpuparams_batch }) {
                if (cp->n_threads <= 0 || cp->n_threads > cpu_threads) {
                    cp->n_threads = cpu_threads;
                }
            }
        }
#ifdef TOSH_ENABLE_DYNAMIC_MOE
        tosh_dmoe_enable();
#endif
    } else if (p.mode == "legacy_offload") {
        for (int i = 0; i < p.ncmoe; ++i) {
            pattern_store().push_back(llm_ffn_block_regex(i, LLM_FFN_EXPS_REGEX));
            add_override(params, pattern_store().back().c_str());
        }
        setenv("GGML_SCHED_PREFETCH_EXPERTS", "1", 0);
        setenv("GGML_CPU_NO_REPACK", "1", 0);
    }
    return true;
}
