#pragma once

#include "ggml.h"

#include <string>
#include <vector>

struct common_params;

// Context-first memory plan for MoE models: the requested context and a VRAM reserve come first,
// expert residency takes what is left. Enabled by TOSH_AUTO (auto, or full / dmoe / legacy to
// force an execution mode); TOSH_AUTO_KV picks the cache type (auto, f16, q8_0, turbo4).

enum tosh_plan_state {
    TOSH_PLAN_NOT_PLANNED,          // dense model, or planning off
    TOSH_PLAN_FULL_GPU_OPTIMAL,     // weights and the requested workload fit with the reserve
    TOSH_PLAN_DMOE_CONTEXT_OPTIMAL, // the weights fit, the requested context does not
    TOSH_PLAN_DMOE_CAPACITY_REQUIRED,
    TOSH_PLAN_LEGACY_OFFLOAD_BETTER,
    TOSH_PLAN_UNSUPPORTED,
};

struct tosh_plan_candidate {
    std::string name;
    std::string mode;               // full_gpu, dmoe, legacy_offload
    ggml_type   kv      = GGML_TYPE_F16;
    int         ubatch  = 512;
    int         ncmoe   = 0;
    bool        valid   = false;
    std::string reason;
    double      private_mib = 0;    // projected VRAM this process holds
    double      free_mib    = 0;    // projected VRAM left on the card
    double      arena_mib   = 0;
    double      kv_mib      = 0;
    double      compute_mib = 0;
};

struct tosh_plan {
    tosh_plan_state state = TOSH_PLAN_NOT_PLANNED;
    std::string mode;               // selected execution mode
    std::string reason;
    std::string fallback;           // why a preferred mode was not used
    ggml_type   kv     = GGML_TYPE_F16;
    int         ubatch = 512;
    int         n_ctx  = 0;
    int         ncmoe  = 0;
    double      reserve_mib        = 0;
    double      arena_mib          = 0;
    double      min_arena_mib      = 0;
    double      private_mib        = 0;
    double      free_mib           = 0;
    double      vram_total_mib     = 0;
    double      vram_free_mib      = 0;
    double      host_required_mib  = 0;
    double      host_ram_mib       = 0;
    double      host_available_mib = 0;
    double      host_reserve_mib   = 0;
    double      bank_mib           = 0;
    std::string mlock;              // required, not_required, not_lockable
    std::string dispersion;         // normal, high
    std::vector<tosh_plan_candidate> candidates;
};

const char * tosh_plan_state_name(tosh_plan_state s);
std::string  tosh_plan_json(const tosh_plan & plan);

bool tosh_plan_apply(common_params & params, tosh_plan * out = nullptr);
