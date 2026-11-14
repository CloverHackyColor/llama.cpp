#include "ggml-metal.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-metal-device.h"
#include "ggml-metal-fusion.h"
#include "ggml-metal-context.h"
#include "ggml-metal-ops.h"
#include "ggml-metal-tuning.h"

#include <mutex>
#include <string>

#define GGML_METAL_NAME "MTL"
#define GGML_METAL_MAX_DEVICES 16

// number of Metal devices
// note: can be overridden with GGML_METAL_DEVICES env to simulate virtual devices
static int g_devices = 1;

// forward declaration
static bool ggml_backend_buffer_is_metal(ggml_backend_buffer_t buffer);

////////////////////////////////////////////////////////////////////////////////
// backend interface
////////////////////////////////////////////////////////////////////////////////

// shared buffer

static void ggml_backend_metal_buffer_shared_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_free(ctx);
}

static void * ggml_backend_metal_buffer_shared_get_base(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    return ggml_metal_buffer_get_base(ctx);
}

static void ggml_backend_metal_buffer_shared_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_memset_tensor(ctx, tensor, value, offset, size);
}

static void ggml_backend_metal_buffer_shared_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_set_tensor(ctx, tensor, data, offset, size);
}

static void ggml_backend_metal_buffer_shared_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_get_tensor(ctx, tensor, data, offset, size);
}

static bool ggml_backend_metal_buffer_shared_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    if (!ggml_backend_buffer_is_metal(src->buffer)) {
        return false;
    }

    return ggml_metal_buffer_cpy_tensor(ctx, src, dst);
}

static void ggml_backend_metal_buffer_shared_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_clear(ctx, value);
}

static ggml_backend_buffer_i ggml_backend_metal_buffer_shared_i = {
    /* .free_buffer   = */ ggml_backend_metal_buffer_shared_free_buffer,
    /* .get_base      = */ ggml_backend_metal_buffer_shared_get_base,
    /* .init_tensor   = */ NULL,
    /* .memset_tensor = */ ggml_backend_metal_buffer_shared_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_buffer_shared_set_tensor,
    /* .get_tensor    = */ ggml_backend_metal_buffer_shared_get_tensor,
    /* .set_tensor_2d = */ NULL,
    /* .get_tensor_2d = */ NULL,
    /* .cpy_tensor    = */ ggml_backend_metal_buffer_shared_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_buffer_shared_clear,
    /* .reset         = */ NULL,
};

// private buffer

static void ggml_backend_metal_buffer_private_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_free(ctx);
}

static void * ggml_backend_metal_buffer_private_get_base(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    return ggml_metal_buffer_get_base(ctx);
}

static void ggml_backend_metal_buffer_private_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_memset_tensor(ctx, tensor, value, offset, size);
}

static void ggml_backend_metal_buffer_private_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_set_tensor(ctx, tensor, data, offset, size);
}

static void ggml_backend_metal_buffer_private_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_get_tensor(ctx, tensor, data, offset, size);
}

static void ggml_backend_metal_buffer_private_set_tensor_2d(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size,
        size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_set_tensor_2d(ctx, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

static bool ggml_backend_metal_buffer_private_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    if (!ggml_backend_buffer_is_metal(src->buffer)) {
        return false;
    }

    return ggml_metal_buffer_cpy_tensor(ctx, src, dst);
}

static void ggml_backend_metal_buffer_private_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_clear(ctx, value);
}

static ggml_backend_buffer_i ggml_backend_metal_buffer_private_i = {
    /* .free_buffer   = */ ggml_backend_metal_buffer_private_free_buffer,
    /* .get_base      = */ ggml_backend_metal_buffer_private_get_base,
    /* .init_tensor   = */ NULL,
    /* .memset_tensor = */ ggml_backend_metal_buffer_private_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_buffer_private_set_tensor,
    /* .get_tensor    = */ ggml_backend_metal_buffer_private_get_tensor,
    /* .set_tensor_2d = */ ggml_backend_metal_buffer_private_set_tensor_2d,
    /* .get_tensor_2d = */ NULL,
    /* .cpy_tensor    = */ ggml_backend_metal_buffer_private_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_buffer_private_clear,
    /* .reset         = */ NULL,
};

static bool ggml_backend_buffer_is_metal(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_metal_buffer_shared_free_buffer ||
           buffer->iface.free_buffer == ggml_backend_metal_buffer_private_free_buffer;
}

//
// buffer types
//

struct ggml_backend_metal_buffer_type {
    int device;
    std::string name;
};

struct ggml_backend_metal_buffer_type_deleter {
    void operator()(ggml_backend_metal_buffer_type * ctx) const {
        delete ctx;
    }
};

typedef std::unique_ptr<ggml_backend_metal_buffer_type, ggml_backend_metal_buffer_type_deleter> ggml_backend_metal_buffer_type_ptr;

// common method for allocating shread or private Metal buffers
static ggml_backend_buffer_t ggml_backend_metal_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size, bool shared) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;
    ggml_metal_buffer_t res = ggml_metal_buffer_init(ctx_dev, size, shared);

    // an allocation that fails returns null here, and reading is_shared off it turned an
    // out-of-memory into a segfault. Diagnosed by Slice on the RX 570:
    // https://www.insanelymac.com/forum/profile/112217-slice/
    if (res == NULL) {
        return NULL;
    }

    if (res == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate Metal buffer of %zu bytes (out of memory)\n", __func__, size);
        return NULL;
    }

    ggml_backend_buffer_i buf_i = ggml_metal_buffer_is_shared(res)
        ? ggml_backend_metal_buffer_shared_i
        : ggml_backend_metal_buffer_private_i;

    return ggml_backend_buffer_init(buft, buf_i, res, size);
}

static size_t ggml_backend_metal_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    size_t res = ggml_nbytes(tensor);

    // some operations require additional memory for fleeting data:
    switch (tensor->op) {
        case GGML_OP_MUL_MAT_ID:
            {
                res += ggml_metal_op_mul_mat_id_extra_tpe(tensor);
                res += ggml_metal_op_mul_mat_id_extra_ids(tensor);
                res += ggml_metal_op_mul_mat_id_extra_amax(tensor);
            } break;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                res += ggml_metal_op_flash_attn_ext_extra_pad(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_blk(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_tmp(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_kv_f16(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_idx(tensor);
            } break;
        case GGML_OP_CUMSUM:
        case GGML_OP_ARGSORT:
            {
                res *= 2;
            } break;
        case GGML_OP_TOP_K:
            {
                res = 2*sizeof(int32_t)*ggml_nelements(tensor->src[0]);
            } break;
        default:
            break;
    }

    return res;

    GGML_UNUSED(buft);
}

// default (shared) buffer type

static const char * ggml_backend_metal_buffer_type_shared_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    return ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_metal_buffer_type_shared_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    return ggml_backend_metal_buffer_type_alloc_buffer(buft, size, true);
}

static size_t ggml_backend_metal_buffer_type_shared_get_alignment(ggml_backend_buffer_type_t buft) {
    return 32;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_metal_buffer_type_shared_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    return ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

static size_t ggml_backend_metal_buffer_type_shared_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    return ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

static bool ggml_backend_metal_buffer_type_shared_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_shared(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::vector<ggml_backend_buffer_type> bufts;
    static std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    static bool initialized = false;
    if (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        for (int i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx =
                new ggml_backend_metal_buffer_type {
                    /* .device = */ i,
                    /* .name   = */ GGML_METAL_NAME + std::to_string(i),
                };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_buffer_type_shared_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_buffer_type_shared_alloc_buffer,
                    /* .alloc_buffer_n      = */ NULL,
                    /* .get_alignment       = */ ggml_backend_metal_buffer_type_shared_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_buffer_type_shared_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_metal_buffer_type_shared_get_alloc_size,
                    /* .get_alloc_size_n    = */ NULL,
                    /* .is_host             = */ ggml_backend_metal_buffer_type_shared_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = true;
    }

    return &bufts[device];
}

// default (private) buffer type

static const char * ggml_backend_metal_buffer_type_private_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    return ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_metal_buffer_type_private_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    return ggml_backend_metal_buffer_type_alloc_buffer(buft, size, false);
}

static size_t ggml_backend_metal_buffer_type_private_get_alignment(ggml_backend_buffer_type_t buft) {
    return 32;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_metal_buffer_type_private_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    return ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

static size_t ggml_backend_metal_buffer_type_private_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    return ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

static bool ggml_backend_metal_buffer_type_private_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_private(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::vector<ggml_backend_buffer_type> bufts;
    static std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    static bool initialized = false;
    if (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        for (int i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx = new ggml_backend_metal_buffer_type{
                /* .device = */ i,
                /* .name   = */ GGML_METAL_NAME + std::to_string(i) + "_Private"
            };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_buffer_type_private_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_buffer_type_private_alloc_buffer,
                    /* .alloc_buffer_n      = */ NULL,
                    /* .get_alignment       = */ ggml_backend_metal_buffer_type_private_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_buffer_type_private_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_metal_buffer_type_private_get_alloc_size,
                    /* .get_alloc_size_n    = */ NULL,
                    /* .is_host             = */ ggml_backend_metal_buffer_type_private_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = true;
    }

    return &bufts[device];
}

// mapped buffer type

static const char * ggml_backend_metal_buffer_type_mapped_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    return ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_metal_buffer_type_mapped_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // for mapped buffers, prefer shared memory
    return ggml_backend_metal_buffer_type_alloc_buffer(buft, size, true);
}

static size_t ggml_backend_metal_buffer_type_mapped_get_alignment(ggml_backend_buffer_type_t buft) {
    return 32;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_metal_buffer_type_mapped_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    return ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

static size_t ggml_backend_metal_buffer_type_mapped_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    return ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

static bool ggml_backend_metal_buffer_type_mapped_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_mapped(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::vector<ggml_backend_buffer_type> bufts;
    static std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    static bool initialized = false;
    if (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        for (int i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx = new ggml_backend_metal_buffer_type{
                /* .device = */ i,
                /* .name   = */ GGML_METAL_NAME + std::to_string(i) + "_Mapped"
            };
            ctxs.emplace_back(raw_ctx);

            // note: not obvious, but this buffer type still needs to implement .alloc_buffer:
            //       https://github.com/ggml-org/llama.cpp/pull/15832#discussion_r2333177099
            ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_buffer_type_mapped_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_buffer_type_mapped_alloc_buffer,
                    /* .alloc_buffer_n      = */ NULL,
                    /* .get_alignment       = */ ggml_backend_metal_buffer_type_mapped_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_buffer_type_mapped_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_metal_buffer_type_mapped_get_alloc_size,
                    /* .get_alloc_size_n    = */ NULL,
                    /* .is_host             = */ ggml_backend_metal_buffer_type_mapped_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = true;
    }

    return &bufts[device];
}

// backend

static const char * ggml_backend_metal_name(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    return ggml_metal_get_name(ctx);
}

static void ggml_backend_metal_free(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    // wait for any ongoing async operations to finish
    ggml_metal_synchronize(ctx);

    ggml_metal_free(ctx);

    free(backend);
}

static void ggml_backend_metal_synchronize(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_synchronize(ctx);
}

static void ggml_backend_metal_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_tensor_async(ctx, tensor, data, offset, size);
}

static void ggml_backend_metal_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_get_tensor_async(ctx, tensor, data, offset, size);
}

static bool ggml_backend_metal_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
#ifdef GGML_BACKEND_READ_BATCH
static void ggml_backend_metal_read_batch_begin(ggml_backend_t backend) {
    ggml_metal_read_batch_begin((ggml_metal_t) backend->context);
}

static void ggml_backend_metal_read_batch_end(ggml_backend_t backend) {
    ggml_metal_read_batch_end((ggml_metal_t) backend->context);
}
#endif
static bool ggml_backend_metal_cpy_tensor_async_ex(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst, bool prefer_events, bool hold_dst) {
    if (!ggml_backend_is_metal(backend_src) || !ggml_backend_is_metal(backend_dst)) {
        return false;
    }

    if (!ggml_backend_buffer_is_metal(src->buffer) || !ggml_backend_buffer_is_metal(dst->buffer)) {
        return false;
    }

    ggml_metal_t ctx_src = (ggml_metal_t)backend_src->context;
    ggml_metal_t ctx_dst = (ggml_metal_t)backend_dst->context;

    //ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    //ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    //ggml_metal_buffer_t buf_ctx_src = (ggml_metal_buffer_t)buf_src->context;
    //ggml_metal_buffer_t buf_ctx_dst = (ggml_metal_buffer_t)buf_dst->context;

    return ggml_metal_cpy_tensor_async_ex(ctx_src, ctx_dst, src, dst, prefer_events, hold_dst);
}

// Peer costs a third of the decode, so it only pays from a batch up. Both the layer
// hand-off below and the allreduce read the same threshold.
static int64_t ggml_metal_peer_min_batch(void) {
    static int64_t v = -1;
    if (v < 0) {
        const char * s = getenv("TOSH_MGPU_PEER_MIN_BATCH");
        v = s ? atoll(s) : 32;
    }
    return v;
}

static bool ggml_backend_metal_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    // A layer hand-off crosses one token at a time, which is the shape peer is worst at.
    const int64_t batch = src->ne[0] > 0 ? ggml_nelements(src)/src->ne[0] : 1;
    const bool prefer_events = batch < ggml_metal_peer_min_batch();
    return ggml_backend_metal_cpy_tensor_async_ex(backend_src, backend_dst, src, dst, prefer_events, /*hold_dst =*/ false);
}

static enum ggml_status ggml_backend_metal_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    return ggml_metal_graph_compute(ctx, cgraph);
}

static void ggml_backend_metal_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_event_record(ctx, ev);
}

static void ggml_backend_metal_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_event_wait(ctx, ev);
}

static void ggml_backend_metal_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(params && params->add_alloc_dep);

    // keep the MoE weighted-reduction inputs alive until the fused output so the
    // allocator cannot reuse them while the fused kernel is still reading them
    ggml_metal_fusion_add_alloc_deps(params->user_data, params->add_alloc_dep, cgraph);

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_graph_optimize(ctx, cgraph);
}

static void ggml_backend_metal_set_n_cb(ggml_backend_t backend, int n_cb) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_n_cb(ctx, n_cb);
}

static ggml_backend_i ggml_backend_metal_i = {
    /* .get_name                = */ ggml_backend_metal_name,
    /* .free                    = */ ggml_backend_metal_free,
    /* .set_tensor_async        = */ ggml_backend_metal_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_metal_get_tensor_async,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ ggml_backend_metal_cpy_tensor_async, // only needed for multi-GPU setups
    /* .synchronize             = */ ggml_backend_metal_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_metal_graph_compute,
    /* .event_record            = */ ggml_backend_metal_event_record,
    /* .event_wait              = */ ggml_backend_metal_event_wait,
    /* .graph_optimize          = */ ggml_backend_metal_graph_optimize,
#ifdef GGML_BACKEND_READ_BATCH
    /* .read_batch_begin        = */ ggml_backend_metal_read_batch_begin,
    /* .read_batch_end          = */ ggml_backend_metal_read_batch_end,
#endif
};

static ggml_guid_t ggml_backend_metal_guid(void) {
    static ggml_guid guid = { 0x81, 0xa1, 0x8b, 0x1e, 0x71, 0xec, 0x79, 0xed, 0x2b, 0x85, 0xdc, 0x8a, 0x61, 0x98, 0x30, 0xe6 };
    return &guid;
}

ggml_backend_t ggml_backend_metal_init(void) {
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_metal_reg(), 0);
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev, false);
    if (ctx == NULL) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return NULL;
    }

    ggml_backend_t backend = (ggml_backend_t) malloc(sizeof(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_set_n_cb(backend, 1);

    return backend;
}

bool ggml_backend_is_metal(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_metal_guid());
}

void ggml_backend_metal_set_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback, void * user_data) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_abort_callback(ctx, abort_callback, user_data);
}

bool ggml_backend_metal_supports_family(ggml_backend_t backend, int family) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    return ggml_metal_supports_family(ctx, family);
}

void ggml_backend_metal_capture_next_compute(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_capture_next_compute(ctx);
}

// backend device

static const char * ggml_backend_metal_device_get_name(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    return props_dev->name;
}

static const char * ggml_backend_metal_device_get_description(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    return ggml_metal_device_get_props(ctx_dev)->desc;
}

static void ggml_backend_metal_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_device_get_memory(ctx_dev, free, total);
}

static enum ggml_backend_dev_type ggml_backend_metal_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_metal_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_metal_device_get_name(dev);
    props->description = ggml_backend_metal_device_get_description(dev);
    props->type        = ggml_backend_metal_device_get_type(dev);

    ggml_backend_metal_device_get_memory(dev, &props->memory_free, &props->memory_total);

    // zero-copy only loses when the fallback would be private VRAM: on a discrete GPU a mapped weight buffer streams weights over PCIe on every access (~25x slower decode).
    // eGPUs default to shared buffers (#17866), so their fallback is shared memory too and they keep zero-copy.
    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props((ggml_metal_device_t) dev->context);

    props->caps = {
        /* .async                = */ true,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ props_dev->has_unified_memory || props_dev->use_shared_buffers,
        /* .events               = */ true,
        /* .mmap_support         = */ true,
    };
}

static ggml_backend_t ggml_backend_metal_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev, params != NULL && strcmp(params, "prefetch") == 0);
    if (ctx == NULL) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return NULL;
    }

    ggml_backend_t backend = (ggml_backend_t) malloc(sizeof(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_set_n_cb(backend, 1);

    return backend;

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_metal_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    return props_dev->use_shared_buffers ? ggml_backend_metal_buffer_type_shared(props_dev->device) : ggml_backend_metal_buffer_type_private(props_dev->device);
}

static ggml_backend_buffer_t ggml_backend_metal_device_buffer_mapped(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_buffer_t res = ggml_metal_buffer_map(ctx_dev, ptr, size, max_tensor_size);

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    return ggml_backend_buffer_init(ggml_backend_metal_buffer_type_mapped(props_dev->device), ggml_backend_metal_buffer_shared_i, res, size);
}

static bool ggml_backend_metal_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    return ggml_metal_device_supports_op(ctx_dev, op);
}

static bool ggml_backend_metal_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return
        buft->device == dev && (
        buft->iface.get_name == ggml_backend_metal_buffer_type_shared_get_name ||
        buft->iface.get_name == ggml_backend_metal_buffer_type_private_get_name ||
        buft->iface.get_name == ggml_backend_metal_buffer_type_mapped_get_name);

    GGML_UNUSED(dev);
}

static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_metal_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    return (op->op == GGML_OP_MUL_MAT ||
            op->op == GGML_OP_MUL_MAT_ID) &&
            get_op_batch_size(op) >= ggml_metal_device_get_props(ctx_dev)->op_offload_min_batch_size;
}

static ggml_backend_event_t ggml_backend_metal_device_event_new(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t event = ggml_metal_device_event_init(ctx_dev);
    GGML_ASSERT(event);

    ggml_backend_event_t ev = new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };

    return ev;
}

static void ggml_backend_metal_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_device_event_free(ctx_dev, ev);

    delete event;
}

static void ggml_backend_metal_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t evt = (ggml_metal_event_t)event->context;

    ggml_metal_device_event_synchronize(ctx_dev, evt);
}

static ggml_backend_device_i ggml_backend_metal_device_i = {
    /* .get_name             = */ ggml_backend_metal_device_get_name,
    /* .get_description      = */ ggml_backend_metal_device_get_description,
    /* .get_memory           = */ ggml_backend_metal_device_get_memory,
    /* .get_type             = */ ggml_backend_metal_device_get_type,
    /* .get_props            = */ ggml_backend_metal_device_get_props,
    /* .init_backend         = */ ggml_backend_metal_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_metal_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_metal_device_buffer_mapped,
    /* .supports_op          = */ ggml_backend_metal_device_supports_op,
    /* .supports_buft        = */ ggml_backend_metal_device_supports_buft,
    /* .offload_op           = */ ggml_backend_metal_device_offload_op,
    /* .event_new            = */ ggml_backend_metal_device_event_new,
    /* .event_free           = */ ggml_backend_metal_device_event_free,
    /* .event_synchronize    = */ ggml_backend_metal_device_event_synchronize,
};

// backend registry

struct ggml_backend_metal_reg {
    std::vector<ggml_backend_dev_t> devices;
};

typedef struct ggml_backend_metal_reg * ggml_backend_metal_reg_t;

static ggml_backend_metal_reg_t ggml_backend_metal_reg_init(void) {
    ggml_backend_metal_reg_t ctx = new struct ggml_backend_metal_reg;

    return ctx;
}

static void ggml_backend_metal_reg_free(ggml_backend_metal_reg_t ctx) {
    delete ctx;
}

struct ggml_backend_metal_reg_deleter {
    void operator()(ggml_backend_metal_reg_t ctx) {
        ggml_backend_metal_reg_free(ctx);
    }
};

typedef std::unique_ptr<struct ggml_backend_metal_reg, ggml_backend_metal_reg_deleter> ggml_backend_metal_reg_ptr;

static const char * ggml_backend_metal_reg_get_name(ggml_backend_reg_t reg) {
    return GGML_METAL_NAME;

    GGML_UNUSED(reg);
}

static size_t ggml_backend_metal_reg_device_count(ggml_backend_reg_t reg) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_metal_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static ggml_backend_feature g_ggml_backend_metal_features[] = {
#if defined(GGML_METAL_EMBED_LIBRARY)
    { "EMBED_LIBRARY", "1" },
#endif
    { NULL, NULL },
};

static ggml_backend_feature * ggml_backend_metal_get_features(ggml_backend_reg_t reg) {
    return g_ggml_backend_metal_features;

    GGML_UNUSED(reg);
}

// test/tune-only override for the FA vec (Q, NE) selection, reached via proc_address.
static void ggml_backend_metal_tuning_set_fa_vec_override(int Q, int NE) {
    ggml_metal_tuning::fa_vec_set_override({ (int8_t) Q, (int8_t) NE });
}

static void ggml_backend_metal_tuning_clear_fa_vec_override(void) {
    ggml_metal_tuning::fa_vec_clear_override();
}

static int ggml_backend_metal_tuning_fa_vec_ne11_bucket(int64_t ne11) {
    return ggml_metal_tuning::fa_vec_ne11_bucket(ne11);
}

static int ggml_backend_metal_tuning_fa_vec_ne01_bucket(int64_t ne01) {
    return ggml_metal_tuning::fa_vec_ne01_bucket(ne01);
}

static int ggml_backend_metal_tuning_fa_vec_baseline_ne(int dk, int dv) {
    return ggml_metal_tuning::fa_vec_baseline_ne(dk, dv);
}

static const char * ggml_backend_metal_tuning_device_token(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    return ggml_metal_device_id_token(ggml_metal_device_get_props(ctx_dev)->device_id);
}

// generic fusion debugging API (ad-hoc proc-address mechanism): the test resolves the device
// fusion context once and passes that opaque handle to the rest of the functions
typedef void * ggml_backend_fusion_t;

static ggml_backend_fusion_t ggml_backend_metal_fusion_get(ggml_backend_dev_t dev) {
    return ggml_metal_device_get_fusion_info((ggml_metal_device_t)dev->context);
}

static void ggml_backend_metal_fusion_stats_init(ggml_backend_fusion_t finfo) {
    ggml_metal_fusion_info_stats_init((struct ggml_metal_fusion_info *) finfo);
}

static void ggml_backend_metal_fusion_stats_reset(ggml_backend_fusion_t finfo) {
    ggml_metal_fusion_info_stats_reset((struct ggml_metal_fusion_info *) finfo);
}

static int ggml_backend_metal_fusion_stats_get(ggml_backend_fusion_t finfo, const char ** labels, uint64_t * counts, int n) {
    return ggml_metal_fusion_info_stats_get((struct ggml_metal_fusion_info *) finfo, labels, counts, n);
}

static void ggml_backend_metal_fusion_set_enabled(ggml_backend_fusion_t finfo, bool enabled) {
    ggml_metal_fusion_info_set_enabled((struct ggml_metal_fusion_info *) finfo, enabled);
}

// Tensor split reduces after every subgraph. The meta backend's generic path spends a full
// graph submission per device on each of those adds; here the copy and the add are issued
// directly, which is what the CUDA and SYCL backends already do for themselves.
struct ggml_backend_metal_comm_context {
    std::vector<ggml_backend_t>         backends;
    // scratch per device and reduction step: reusing one per device would let a step
    // overwrite what the previous one is still reducing
    std::vector<ggml_backend_buffer_t>  tmp;
    std::vector<size_t>                 tmp_size;
    size_t                              n_steps = 0;
// Butterfly reduction over the largest power of two that fits, same schedule as the generic
// path: fold the devices past it, exchange, then copy the result back to them.
static size_t ggml_backend_metal_comm_offset_max(size_t n_backends) {
    size_t offset = n_backends/2;
    while ((offset & (offset - 1)) != 0) {
        offset--;
    }
    return offset;
}

static size_t ggml_backend_metal_comm_n_steps(size_t n_backends) {
    const size_t offset_max = ggml_backend_metal_comm_offset_max(n_backends);

    size_t n_steps = n_backends > 2*offset_max ? 1 : 0;
    for (size_t offset = offset_max; offset >= 1; offset /= 2) {
        n_steps++;
    }

    return n_steps;
}

static void * ggml_backend_metal_comm_init(ggml_backend_t * backends, size_t n_backends) {
    if (n_backends < 2 || getenv("TOSH_MGPU_COMM_DISABLE") != NULL) {
        return nullptr;
    }

    for (size_t i = 0; i < n_backends; i++) {
        if (!ggml_backend_is_metal(backends[i])) {
            return nullptr;
        }
    }

    auto * comm_ctx = new ggml_backend_metal_comm_context;
    comm_ctx->backends.assign(backends, backends + n_backends);
    comm_ctx->n_steps = ggml_backend_metal_comm_n_steps(n_backends);
    comm_ctx->tmp.assign(n_backends*comm_ctx->n_steps, nullptr);
    comm_ctx->tmp_size.assign(n_backends*comm_ctx->n_steps, 0);

    if (n_backends > 2) {
        GGML_LOG_WARN("%s: reducing across %zu devices in %zu steps\n", __func__, n_backends, comm_ctx->n_steps);
    }

    return comm_ctx;
}

static void ggml_backend_metal_comm_free(void * comm_ctx_v) {
    auto * comm_ctx = (ggml_backend_metal_comm_context *) comm_ctx_v;
    for (auto & buf : comm_ctx->tmp) {
        if (buf != nullptr) {
            ggml_backend_buffer_free(buf);
        }
    delete comm_ctx;
}

// TOSH_MGPU_ALLREDUCE_NOOP=1 keeps the split graph and every subgraph submission but performs
// no reduction and no cross-device synchronisation at all. The output is wrong by construction.
// It exists for one measurement: what tensor-split decode would cost if communication were free.
static bool ggml_backend_metal_comm_allreduce_noop(void) {
    static int v = -1;
    if (v < 0) {
        const char * s = getenv("TOSH_MGPU_ALLREDUCE_NOOP");
        v = (s && s[0] == '1') ? 1 : 0;
        if (v) {
            fprintf(stderr, "ggml_metal: WARNING TOSH_MGPU_ALLREDUCE_NOOP=1, the reduction is "
                            "skipped and the output is meaningless. Benchmark only.\n");
        }
    }
    return v == 1;
}

static bool ggml_backend_metal_comm_allreduce_tensor(void * comm_ctx_v, ggml_tensor ** tensors) {
    if (ggml_backend_metal_comm_allreduce_noop()) {
        return true;
    }

    auto * comm_ctx = (ggml_backend_metal_comm_context *) comm_ctx_v;

    const size_t n_backends = comm_ctx->backends.size();

    if (tensors[0] == nullptr || tensors[0]->type != GGML_TYPE_F32) {

    const int64_t ne = ggml_nelements(tensors[0]);
    if (ne == 0) {
        return true;
    }
    for (size_t i = 0; i < n_backends; i++) {
        ggml_tensor * t = tensors[i];
        if (t == nullptr || t->type != GGML_TYPE_F32 || ggml_nelements(t) != ne) {
        }
        if (!ggml_are_same_shape(t, tensors[0]) || !ggml_is_contiguous(t)) {
        }
        if (!ggml_backend_buffer_is_metal(t->buffer)) {
        }
        if (!(t->flags & GGML_TENSOR_FLAG_COMPUTE)) {

    const size_t size = ggml_nbytes(tensors[0]);
    const size_t n_steps = comm_ctx->n_steps;

    for (size_t k = 0; k < n_backends*n_steps; k++) {
        if (comm_ctx->tmp_size[k] < size) {
            ggml_backend_buffer_t buf = ggml_backend_alloc_buffer(comm_ctx->backends[k/n_steps], size);
            if (buf == nullptr) {
                return false;
            }
            if (comm_ctx->tmp[k] != nullptr) {
                ggml_backend_buffer_free(comm_ctx->tmp[k]);
            }
            comm_ctx->tmp[k] = buf;
            comm_ctx->tmp_size[k] = size;
        }
    }

    // stack views onto the scratch buffers, so the existing cross-device copy is reused as is
    std::vector<ggml_tensor> tmp(n_backends);
    auto view_tmp = [&](size_t i, size_t step) {
        tmp[i] = *tensors[i];
        tmp[i].buffer    = comm_ctx->tmp[i*n_steps + step];
        tmp[i].data      = ggml_backend_buffer_get_base(tmp[i].buffer);
        tmp[i].view_src  = nullptr;
        tmp[i].view_offs = 0;
        tmp[i].src[0]    = nullptr;
        tmp[i].src[1]    = nullptr;
    };

    for (size_t i = 0; i < n_backends; i++) {
        view_tmp(i, 0);
        ggml_metal_t ctx = (ggml_metal_t) comm_ctx->backends[i]->context;
        if (!ggml_metal_add_inplace_supported(ctx, tensors[i], &tmp[i])) {
    }

    // The peer copy has the higher bandwidth and the event hand-off the lower fixed cost, so
    // the batch decides: peer wins the prefill, events the decode.
    const bool prefer_events = ne / tensors[0]->ne[0] < ggml_metal_peer_min_batch();

    // the add goes into the command buffer the copy already opened; kill switch because that
    // buffer stays open between the two calls
    static int fuse = -1;
    if (fuse < 0) {
        fuse = getenv("TOSH_MGPU_FUSE_ADD_DISABLE") == NULL ? 1 : 0;
    }
    const bool hold_dst = fuse == 1;

    // Bailing out is only safe while no device has reduced anything: past that the generic
    // path would add the partial sums a second time.
    bool reduced = false;
        view_tmp(j_dst, step);
        return ok;
    };
    auto reduce = [&](size_t j_dst) {
        ggml_metal_t ctx = (ggml_metal_t) comm_ctx->backends[j_dst]->context;
        reduced = true;
    const size_t offset_max = ggml_backend_metal_comm_offset_max(n_backends);

    size_t step = 0;
    if (n_backends > 2*offset_max) {
        for (size_t j_src = 2*offset_max; j_src < n_backends; j_src++) {
        }
        for (size_t j_src = 2*offset_max; j_src < n_backends; j_src++) {
        step++;
    }

    // Every exchange of a step must read the partner's tensor before its own add rewrites it,
    // so the copies all go out before any of the adds.
    for (size_t offset = offset_max; offset >= 1; offset /= 2) {
            // TOSH_MGPU_SUBMIT_ORDER, benchmark only: 0 normal, 1 reversed, 2 alternating.
            static int order = -1;
            if (order < 0) {
                const char * v = getenv("TOSH_MGPU_SUBMIT_ORDER");
                order = v ? atoi(v) : 0;
            }
            const bool flip = order == 1 || (order == 2 && (seq_round & 1));
            for (size_t jj = 0; jj < 2*offset_max && ok; jj++) {
                const size_t j = flip ? (2*offset_max - 1 - jj) : jj;
        for (size_t j = 0; j < 2*offset_max; j++) {
        }
        for (size_t j = 0; j < 2*offset_max; j++) {
        step++;
    }

    for (size_t j = 2*offset_max; j < n_backends; j++) {
        const size_t j_src = j - 2*offset_max;
}

static void * ggml_backend_metal_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *)ggml_backend_metal_comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *)ggml_backend_metal_comm_free;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *)ggml_backend_metal_comm_allreduce_tensor;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_metal_get_features;
    }
    if (strcmp(name, "ggml_backend_metal_tuning_set_fa_vec_override") == 0) {
        return (void *)ggml_backend_metal_tuning_set_fa_vec_override;
    }
    if (strcmp(name, "ggml_backend_metal_tuning_clear_fa_vec_override") == 0) {
        return (void *)ggml_backend_metal_tuning_clear_fa_vec_override;
    }
    if (strcmp(name, "ggml_backend_metal_tuning_fa_vec_ne11_bucket") == 0) {
        return (void *)ggml_backend_metal_tuning_fa_vec_ne11_bucket;
    }
    if (strcmp(name, "ggml_backend_metal_tuning_fa_vec_ne01_bucket") == 0) {
        return (void *)ggml_backend_metal_tuning_fa_vec_ne01_bucket;
    }
    if (strcmp(name, "ggml_backend_metal_tuning_fa_vec_baseline_ne") == 0) {
        return (void *)ggml_backend_metal_tuning_fa_vec_baseline_ne;
    }
    if (strcmp(name, "ggml_backend_metal_tuning_device_token") == 0) {
        return (void *)ggml_backend_metal_tuning_device_token;
    }
    // generic fusion debugging API (ad-hoc proc-address mechanism, not part of the official
    // ggml backend interface yet; a backend that adopts it exports these exact names)
    if (strcmp(name, "ggml_backend_fusion_get") == 0) {
        return (void *)ggml_backend_metal_fusion_get;
    }
    if (strcmp(name, "ggml_backend_fusion_stats_init") == 0) {
        return (void *)ggml_backend_metal_fusion_stats_init;
    }
    if (strcmp(name, "ggml_backend_fusion_stats_reset") == 0) {
        return (void *)ggml_backend_metal_fusion_stats_reset;
    }
    if (strcmp(name, "ggml_backend_fusion_stats_get") == 0) {
        return (void *)ggml_backend_metal_fusion_stats_get;
    }
    if (strcmp(name, "ggml_backend_fusion_set_enabled") == 0) {
        return (void *)ggml_backend_metal_fusion_set_enabled;
    }

    return NULL;

    GGML_UNUSED(reg);
}

static ggml_backend_reg_i ggml_backend_metal_reg_i = {
    /* .get_name         = */ ggml_backend_metal_reg_get_name,
    /* .get_device_count = */ ggml_backend_metal_reg_device_count,
    /* .get_device       = */ ggml_backend_metal_reg_device_get,
    /* .get_proc_address = */ ggml_backend_metal_get_proc_address,
};

static ggml_backend_dev_t ggml_backend_metal_device_init(ggml_backend_reg_t reg, int device) {
    return new ggml_backend_device {
        /* .iface   = */ ggml_backend_metal_device_i,
        /* .reg     = */ reg,
        /* .context = */ ggml_metal_device_get(device, g_devices),
    };
}

static void ggml_backend_metal_device_free(ggml_backend_dev_t dev) {
    delete dev;
}

struct ggml_backend_device_deleter {
    void operator()(ggml_backend_dev_t ctx) {
        ggml_backend_metal_device_free(ctx);
    }
};

typedef std::unique_ptr<ggml_backend_device, ggml_backend_device_deleter> ggml_backend_device_ptr;

ggml_backend_reg_t ggml_backend_metal_reg(void) {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);

        const char * env = getenv("GGML_METAL_DEVICES");
        if (env) {
            g_devices = atoi(env);
        }

        // an explicit device list registers one slot per listed physical GPU
        const char * env_list = getenv("GGML_METAL_DEVICE_LIST");
        if (env_list && *env_list) {
            g_devices = 1;
            for (const char * p = env_list; *p; ++p) {
                if (*p == ',') {
                    ++g_devices;
                }
            }
        }

        static std::vector<ggml_backend_device_ptr> devs;

        if (!initialized) {
            // workaround macOS limitation (kIOGPUCommandBufferCallbackErrorImpactingInteractivity) until proper fix becomes possible
            // ref: https://github.com/ggml-org/llama.cpp/issues/20141#issuecomment-4272947703
            setenv("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1", true);

            static ggml_backend_metal_reg_ptr reg_ctx(ggml_backend_metal_reg_init());

            for (int i = 0; i < g_devices; ++i) {
                // i is the logical slot (0..g_devices-1) used everywhere as the
                // device index (buffer types, naming). Physical GPU selection lives
                // in ggml_metal_device_init, which maps the slot to a real Metal
                // device via GGML_METAL_DEVICE_INDEX (see there).
                auto * dev = ggml_backend_metal_device_init(&reg, i);
                devs.emplace_back(dev);

                reg_ctx->devices.push_back(dev);
            }

            reg = {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_metal_reg_i,
                /* .context     = */ reg_ctx.get(),
            };
        }

        initialized = true;
    }

    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_metal_reg)
