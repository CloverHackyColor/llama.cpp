#pragma once

#include "ggml-metal-device.h"

#ifdef __cplusplus
extern "C" {
#endif

//
// backend context
//

typedef struct ggml_metal * ggml_metal_t;

// prefetch contexts get their own command queue and the no-copy upload path
// (see ggml_metal_set_tensor_async); everything else shares the device queue
ggml_metal_t ggml_metal_init(ggml_metal_device_t dev, bool prefetch);
void ggml_metal_free(ggml_metal_t ctx);

const char * ggml_metal_get_name(ggml_metal_t ctx);

void ggml_metal_synchronize(ggml_metal_t ctx);

void ggml_metal_set_tensor_async(ggml_metal_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void ggml_metal_get_tensor_async(ggml_metal_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
// batch small synchronous reads into one device round-trip
void ggml_metal_read_batch_begin(ggml_metal_t ctx);
void ggml_metal_read_batch_end  (ggml_metal_t ctx);
// prefer_events picks the event hand-off over the peer copy when both are enabled; the caller
// decides, since peer wins the prefill and events the decode.
// hold_dst leaves the destination command buffer open for the add that follows the copy, which
// is only correct when the caller does issue that add right away. The source side is never held:
// it has to start running while the rest is still being encoded.
bool ggml_metal_cpy_tensor_async_ex(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, const struct ggml_tensor * src, struct ggml_tensor * dst, bool prefer_events, bool hold_dst);
bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);

// acc += add, both f32 and contiguous. Encodes the kernel directly so a tensor-split
// allreduce does not pay a full graph submission per step.
bool ggml_metal_add_inplace_supported(ggml_metal_t ctx, const struct ggml_tensor * acc, const struct ggml_tensor * add);
bool ggml_metal_add_inplace_async    (ggml_metal_t ctx, struct ggml_tensor * acc, const struct ggml_tensor * add);

enum ggml_status ggml_metal_graph_compute (ggml_metal_t ctx, struct ggml_cgraph * gf);
void             ggml_metal_graph_optimize(ggml_metal_t ctx, struct ggml_cgraph * gf);

void ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev);
void ggml_metal_event_wait  (ggml_metal_t ctx, ggml_metal_event_t ev);

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx);

void ggml_metal_set_n_cb            (ggml_metal_t ctx, int n_cb);
void ggml_metal_set_abort_callback  (ggml_metal_t ctx, ggml_abort_callback abort_callback, void * user_data);

bool ggml_metal_supports_family     (ggml_metal_t ctx, int family);
void ggml_metal_capture_next_compute(ggml_metal_t ctx);

#ifdef __cplusplus
}
#endif
