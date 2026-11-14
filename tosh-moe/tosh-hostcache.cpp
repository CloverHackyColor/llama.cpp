#ifdef __APPLE__
#include <sys/sysctl.h>
#endif
    uint64_t evictions = 0, reloads = 0, reads = 0, read_bytes = 0, prefetches = 0, stored = 0;
// Wired pages cannot be compressed or paged out: locking faster than the system frees memory
// evicts the code pages of every other process, and a desktop stalls until the watchdog reboots it.
bool mlock_paced(void * addr, size_t bytes, char * err, size_t err_len) {
    static const size_t step    = (size_t) 256 << 20;
    static const double wait_us = 30e6;
    char * base = (char *) addr;
    for (size_t o = 0; o < bytes; o += step) {
#ifdef __APPLE__
        const double t0 = now_us();
        for (;;) {
            int lvl = 1; size_t len = sizeof(lvl);
            if (sysctlbyname("kern.memorystatus_vm_pressure_level", &lvl, &len, nullptr, 0) != 0 || lvl <= 1) {
                break;
            }
            if (lvl >= 4 || now_us() - t0 > wait_us) {
                snprintf(err, err_len, "DMOE_HOST_MEMORY_PRESSURE: the system reported %s memory pressure after %.0f of %.0f MiB were locked",
                         lvl >= 4 ? "critical" : "sustained", o/1048576.0, bytes/1048576.0);
                munlock(base, o);
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
#endif
        if (getenv("TOSH_DMOE_TEST_MLOCK_FAIL") != nullptr || mlock(base + o, std::min(step, bytes - o)) != 0) {
            const int e = getenv("TOSH_DMOE_TEST_MLOCK_FAIL") != nullptr ? EAGAIN : errno;
            snprintf(err, err_len, "DMOE_HOST_BANK_NOT_LOCKABLE: mlock failed after %.0f of %.0f MiB (%s)",
                     o/1048576.0, bytes/1048576.0, strerror(e));
            munlock(base, o);
            return false;
        }
    }
    return true;
}

            char err[256];
            if (!mlock_paced(pl.base, pl.bytes, err, sizeof(err))) {
                std::lock_guard<std::mutex> lk(g.m);
                if (g.error.empty()) g.error = err;
                return;
// slots from the moment the model finishes loading, past ready, until the budget or the slots run
// out. Demand never waits on it: an entry is published only once read, and last in line for eviction.
            for (int key; (key = (*next)++) < n_keys;) {
bool tosh_mlock_paced(void * addr, size_t bytes, char * err, size_t err_len) {
    return mlock_paced(addr, bytes, err, err_len);
}

// a slot for an expert coming back from VRAM, or -1 when RAM still holds it or has no room; g.m held
static int readback_slot(int il, int e) {
    const int key = il*g.n_expert + e;
    if (g.base == nullptr) return -1;
    if (g.slot_of[key] >= 0) {
        // a copy kept on promotion: it is needed again
        const int s = g.slot_of[key];
        if (g.ref[s] == 0 && g.state[s] == RESIDENT && g.in_lru(s)) { g.lru_unlink(s); g.lru_push(s); g.refreshed++; }
        return -1;
    }
    const int s = g.take_slot(il);
    if (s < 0) return -1;
    g.key_of[s] = key;
    g.slot_of[key] = s;
    g.state[s] = LOADING;
    g.ref[s] = 0;
    g.ever[key] = 1;
    g.enq_t[s] = now_us();
    g.bg[s] = 1;
    g.bg_unused[s] = 1;
    g.origin[s] = g.ready ? 0 : 1;
    g.reused[s] = 0;
    return s;
}

        const int s = readback_slot(il, e);
bool tosh_hostcache_store(int il, int e) {
    std::lock_guard<std::mutex> lock(g.m);
    return readback_slot(il, e) >= 0;
}

void tosh_hostcache_store_done(int il, int e, bool ok) {
    {
        std::lock_guard<std::mutex> lock(g.m);
        const int s = g.base == nullptr ? -1 : g.slot_of[il*g.n_expert + e];
        if (s < 0 || g.state[s] != LOADING) return;
        if (ok) {
            g.state[s] = RESIDENT;
            if (g.ref[s] == 0 && !g.in_lru(s)) g.lru_push(s);
            g.stored++;
        } else {
            g.prefetches++;
            g.queue.push_back(s);
        }
    }
    if (ok) g.loaded.notify_all(); else g.work.notify_one();
}

    uint64_t hits = 0;
    for (int k = 0; k < TOSH_HC_KINDS; ++k) hits += g.hit[k];
    fprintf(stderr, "tosh_hostcache: read back on eviction %llu, copied back from VRAM %llu\n",
            (unsigned long long) g.prefetches, (unsigned long long) g.stored);
