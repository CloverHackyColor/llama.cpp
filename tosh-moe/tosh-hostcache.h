// locks [addr, addr+bytes) a step at a time, waiting while the system reports memory pressure;
// false with a message in err (at least 256 bytes) when the lock fails or the pressure stays up
bool tosh_mlock_paced(void * addr, size_t bytes, char * err, size_t err_len);

// the same expert, when the caller copies its bytes out of VRAM instead: true when a slot waits
// for them at tosh_hostcache_part(); done publishes it, or falls back to the file read
bool tosh_hostcache_store(int il, int e);
void tosh_hostcache_store_done(int il, int e, bool ok);
