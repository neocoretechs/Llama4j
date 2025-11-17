package com.llama4j;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

public final class DeviceMemoryLedger {
	public static boolean DEBUG = true;
	private static final Log log = LogFactory.getLog(DeviceMemoryLedger.class);
    private static long baselineFree;
    private static long total;
    private static long allocated;
    private static final int MIN_INTERVAL = 8;
    private static int releaseCount = 0;
    private static int refreshInterval = MIN_INTERVAL;
    private static long lastRefreshNanos = System.nanoTime();
    private static final int MAX_INTERVAL = 1024;
    private static final long MAX_REFRESH_GAP_NANOS = 5_000_000_000L; // 5s
    private static final int REFRESH_INTERVAL = 32; // every 32 frees
  
    static {
        refresh();
    }

    public static synchronized boolean tryReserve(long bytes) {
    	 // Safety margin: 10% of requested size, clamped between 4MB and 256MB
        long margin = Math.max(4 << 20, Math.min(256 << 20, bytes / 10));
        if(allocated + bytes + margin > baselineFree) {
        	if(DEBUG)
        		log.error("Failed to allocate device:"+bytes+" bytes, only "+baselineFree);
            return false;
        }
        allocated += bytes;
        if(DEBUG)
        	log.info("Allocated:"+allocated+" device bytes");
        return true;
    }

    public static synchronized void release(long bytes) {
        allocated -= bytes;
        if (allocated < 0) allocated = 0;
        releaseCount++;
        long now = System.nanoTime();
        boolean timeExpired = (now - lastRefreshNanos) > MAX_REFRESH_GAP_NANOS;
        boolean countExpired = releaseCount >= refreshInterval;
        if (timeExpired || countExpired) {
            refresh();
            // logarithmic wind‑up
            refreshInterval = Math.min(
                MAX_INTERVAL,
                MIN_INTERVAL * (1 + (int)(Math.log(releaseCount + 1) / Math.log(2)))
            );
            releaseCount = 0;
            lastRefreshNanos = now;
        }
        if(DEBUG)
        	log.info("DeviceMemoryLedger release:"+bytes);
    }

    private static void refresh() {
        MemorySegment mfree = Llama3.autoArena.allocate(8);
        MemorySegment mtotal = Llama3.autoArena.allocate(8);
		try {
			Llama3.cudaGetMemInfo.invokeExact(mfree, mtotal);
		} catch (Throwable e) {
			e.printStackTrace();
			return;
		}
        baselineFree = mfree.get(ValueLayout.JAVA_LONG, 0);
        total        = mtotal.get(ValueLayout.JAVA_LONG, 0);
        allocated    = 0;
        lastRefreshNanos = System.nanoTime();
        if(DEBUG)
        	log.info("Refreshing DeviceMemoryLedger free:"+baselineFree+" total:"+total);
    }

    public static synchronized void onAllocationFailure() {
        refreshInterval = MIN_INTERVAL;
        releaseCount = 0;
        lastRefreshNanos = System.nanoTime();
        if(DEBUG)
        	log.info("OnAllocationFailure DeviceMemoryLedger free:"+baselineFree+" total:"+total);
    }
}
