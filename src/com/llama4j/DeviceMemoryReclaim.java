package com.llama4j;

import java.lang.ref.Cleaner;
public final class DeviceMemoryReclaim implements AutoCloseable {
	public static boolean DEBUG = true;
    private static final Cleaner cleaner = Cleaner.create();
    private final Cleaner.Cleanable cleanable;

    private static final class State implements Runnable {
        private final long devicePtr;
        private final long sizeBytes;
        State(long devicePtr, long sizeBytes) {
            this.devicePtr = devicePtr;
            this.sizeBytes = sizeBytes;
        }
        @Override public void run() {
            if (devicePtr != 0L) {
                // Idempotent native free; must not touch the referent
                try { Llama3.freeDevicePtr.invokeExact(devicePtr); } catch (Throwable ignored) {}
                DeviceMemoryLedger.release(sizeBytes);
                DeviceManager.reclaim(devicePtr);
                if(DEBUG)
                	System.out.printf("%s freeing %d from %x%n",this.getClass().getName(),sizeBytes,devicePtr);
            }
        }
    }
    
    public DeviceMemoryReclaim(FloatTensor referent) {
        long ptr = referent.devicePtrOr0();
        long size = referent.totalBytes();
        this.cleanable = cleaner.register(referent, new State(ptr, size));
    }

    public DeviceMemoryReclaim(DeviceTensor referent) {
        long ptr = referent.devicePtrOr0();
        long size = referent.totalBytes();
        this.cleanable = cleaner.register(referent, new State(ptr, size));
    }

    @Override public void close() { cleanable.clean(); }
}
