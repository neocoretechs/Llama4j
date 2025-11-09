package com.llama4j;

import java.lang.ref.Cleaner;

public final class DeviceMemoryReclaim implements AutoCloseable {
    private static final Cleaner cleaner = Cleaner.create();

    private final Cleaner.Cleanable cleanable;

    private static class State implements Runnable {
        private long ptr;
        private final long bytes;
        State(long ptr, long bytes) { 
        	this.ptr = ptr; 
        	this.bytes = bytes; 
        }
        @Override public void run() {
            if(ptr != 0) {
            	System.out.println("Freeing:"+ptr+" "+bytes+" bytes");
                try {
					Llama3.cublasFreeHandle.invokeExact(ptr);
				} catch (Throwable e) {
					e.printStackTrace();
				}
                DeviceMemoryLedger.release(bytes);
                ptr = 0;
            }
        }
    }
    
    public DeviceMemoryReclaim(long ptr, long bytes) {
        this.cleanable = cleaner.register(this, new State(ptr, bytes));
    }
    
    @Override
    public void close() {
        cleanable.clean(); // deterministic release
    }
}
