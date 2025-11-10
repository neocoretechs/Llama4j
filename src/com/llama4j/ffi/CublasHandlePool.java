package com.llama4j.ffi;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

import com.llama4j.FloatTensor;
import com.llama4j.Llama3;

public final class CublasHandlePool implements AutoCloseable {
    private BlockingQueue<Long> pool;

    public CublasHandlePool(int poolSize) {
    	//if(FloatTensor.USE_CUDA) {
    		pool = new ArrayBlockingQueue<>(poolSize);
    		for (int i = 0; i < poolSize; i++) {
    			long handle = 0;
    			try {
    				handle = (long) Llama3.cublasGetHandle.invokeExact();
    			} catch (Throwable e) {
    				e.printStackTrace();
    			} // your JNI/FFI wrapper
    			pool.add(handle);
    		}
    	//}
    }

    public long acquire()  {
    	long handle = 0L;
    	//if(FloatTensor.USE_CUDA) {
    		try {
    			handle = pool.take();
    		} catch (InterruptedException e) {
    			e.printStackTrace();
    			throw new RuntimeException(e);
    		} // blocks until a handle is available
    	//}
        return handle;
    }

    public void release(long handle) {
    	//if(FloatTensor.USE_CUDA)
    		pool.offer(handle);
    }

    @Override
    public void close() {
    	//if(FloatTensor.USE_CUDA) {
    		for (Long handle : pool) {
    			try {
    				Llama3.cublasFreeHandle.invokeExact(handle);
    			} catch (Throwable e) {
    				e.printStackTrace();
    			}
    		}
    		pool.clear();
    	//}
    }
}
