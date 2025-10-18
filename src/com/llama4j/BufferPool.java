package com.llama4j;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedDeque;

final class BufferPool {
    private final Map<Integer, ConcurrentLinkedDeque<float[]>> pools = new ConcurrentHashMap<>();
    public float[] acquire(int length) {
        ConcurrentLinkedDeque<float[]> q = pools.computeIfAbsent(length, k -> new ConcurrentLinkedDeque<>());
        float[] buf = q.pollFirst();
        return (buf != null) ? buf : new float[length];
    	//return new float[length];
    }
    public void release(float[] buf) {
        pools.get(buf.length).offerFirst(buf);
    }

}
