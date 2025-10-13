package com.llama4j;

import java.util.ArrayDeque;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

final class BufferPool {
    private final Map<Integer, ArrayDeque<float[]>> pools = new ConcurrentHashMap<>();
    public float[] acquire(int length) {
        ArrayDeque<float[]> q = pools.computeIfAbsent(length, k -> new ArrayDeque<>());
        float[] buf = q.pollFirst();
        return (buf != null) ? buf : new float[length];
    }
    public void release(float[] buf) {
        pools.get(buf.length).offerFirst(buf);
    }

}
