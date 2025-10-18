package com.llama4j;
/**
 * How you use it
• 	Init once at model/session start: 
• 	Reuse  in all JNI calls (, , , etc.).
• 	Destroy when you’re done: 
 */
public final class Attn {
    static { System.loadLibrary("attn_jni"); }

    // returns a native context handle
    public static native long init(int maxB, int maxH, int maxTq, int maxTk, int d);

    // frees all device buffers and handle
    public static native void destroy(long ctx);

    // ... your uploadSlice, scores, output, etc. go here ...
}
