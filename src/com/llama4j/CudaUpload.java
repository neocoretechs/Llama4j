package com.llama4j;

public class CudaUpload {
	public static native int uploadSlice(
		    long ctx,          // native AttnCtx handle
		    float[] hostBuf,   // scratch array
		    long devicePtr,    // which device buffer (dQ, dK, dV)
		    long offset,       // element offset into that buffer
		    int count          // number of floats to copy
		);
}
