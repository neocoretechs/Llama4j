package com.llama4j;

interface FloatSliceView {
	int length();
	float get(int i);           // i in [0, length)
	// Optional: bulk read for JNI packing
	default void copyTo(float[] dst, int dstOffset) {
		for(int i = 0; i < length(); i++) 
			dst[dstOffset + i] = get(i);
	}
}
