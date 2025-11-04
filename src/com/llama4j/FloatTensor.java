package com.llama4j;

import java.io.Externalizable;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.DoubleAdder;

import com.neocoretechs.cublas.DeviceBuffer;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;

/**
 * Over-simplified, shapeless, float tensor.
 * <p>
 * Not a strict tensor, but rather just a sequence of floats, not required to be backed by memory
 * e.g. can represent a sequence of quantized floats.
 */
public abstract class FloatTensor implements Externalizable, Comparable {
	public static boolean DEBUG = true;
	static final boolean USE_CUDA = true;
    static final int VECTOR_BIT_SIZE = Integer.getInteger("llama.VectorBitSize", VectorShape.preferredShape().vectorBitSize());
    static final boolean USE_VECTOR_API = VECTOR_BIT_SIZE != 0;

    static short readShort(MemorySegment memorySegment, long offset) {
        return memorySegment.get(ValueLayout.JAVA_SHORT, offset);
        //return UNSAFE.getShort(memorySegment.address() + offset);
    }  
    static int readInt(MemorySegment memorySegment, long offset) {
        return memorySegment.get(ValueLayout.JAVA_INT, offset);
        //return UNSAFE.getShort(memorySegment.address() + offset);
    }
    static float readFloat(MemorySegment memorySegment, long offset) {
        return memorySegment.get(ValueLayout.JAVA_FLOAT, offset);
        //return UNSAFE.getShort(memorySegment.address() + offset);
    }  
    static byte readByte(MemorySegment memorySegment, long offset) {
        return memorySegment.get(ValueLayout.JAVA_BYTE, offset);
        //return UNSAFE.getByte(memorySegment.address() + offset);
    }
    // Preferred vector size for the fast multiplication routines.
    // (Apple Silicon) NEON only supports up-to 128bit vectors.
    static final VectorSpecies<Float> F_SPECIES;
    static final VectorSpecies<Integer> I_SPECIES;
    static final VectorSpecies<Short> S_SPECIES_HALF;

    static {
        if (USE_VECTOR_API) {
            F_SPECIES = VectorShape.forBitSize(VECTOR_BIT_SIZE).withLanes(float.class);
            I_SPECIES = F_SPECIES.withLanes(int.class);
            S_SPECIES_HALF = VectorShape.forBitSize(F_SPECIES.vectorBitSize() / 2).withLanes(short.class);
            assert F_SPECIES.length() == S_SPECIES_HALF.length();
        } else {
            F_SPECIES = null;
            I_SPECIES = null;
            S_SPECIES_HALF = null;
        }
    }
 
    public abstract int size();
    public abstract float getFloat(int index);
    public abstract void setFloat(int index, float value);
    abstract FloatVector getFloatVector(VectorSpecies<Float> species, int offset);
    public abstract DeviceBuffer getDevice();
    public abstract MemorySegment asSlice(long offSet1, long offSet2);
    public abstract MemorySegment getSegment();
    public abstract long getOffsetBytes(long offset);
    public abstract long getLengthBytes(long size, long offset);
    abstract GGMLType type();

    public static int numberOfElements(int... dimensions) {
        assert Arrays.stream(dimensions).allMatch(i -> i > 0);
        return Arrays.stream(dimensions).reduce(Math::multiplyExact).orElseThrow();
    }
    public static float scalarDot(FloatTensor thiz, int thisOffset, FloatTensor that, int thatOffset, int size) {
        float result = 0f;
        for (int j = 0; j < size; j++) {
            result += thiz.getFloat(thisOffset + j) * that.getFloat(thatOffset + j);
        }
        return result;
    }
    public float dot(long cublasHandle, int thisOffset, FloatTensor that, int thatOffset, int size) {
      	if(USE_CUDA) {
    		//boolean success = getDevice().upload() && that.getDevice().upload();
    		//if(success) {
    		//	return cuBLASdotDevice(thisOffset, (ArrayFloatTensor) that, thatOffset, size);
    		//}
      		try {
				return cuBLASdotSlice(cublasHandle, thisOffset, that, thatOffset, size);
			} catch (Throwable e) {
				System.out.println("FloatTensor dot fail:"+e+" falling back to scalarDot");
				return scalarDot(this, thisOffset, that, thatOffset, size);
			}
      	}
		return scalarDot(this, thisOffset, that, thatOffset, size);
    }
    /**
     * If we are here, we have to slice the MemorySegment, if we are in a quantized type, we send the whole thing.
     * It is the responsibility of the subclass NOT to slice it, but to send the entire segment, as we are using 
     * math designed to work on the entire segment in the native methods.
     * @param cublasHandle the handle for CUBLAS
     * @param thisOffset offset into this segment
     * @param that the other segment
     * @param thatOffset offset into the other segment
     * @param size number of elements
     * @return the result of the sdot
     * @throws Throwable native method fail code
     */
    public float cuBLASdotSlice(long cublasHandle, int thisOffset, FloatTensor that, int thatOffset, int size) throws Throwable {
    	   if (this instanceof Q8_0FloatTensor q8) {
    	       MemorySegment qSeg = this.getSegment();//.sliceElements(thisOffset, size);
    	       MemorySegment kSeg = that.sliceElements(thatOffset, size);
    	     	if(DEBUG)
    	    		System.out.printf("%s thread:%s CUBLAS handle:%x thisOffset:%d thatOffset:%d size:%d this:%s that:%s%n", 
    	    				this.getClass().getName(), Thread.currentThread().getName(), cublasHandle, thisOffset, thatOffset, size, qSeg, kSeg);
    	        float result = (float) Llama3.sdotSliceQ8Handle.invokeExact(cublasHandle,
    	            qSeg, kSeg, size,
    	            GGMLType.Q8_0.getBlockSize(),
    	            thisOffset,
    	            GGMLType.Q8_0.getTypeSize(),
    	            GGMLType.FLOAT16_BYTES
    	        );
    	        Llama3.cublasHandlePool.release(cublasHandle);
    	        return result;
    	    } else {
    	        // Float: slice both sides
    	        MemorySegment qSeg   = sliceElements(thisOffset, size);
    	        MemorySegment kSeg   = that.sliceElements(thatOffset, size);
    	        assert qSeg.byteSize() == (long) size * Float.BYTES;
    	        assert kSeg.byteSize() == (long) size * Float.BYTES;
    	        float result = (float) Llama3.sdotSliceHandle.invokeExact(cublasHandle, qSeg, kSeg, size);
      	        Llama3.cublasHandlePool.release(cublasHandle);
      	        return result;
    	    }
    }
    void matmul(FloatTensor that, FloatTensor out, int dim0, int dim1) {
        Parallel.parallelFor(0, dim0, i -> out.setFloat(i, dot(Llama3.cublasHandlePool.acquire(), i * dim1, that, 0, dim1)));
    }
    void matmul(int context, FloatTensor[] that, FloatTensor[] out, int dim0, int dim1) {
        if (that.length != out.length) {
            throw new IllegalArgumentException(String.format("that.len=%d, out.len=%d", that.length, out.length));
        }
        Parallel.parallelForLong(0, dim0 * context, ti -> {
            int idxArr = (int) (ti / dim0);
            int i = (int) (ti % dim0);
            out[idxArr].setFloat(i, dot(Llama3.cublasHandlePool.acquire(), i * dim1, that[idxArr], 0, dim1)); 
        });
    }

    @FunctionalInterface
    interface AggregateFunction {
        float apply(float acc, float value);
    }

    float reduce(int thisOffset, int size, float seed, AggregateFunction reduce) {
        float result = seed;
        for (int i = 0; i < size; ++i) {
            result = reduce.apply(result, getFloat(thisOffset + i));
        }
        return result;
    }
    float sum(int thisOffset, int size) {
        return reduce(thisOffset, size, 0f, Float::sum);
    }
    float max(int thisOffset, int size) {
        return reduce(thisOffset, size, Float.NEGATIVE_INFINITY, Float::max);
    }
    void copyTo(int thisOffset, FloatTensor that, int thatOffset, int size) {
        that.mapWithIndexInPlace(thatOffset, size, (value, index) -> this.getFloat(index - thatOffset + thisOffset));
    }
    int argmax(int thisOffset, int size) {
        assert size > 0;
        int maxIndex = thisOffset;
        float maxValue = this.getFloat(maxIndex);
        int endIndex = thisOffset + size;
        for (int i = thisOffset; i < endIndex; ++i) {
            float f = this.getFloat(i);
            if (f > maxValue) {
                maxValue = f;
                maxIndex = i;
            }
        }
        return maxIndex;
    }
    int argmax() {
        return argmax(0, size());
    }

    @FunctionalInterface
    interface MapFunction {
        float apply(float value);
    }

    @FunctionalInterface
    interface MapWithIndexFunction {
        float apply(float value, int index);
    }

    FloatTensor mapInPlace(int thisOffset, int size, MapFunction mapFunction) {
        int endIndex = thisOffset + size;
        for (int i = thisOffset; i < endIndex; ++i) {
            setFloat(i, mapFunction.apply(getFloat(i)));
        }
        return this;
    }
    FloatTensor mapInPlace(MapFunction mapFunction) {
        return mapInPlace(0, size(), mapFunction);
    }
    FloatTensor mapWithIndexInPlace(int thisOffset, int size, FloatTensor.MapWithIndexFunction mapWithIndexFunction) {
        int endOffset = thisOffset + size;
        for (int i = thisOffset; i < endOffset; ++i) {
        	//System.out.println("setFloat:"+i+" of size:"+size);
            setFloat(i, mapWithIndexFunction.apply(getFloat(i), i));
        }
        return this;
    }
    FloatTensor addInPlace(int thisOffset, FloatTensor that, int thatOffset, int size) {
        return mapWithIndexInPlace(thisOffset, size, (value, index) -> value + that.getFloat(index - thisOffset + thatOffset));
    }
    FloatTensor addInPlace(FloatTensor that) {
        return addInPlace(0, that, 0, size());
    }
    FloatTensor multiplyInPlace(int thisOffset, FloatTensor that, int thatOffset, int size) {
        return mapWithIndexInPlace(thisOffset, size, (value, index) -> value * that.getFloat(index - thisOffset + thatOffset));
    }
    FloatTensor multiplyInPlace(FloatTensor that) {
        return multiplyInPlace(0, that, 0, size());
    }
    FloatTensor divideInPlace(int thisOffset, int size, float value) {
        return mapInPlace(thisOffset, size, f -> f / value);
    }
    FloatTensor fillInPlace(int thisOffset, int size, float value) {
        return mapInPlace(thisOffset, size, unused -> value);
    }
    FloatTensor softmaxInPlace(int thisOffset, int size) {
    	//if(USE_CUDA) {
    		/*try (Timer timer = Timer.log("CUDA SoftMax:"+String.valueOf(size), TimeUnit.MICROSECONDS)) {       	
    			// 1. Extract the slice into a flat float[]
    			float[] slice = new float[size];
    			for (int i = 0; i < size; i++) {
    				slice[i] = getFloat(thisOffset + i);
    			}
    			// 2. Call the JNI CUDA kernel
    			float[] softmaxed = Attn.softMax(slice, 1, size);
    			// 3. Write results back into this tensor
    			for (int i = 0; i < size; i++) {
    				setFloat(thisOffset + i, softmaxed[i]);
    			}
    			return this;
    		}*/
    	//} else {
    		//try (Timer timer = Timer.log("CPU SoftMax:"+String.valueOf(size),TimeUnit.MICROSECONDS)) {
    			// find max value (for numerical stability)
    			float maxVal = max(thisOffset, size);
    			// exp and sum
    			mapInPlace(thisOffset, size, f -> (float) Math.exp(f - maxVal));
    			float sum = sum(thisOffset, size);
    			// normalize
    			return divideInPlace(thisOffset, size, sum);
    		//}
    	//}
    }

    FloatTensor saxpyInPlace(int thisOffset, FloatTensor that, int thatOffset, int size, float a) {
        // this[thatOffset ... thatOffset + size) = a * that[thatOffset ... thatOffset + size) + this[thisOffset ... thisOffset + size)
        for (int i = 0; i < size; ++i) {
            setFloat(thisOffset + i, a * that.getFloat(thatOffset + i) + this.getFloat(thisOffset + i));
        }
        return this;
    }
    
    static float cosineSimilarity(FloatTensor a, FloatTensor b) {
    	long handle = Llama3.cublasHandlePool.acquire();
    	float dotProduct = a.dot(handle ,0, b, 0, a.size());
    	DoubleAdder aNormAdder = new DoubleAdder();
    	DoubleAdder bNormAdder = new DoubleAdder();
    	Parallel.parallelFor(0, a.size(), t -> {
    	    aNormAdder.add(a.getFloat(t) * a.getFloat(t));
    	    bNormAdder.add(b.getFloat(t) * b.getFloat(t));
    	});
    	float aNorm = (float) Math.sqrt(aNormAdder.sum());
    	float bNorm = (float) Math.sqrt(bNormAdder.sum());
    	Llama3.cublasHandlePool.release(handle);
    	return (dotProduct / (aNorm * bNorm));
    }
    
    public void verify() {
    	System.out.println("size:"+size());
      	System.out.println("Verified via String of length:"+toString().length());
    }
    
    public String toString() {
    	StringBuilder sb = new StringBuilder("[");
    	for(int i = 0; i < size(); i++) {
    		sb.append(getFloat(i));
    		if(i == (size()-1)) 
    			sb.append("]");
    		else
    			sb.append(",");
    	}
    	return sb.toString();
    }
	protected abstract long devicePtr(); 
	
	/*public float cuBLASdotDevice(long cublasHandle, int thisOffset, FloatTensor that, int thatOffset, int size) {
		// Preallocated device-side scalar for the result
		ResultScalarPool.Scalar s = ResultScalarPool.acquire();
		long dX = this.devicePtr() + (long)thisOffset * Float.BYTES;
		long dY = that.devicePtr() + (long)thatOffset * Float.BYTES;

		int rc = Gemm.sdotDevice(cublasHandle, size, dX, 1, dY, 1, s.dPtr);
		if (rc != 0) 
			throw new RuntimeException("sdotDevice rc=" + rc);

		// Copy back one float
		s.download(); // copies device->host for the one float
		float out = s.get();
		ResultScalarPool.release(s);
		return out;
	}*/
	
	public MemorySegment sliceElements(long elementOffset, long elementCount) {
		long off = getOffsetBytes(elementOffset);
		long len = getLengthBytes(elementCount, elementOffset);
		MemorySegment seg = getSegment();
		if (off + len > seg.byteSize()) {
			throw new IllegalArgumentException(
					"Slice out of bounds: off=" + off + " len=" + len + " segSize=" + seg.byteSize());
		}
		return seg.asSlice(off, len);
	}

    // Returns a lightweight read-only view (no allocation if possible).
    public abstract FloatSliceView sliceView(int offset, int length);
    // Export the slice as contiguous floats into a provided buffer (pooled).
    // Returns the same dst for chaining.
    public abstract float[] exportSlice(float[] dst, int dstOffset, int offset, int length);
    // Convenience: allocate from a pool and export
    public float[] exportSlicePooled(BufferPool pool, int offset, int length) {
    	float[] dst = pool.acquire(length);
    	if(DEBUG)
    		System.out.println(this.getClass().getName()+".exportSlicePooled dst="+(dst == null ? "pool acquire dst length="+length+" FAIL, null!": dst.length));
    	exportSlice(dst, 0, offset, length);
    	return dst;
    } 
}
