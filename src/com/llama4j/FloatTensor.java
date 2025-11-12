package com.llama4j;

import java.io.Externalizable;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.function.IntFunction;

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
	public static boolean DEBUG = false;
	public static final boolean USE_CUDA = true;
    static final int VECTOR_BIT_SIZE = Integer.getInteger("llama.VectorBitSize", VectorShape.preferredShape().vectorBitSize());
    static final boolean USE_VECTOR_API = VECTOR_BIT_SIZE != 0;
    private long devicePtr; // 0 if not uploaded
    
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
    public abstract Arena getArena();
    public abstract MemorySegment asSlice(long offSet1, long offSet2);
    public abstract MemorySegment getSegment();
    public abstract long getOffsetBytes(long offset);
    public abstract long getLengthBytes(long size, long offset);
    abstract GGMLType type();
    abstract int getHeadSize();
    abstract int getFormatType(); // for GPU side quantized conversion

    public static int numberOfElements(int... dimensions) {
        assert Arrays.stream(dimensions).allMatch(i -> i > 0);
        return Arrays.stream(dimensions).reduce(Math::multiplyExact).orElseThrow();
    }
    public static float scalarDot(FloatTensor thiz, int thisOffset, FloatTensor that, int thatOffset, int size) {
        float result = 0f;
        for (int j = 0; j < size; j++) {
            result += thiz.getFloat(thisOffset + j) * that.getFloat(thatOffset + j);
            //System.out.printf("CPU %d) dot1 = %.6f dot2 = %.6f result = %.6f %n", j, thiz.getFloat(thisOffset + j) , that.getFloat(thatOffset + j), result);
        }
        return result;
    }
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
      	if(USE_CUDA) {
    		//boolean success = getDevice().upload() && that.getDevice().upload();
    		//if(success) {
    		//	return cuBLASdotDevice(thisOffset, (ArrayFloatTensor) that, thatOffset, size);
    		//}
      		try {
      			//float result1, result2;
      			//try (Timer timer = Timer.log("cublas dot:"+String.valueOf(size),TimeUnit.MICROSECONDS)) {
				//return cuBLASdotSlice(this, thisOffset, that, thatOffset, size);
				return cudaSdotSliceDevice(this, thisOffset, that, thatOffset, size);
      			//}
    			//try (Timer timer = Timer.log("scalar dot:"+String.valueOf(size),TimeUnit.MICROSECONDS)) {
				//result2 = return scalarDot(this, thisOffset, that, thatOffset, size);
    			//}
    			//if(result1 != result2) {
    			//	System.out.println("FloatTensor dot results differ:"+result1+", "+result2);
    			//}
    			//return result1;
			} catch (Throwable e) {
				System.out.println("FloatTensor dot fail:"+e+" falling back to scalarDot");
				return scalarDot(this, thisOffset, that, thatOffset, size);
			}
      	}
		return scalarDot(this, thisOffset, that, thatOffset, size);
    }
 
    public static Object mutex = new Object();
    public static float cudaSdotSliceDevice(FloatTensor thiz, int thisOffset, FloatTensor that, int thatOffset, int size) throws Throwable {
    	if(DEBUG)
    		System.out.printf("%s thread:%s thisOffset:%d thatOffset:%d size:%d%n", 
    				thiz.getClass().getName(), Thread.currentThread().getName(), thisOffset, thatOffset, size);
    		synchronized(mutex) {
     		float result2 = scalarDot(thiz, thisOffset, that, thatOffset, size);
     		//float sdotSliceDevice(const uint8_t* qA, int indexA, int formatA, int blockSizeA, int typeSizeA, int headerBytesA,
			//	    const uint8_t* qB, int indexB, int formatB, int blockSizeB, int typeSizeB, int headerBytesB,
			//	    int N)
    		float result = (float) Llama3.sdotSliceDeviceHandle.invokeExact(
    				thiz.devicePtrOr0(), thisOffset, thiz.getFormatType(), thiz.type().getBlockSize(), thiz.type().getTypeSize(), thiz.getHeadSize(),
    				that.devicePtrOr0(), thatOffset, that.getFormatType(), that.type().getBlockSize(), that.type().getTypeSize(), that.getHeadSize(),
    				size);
    		if(result != result2)
    			System.out.printf("Sdot values dont match: Q8 %s %s thread:%s thisOffset:%d thatOffset:%d size:%d  r=%.6f r2=%.6f%n", 
    				thiz.getClass().getName(), that.getClass().getName(), Thread.currentThread().getName(), thisOffset, thatOffset, size, result, result2);
      		return result;
    		}
    }
    
    public void allocDevice() {
    	devicePtr = allocDevice(getSegment().byteSize());
    }
    
    public void freeDevice() {
    	if(isOnDevice()) {
    		DeviceMemoryLedger.release(getSegment().byteSize());
    		try {
				Llama3.freeDevicePtr.invokeExact(devicePtr);
			} catch (Throwable e) {
			}
    		devicePtr = 0L;
    	}
    }
    
    public static long allocDevice(long bytes) {
    	if(DeviceMemoryLedger.tryReserve(bytes)) {
    		try {
				return (long) Llama3.allocDevicePtr.invokeExact(bytes);
			} catch (Throwable e) {
				e.printStackTrace();
				DeviceMemoryLedger.onAllocationFailure();
			}
    	}
		return 0L;
    }
    
    /**
     * every time you need to pass a device buffer into a downcall, 
     * call devSeg(ptr, size, arena) and get a properly bounded MemorySegment. 
     * avoids repeating FFI Address/reinterpret 
     * @param devicePtr The GPU device pointer from cudaMalloc
     * @param bytes the number of bytes in the segment
     * @param arena The Arena that allocated the MemorySegment
     * @return
     */
    static MemorySegment devSeg(long devicePtr, long bytes, Arena arena) {
    	if(devicePtr == 0L)
    		throw new RuntimeException("devicePtr is unallocated for devSeg call of "+bytes+" bytes using Arena "+arena);
    	MemorySegment base = MemorySegment.ofAddress(devicePtr);
    	return base.reinterpret(bytes, arena, null);
    }
    public long devicePtrOr0() {
        return isOnDevice() ? devicePtr : 0L;
    }
    public boolean isOnDevice() {
        return devicePtr != 0L;
    }
    public void copyHostToDevice() {
    	MemorySegment seg = getSegment();
    	try {
    		Llama3.copyHostToDeviceMH.invokeExact(seg, devicePtr, seg.byteSize());
    	} catch (Throwable e) {
    		e.printStackTrace();
    	}
    }
    public void copyDeviceToHost() {
    	MemorySegment seg = getSegment();
    	seg = devSeg(devicePtrOr0(), (long)size(), getArena());
    	try {
    		if(isOnDevice())
    			Llama3.copyDeviceToHostMH.invokeExact(devicePtr, seg, (long)size());
    		else
    			throw new RuntimeException("Device is not initialized for segment:"+getSegment()+" tensor:"+this);
    	} catch (Throwable e) {
    		e.printStackTrace();
    	}
    }

    static void rmsnormGpu(FloatTensor out, FloatTensor x, FloatTensor weight, int size, float eps) throws Throwable {
    	MemorySegment xDev  = devSeg(x.devicePtrOr0(),      (long) size * Float.BYTES, x.getArena());
        MemorySegment wDev  = devSeg(weight.devicePtrOr0(), (long) size * Float.BYTES, weight.getArena());
        MemorySegment oDev  = devSeg(out.devicePtrOr0(),    (long) size * Float.BYTES, out.getArena());
        Llama3.launchRmsnorm.invokeExact(xDev, wDev, oDev, size, eps);
    }
    
    public static FloatTensor[] loadArrayOfF32Tensors(int size, IntFunction<GGMLTensorEntry> get) {
        FloatTensor[] array = new FloatTensor[size];
        for (int i = 0; i < size; i++) {
            GGMLTensorEntry e = get.apply(i);
            if (e.ggmlType() != GGMLType.F32) throw new UnsupportedOperationException("Expected F32");
            array[i] = new F32FloatTensor(FloatTensor.numberOfElements(e.shape()), e.memorySegment());
        }
        return array;
    }
    
    void matmul(FloatTensor that, FloatTensor out, int dim0, int dim1) {
        Parallel.parallelFor(0, dim0, i -> out.setFloat(i, dot(i * dim1, that, 0, dim1)));
    }
    void matmul(int context, FloatTensor[] that, FloatTensor[] out, int dim0, int dim1) {
        if (that.length != out.length) {
            throw new IllegalArgumentException(String.format("that.len=%d, out.len=%d", that.length, out.length));
        }
        Parallel.parallelForLong(0, dim0 * context, ti -> {
            int idxArr = (int) (ti / dim0);
            int i = (int) (ti % dim0);
            out[idxArr].setFloat(i, dot(i * dim1, that[idxArr], 0, dim1)); 
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
    	if(USE_CUDA) {
    		MemorySegment xDev = devSeg(devicePtrOr0(), (long) size * Float.BYTES, getArena());
    		try {
    			return (FloatTensor) Llama3.launchSoftmaxInplace.invokeExact(xDev, size, 1, thisOffset);
    		} catch (Throwable e) {}
    	}
    	//try (Timer timer = Timer.log("CPU SoftMax:"+String.valueOf(size),TimeUnit.MICROSECONDS)) {
    	// find max value (for numerical stability)
    	float maxVal = max(thisOffset, size);
    	// exp and sum
    	mapInPlace(thisOffset, size, f -> (float) Math.exp(f - maxVal));
    	float sum = sum(thisOffset, size);
    	// normalize
    	return divideInPlace(thisOffset, size, sum);
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
    	float dotProduct = a.dot(0, b, 0, a.size());
    	DoubleAdder aNormAdder = new DoubleAdder();
    	DoubleAdder bNormAdder = new DoubleAdder();
    	Parallel.parallelFor(0, a.size(), t -> {
    	    aNormAdder.add(a.getFloat(t) * a.getFloat(t));
    	    bNormAdder.add(b.getFloat(t) * b.getFloat(t));
    	});
    	float aNorm = (float) Math.sqrt(aNormAdder.sum());
    	float bNorm = (float) Math.sqrt(bNormAdder.sum());
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
