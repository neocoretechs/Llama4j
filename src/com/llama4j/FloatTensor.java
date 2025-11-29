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
	public static final boolean DO_SDOT_COMPARE = false;
    static final int VECTOR_BIT_SIZE = Integer.getInteger("llama.VectorBitSize", VectorShape.preferredShape().vectorBitSize());
    static final boolean USE_VECTOR_API = VECTOR_BIT_SIZE != 0;
	private boolean VERIFY_GPU_DATA = false;
	
    private long devicePtr; // 0 if not uploaded
    private boolean uploaded = false;
    private volatile DeviceMemoryReclaim deviceReclaim;
	public static int dontMatch = 0;
	public static int totalSdot = 0;
	//public static Object mutex = new Object();
    
    static short readShort(MemorySegment memorySegment, long offset) {
        return memorySegment.get(ValueLayout.JAVA_SHORT, offset);
    }  
    static int readInt(MemorySegment memorySegment, long offset) {
        return memorySegment.get(ValueLayout.JAVA_INT, offset);
    }
    static float readFloat(MemorySegment memorySegment, long offset) {
        return memorySegment.get(ValueLayout.JAVA_FLOAT, offset);
    }  
    static byte readByte(MemorySegment memorySegment, long offset) {
        return memorySegment.get(ValueLayout.JAVA_BYTE, offset);
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
    public abstract boolean isImmutable();
    public abstract int size();
    public abstract float getFloat(int index);
    public abstract void setFloat(int index, float value);
    abstract FloatVector getFloatVector(VectorSpecies<Float> species, int offset);
    public abstract Arena getArena();
    public abstract MemorySegment getSegment();
    public abstract long getOffsetBytes(long offset);
    public abstract long getLengthBytes(long size, long offset);
    abstract GGMLType type();
    abstract int getHeadSize();
    abstract int getFormatType(); // for GPU side quantized conversion
    // Explicit byte count vs element count
    protected abstract long totalBytes();

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
      	if(FloatTensor.USE_CUDA) 
      		return cudaDot(this, thisOffset, that, thatOffset, size);
		return scalarDot(this, thisOffset, that, thatOffset, size);
    }
    /**
     * If USE_CUDA flag is set, all sdot routes throug here.
     * @param thiz
     * @param thisOffset
     * @param that
     * @param thatOffset
     * @param size
     * @return
     */
    public static float cudaDot(FloatTensor thiz, int thisOffset, FloatTensor that, int thatOffset, int size) {
    	if(DEBUG)
    		System.out.printf("%s thread:%s thisOffset:%d thatOffset:%d size:%d%n", 
    				thiz.getClass().getName(), Thread.currentThread().getName(), thisOffset, thatOffset, size);
    	float result = 0.0f;
    	try {
    		if(Llama3.SDOT_CPU) {
    			result = DeviceManager.sdotCpu(thiz, thisOffset,that, thatOffset, size);
    		} else {
    			result = DeviceManager.sdot(thiz, thisOffset,that, thatOffset, size);
    		}
    		if(DO_SDOT_COMPARE)
    			compareSdotTest(result, thiz, thisOffset, that, thatOffset, size);
    	} catch(Throwable e) {
    		throw new RuntimeException(e);
    	}
    	return result;
    }
    /**
     * Compare the result of GPU sdot and CPU sdot to test numeric drift. Since sdot is the backbone of all our
     * computation, ensure the variance is within tolerance. Increase global counters with results.
     * @param GPUresult The result of the GPU sdot operation
     */
    private static synchronized void compareSdotTest(float GPUresult, FloatTensor thiz, int thisOffset, FloatTensor that, int thatOffset, int size) {
 		float result2 = scalarDot(thiz, thisOffset, that, thatOffset, size);
  		if(Math.abs(GPUresult - result2) > 1e-5f) {
			++dontMatch;
			//if(DEBUG)
				System.out.printf("Sdot values dont match: %s %s thread:%s thisOffset:%d thatOffset:%d size:%d  GPU=%.6f CPU=%.6f %d dont match out of %d%n", 
				thiz.getClass().getName(), that.getClass().getName(), Thread.currentThread().getName(), thisOffset, thatOffset, size, GPUresult, result2, dontMatch, totalSdot);	
		}
 		++totalSdot;
    }
    
    public void allocDevice() {
    	devicePtr = allocDevice(getSegment().byteSize());
     	deviceReclaim = new DeviceMemoryReclaim(this);	
    }
    
    public void freeDevice() {
    	if(isAllocated()) {
    		try {
				Llama3.freeDevicePtr.invokeExact(devicePtr);
			} catch (Throwable e) {}
       		DeviceMemoryLedger.release(getSegment().byteSize());
    		DeviceManager.reclaim(devicePtr);
    		uploaded = false;
    		devicePtr = 0L;
    	}
    }
    
    private static long allocDevice(long bytes) {
    	if(DeviceMemoryLedger.tryReserve(bytes)) {
    		try {
				return (long) Llama3.allocDevicePtr.invokeExact(bytes);
			} catch (Throwable e) {
				DeviceMemoryLedger.onAllocationFailure();
				throw new RuntimeException("Failed to reserve "+bytes+" on device!", e);
			}
    	}
    	DeviceMemoryLedger.onAllocationFailure();
		throw new RuntimeException("Failed to reserve "+bytes+" on device!");
    }
    
    public long devicePtrOr0() {
        return isAllocated() ? devicePtr : 0L;
    }
    public boolean isAllocated() {
        return devicePtr != 0L;
    }
    public boolean isUploaded() {
    	return uploaded;
    }
    public void setModified() {
    	uploaded = false;
    }
    public void copyHostToDevice(String id) {
        MemorySegment hostSeg = getSegment();
        long bytes = totalBytes();
        if (!isAllocated())
            throw new RuntimeException("Device "+id+" is not initialized for HostToDevice transfer: " + this.getSegment());
        try {
            // Signature should be (hostSeg, devicePtr, bytes)
            Llama3.copyHostToDeviceMH.invokeExact(hostSeg, devicePtr, bytes);
            uploaded = true;
        } catch (Throwable e) {
            throw new RuntimeException("HostToDevice transfer failed for id:"+id+", "+this, e);
        }
    }
    
    public void copyDeviceToHost(String id) {
        long bytes = totalBytes();
        if (!isAllocated())
            throw new RuntimeException("Device "+id+" is not initialized for DeviceToHost transfer: " + this.getSegment());
        MemorySegment hostSeg = getSegment();
        try {
            Llama3.copyDeviceToHostMH.invokeExact(devicePtrOr0(), hostSeg.address(), bytes);
        } catch (Throwable e) {
            throw new RuntimeException("DeviceToHost transfer failed for id:"+id+", "+ this.getSegment(), e);
        }
        if(USE_CUDA && VERIFY_GPU_DATA) {
        	try {
        		for(int i = 0; i < size(); i++) {
        			if(getFormatType() == 1) {
        				float f1 = getFloat(i);
        				float f2 = (float)Llama3.getFloatQ8.invokeExact(devicePtrOr0(), i, 1, type().getBlockSize(), type().getTypeSize(), getHeadSize());
        				if(f1 != f2)
        					System.out.println("Q8 Data mismatch: host="+f1+", "+f2+" ptr="+devicePtrOr0()+" id:"+id+" offset:"+i+" for tensor:"+this);
        			} else {
          				float f1 = getFloat(i);
          				float f2 = (float)Llama3.getFloat.invokeExact(devicePtrOr0(), i);
        					if(f1 != f2)
        						System.out.println("Float Data mismatch: host="+f1+", "+f2+" ptr="+devicePtrOr0()+" id:"+id+" offset:"+i+" for tensor:"+this);
        			}
        		}
        	} catch(Throwable e) {
        		throw new RuntimeException(e);
        	}
        }
        //if(id.contains("test"))
        //	System.out.printf("%s bytes=%d seg=%s%n",id,bytes,hostSeg);
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
    /**
     * Matrix multiply single tensor
     * @param that
     * @param out
     * @param dim0
     * @param dim1
     */
    void matmul(FloatTensor that, FloatTensor out, int dim0, int dim1) {
    	//long nanos1 = System.nanoTime();
    	if(FloatTensor.USE_CUDA) {
      		//System.out.println("GPU test STARTED for SINGLE: dim0:"+dim0+" dim1:"+dim1+" this len:"+size()+" that len:"+that.size()+" out len:"+out.size());
    		DeviceManager.matmul(this, that, out, dim0, dim1);
		   	return;
    	}
    	/*
    	out.copyDeviceToHost("comparison test");
    	nanos1 = System.nanoTime() - nanos1;
    	FloatTensor test = ArrayFloatTensor.allocate(dim0);
    	long nanos2 = System.nanoTime();
    	Parallel.parallelFor(0, dim0, i -> test.setFloat(i, dot(i * dim1, that, 0, dim1))); // this is matmul
    	nanos2 = System.nanoTime() - nanos2;
    	if(nanos1 > nanos2)
    		System.out.println("GPU was "+(nanos1-nanos2)+" ns slower");
    	else
    		System.out.println("GPU was "+(nanos2-nanos1)+" ns faster");
    	for(int i = 0; i < dim0; i++)
    		if(Math.abs(test.getFloat(i)-out.getFloat(i)) > 1e-5f)
    			System.out.println("Gpu does not match "+i+".) "+out.getFloat(i)+", "+test.getFloat(i));
   		System.out.println("GPU test ENDED for SINGLE: dim0:"+dim0+" dim1:"+dim1+" this len:"+size()+" that len:"+that.size()+" out len:"+out.size());
   		*/
    	//-----
    	// CPU implementation for vector processing if available
   		Parallel.parallelFor(0, dim0, i -> out.setFloat(i, dot(i * dim1, that, 0, dim1)));
    }
    /**
     * Matrix multiply array of tensors
     * @param context
     * @param that
     * @param out
     * @param dim0
     * @param dim1
     */
    void matmul(int context, FloatTensor[] that, FloatTensor[] out, int dim0, int dim1) {
    	if (that.length != out.length) {
    		throw new IllegalArgumentException(String.format("that.len=%d, out.len=%d", that.length, out.length));
    	}
    	if(FloatTensor.USE_CUDA) {
    		//Parallel.parallelFor(0, context, idxArr -> {
    		//System.out.println("GPU test STARTED for context:"+context+" dim0:"+dim0+" dim1:"+dim1+" this len:"+size()+" that array len:"+that.length+" out array len:"+out.length);
			for(int idxArr=0 ; idxArr < context; idxArr++) {
    				DeviceManager.matmul(this, that[idxArr], out[idxArr], dim0, dim1); // we do the dot inside the device
			}
			return;
 	   		/*for(int i = 0; i < context; i++)
	   			out[i].copyDeviceToHost("comparison test");
    		FloatTensor[] test = new FloatTensor[out.length];
    		for(int i = 0; i < out.length; i++)
    			test[i] = ArrayFloatTensor.allocate(out[i].size());
       		//for(int ti=0 ; ti < dim0*context; ti++) {
    	        	//nanos1 = System.nanoTime() - nanos1;
    	   	Parallel.parallelForLong(0, dim0 * context, ti -> {
        		int idxArr = (int) (ti / dim0);
        		int i = (int) (ti % dim0);
        		test[idxArr].setFloat(i, dot(i * dim1, that[idxArr], 0, dim1)); 
        	});
    	    //long nanos2 = System.nanoTime();
    	    //Parallel.parallelFor(0, dim0 , ti -> {
    	    //for(int ti = 0; ti < dim0; ti++)
    	    //	test.setFloat(i, dot(i * dim1, that[idxArr], 0, dim1));
    	    //});
    	    //if(nanos1 > nanos2)
    	    //	System.out.println("GPU was "+(nanos1-nanos2)+" ns slower");
    	    //else
    	    //	System.out.println("GPU was "+(nanos2-nanos1)+" ns faster");
    	    for(int ti = 0; ti < context * dim0; ti++) {
    	          	int idxArr = (int) (ti / dim0);
    	        	int i = (int) (ti % dim0);
    	        	if(Math.abs(test[idxArr].getFloat(i)-out[idxArr].getFloat(i)) > 1e-5f)
    	        		System.out.println("Gpu does not match "+i+".) "+out[idxArr].getFloat(i)+", "+test[idxArr].getFloat(i));
    	        }
    	        System.out.println("GPU test finished for context:"+context+" dim0:"+dim0+" dim1:"+dim1+" this len:"+size()+" that array len:"+that.length+" out array len:"+out.length);
    			//} catch (Throwable e) {
    			//	throw new RuntimeException(e);
    			//}*/
    		//});
    		//return;
 		   	//}
    	}
    	//
    	// matrix multiply using scalar dot product with vector processing if available, CPU otherwise
    	//
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
    /**
     * @param thisOffset offset into this tensor
     * @param size number of elements to fill
     * @param value value to fill with
     * @return this filled tensor
     */
    FloatTensor fillInPlace(int thisOffset, int size, float value) {
        return mapInPlace(thisOffset, size, unused -> value);
    }
    
    FloatTensor softmaxInPlace(int thisOffset, int size) {
    	if(USE_CUDA) {
    		DeviceManager.softmax(this, thisOffset, size);
    		return this;
    	}
    	//----------------------
    	// CPU implementation for vector processing if available
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
    /**
     * ax + y (get it? single prec. ax, plus y is saxpy)
     * @param thisOffset
     * @param that
     * @param thatOffset
     * @param size
     * @param a
     * @return
     */
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

}
