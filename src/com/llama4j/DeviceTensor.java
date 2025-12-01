package com.llama4j;

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;

import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;

/**
 * Tensor of device pointers, designed to be used device-side.
 * <p>
 * Not a strict tensor, but rather just a sequence of long device pointers
 */
public class DeviceTensor implements Externalizable, Comparable {
	public static boolean DEBUG = false;
    static final int VECTOR_BIT_SIZE = Integer.getInteger("llama.VectorBitSize", VectorShape.preferredShape().vectorBitSize());
    static final boolean USE_VECTOR_API = VECTOR_BIT_SIZE != 0;

    private long devicePtr; // 0 if not uploaded
    private volatile DeviceMemoryReclaim deviceReclaim;
    private boolean uploaded = false;
	//public static Object mutex = new Object();
	
	MemorySegment memorySegment;
    
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
    static final VectorSpecies<Long> L_SPECIES;
    static final VectorSpecies<Integer> I_SPECIES;
    static final VectorSpecies<Short> S_SPECIES_HALF;

    static {
        if (USE_VECTOR_API) {
            L_SPECIES = VectorShape.forBitSize(VECTOR_BIT_SIZE).withLanes(long.class);
            I_SPECIES = L_SPECIES.withLanes(int.class);
            S_SPECIES_HALF = VectorShape.forBitSize(I_SPECIES.vectorBitSize() / 2).withLanes(short.class);
            assert I_SPECIES.length() == S_SPECIES_HALF.length();
        } else {
            L_SPECIES = null;
            I_SPECIES = null;
            S_SPECIES_HALF = null;
        }
    }

	public DeviceTensor() {
	}

	DeviceTensor(long[] values) {
			// allocate off-heap space 
			memorySegment = getArena().allocate(ValueLayout.JAVA_LONG, values.length);
			// bulk copy from heap arrays if you have them
			MemorySegment.copy(values, 0, memorySegment, ValueLayout.JAVA_LONG, 0, values.length);
	}
	
	DeviceTensor(FloatTensor[] ftensors) {
		long[] l = new long[ftensors.length];
		for(int i = 0; i < ftensors.length; i++) {
			l[i] = ftensors[i].devicePtrOr0();
			if(l[i] == 0L)
				throw new RuntimeException("Attempt to use unallocated FloatTensor "+l[i]);
		}
		memorySegment = getArena().allocate(ValueLayout.JAVA_LONG, l.length);
		MemorySegment.copy(l, 0, memorySegment, ValueLayout.JAVA_LONG, 0, l.length);
	}
	
    public static int numberOfElements(int... dimensions) {
        assert Arrays.stream(dimensions).allMatch(i -> i > 0);
        return Arrays.stream(dimensions).reduce(Math::multiplyExact).orElseThrow();
    }
    /**
     * Allocate the memorySegment to device pointer and register it with DeviceReclaim
     */
    public void allocDevice() {
    	devicePtr = allocDevice(getSegment().byteSize());
     	deviceReclaim = new DeviceMemoryReclaim(this);	
    }
    /**
     * Call the freeDevicePtr FFI handle to free device memory, set devicePtr to 0 call {@link DeviceManager#reclaim} {@link DeviceMemoryLedger#release}
     */
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
    
    public void copyHostToDevicePointerTable(String id) {
        MemorySegment hostSeg = getSegment();
        long bytes = totalBytes();
        if (!isAllocated())
            throw new RuntimeException("Pointer table "+id+" is not allocated for HostToDevice");
        try {
            // Signature should be (hostSeg, devicePtr, bytes)
            Llama3.copyHostToDeviceMH.invokeExact(hostSeg, devicePtr, bytes);
            uploaded = true;
        } catch (Throwable e) {
            throw new RuntimeException("HostToDevice transfer failed for pointer table:"+id, e);
        }
    }
    
    public void copyDeviceToHostPointerTable(String id) {
        long bytes = totalBytes();
        if (!isAllocated())
            throw new RuntimeException("Pointer table "+id+" is not allocated for DeviceToHost");
        MemorySegment hostSeg = getSegment();
        try {
            Llama3.copyDeviceToHostMH.invokeExact(devicePtrOr0(), hostSeg.address(), bytes);
        } catch (Throwable e) {
            throw new RuntimeException("DeviceToHost transfer failed for id:"+id+", "+ this.getSegment(), e);
        }
 
    }
     
    @FunctionalInterface
    interface AggregateFunction {
        long apply(long acc, long value);
    }

     long reduce(int thisOffset, int size, long seed, AggregateFunction reduce) {
        long result = seed;
        for (int i = 0; i < size; ++i) {
            result = reduce.apply(result, getLong(thisOffset + i));
        }
        return result;
    }
    float sum(int thisOffset, int size) {
        return reduce(thisOffset, size, 0L, Long::sum);
    }
    float max(int thisOffset, int size) {
        return reduce(thisOffset, size, -Long.MAX_VALUE, Long::max);
    }
    void copyTo(int thisOffset, DeviceTensor that, int thatOffset, int size) {
        that.mapWithIndexInPlace(thatOffset, size, (value, index) -> this.getLong(index - thatOffset + thisOffset));
    }
    int argmax(int thisOffset, int size) {
        assert size > 0;
        int maxIndex = thisOffset;
        float maxValue = this.getLong(maxIndex);
        int endIndex = thisOffset + size;
        for (int i = thisOffset; i < endIndex; ++i) {
            float f = this.getLong(i);
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
        long apply(long value);
    }

    @FunctionalInterface
    interface MapWithIndexFunction {
        long apply(long value, int index);
    }

    DeviceTensor mapInPlace(int thisOffset, int size, MapFunction mapFunction) {
        int endIndex = thisOffset + size;
        for (int i = thisOffset; i < endIndex; ++i) {
            setLong(i, mapFunction.apply(getLong(i)));
        }
        return this;
    }
    DeviceTensor mapInPlace(MapFunction mapFunction) {
        return mapInPlace(0, size(), mapFunction);
    }
    DeviceTensor mapWithIndexInPlace(int thisOffset, int size, DeviceTensor.MapWithIndexFunction mapWithIndexFunction) {
        int endOffset = thisOffset + size;
        for (int i = thisOffset; i < endOffset; ++i) {
        	//System.out.println("setFloat:"+i+" of size:"+size);
            setLong(i, mapWithIndexFunction.apply(getLong(i), i));
        }
        return this;
    }
    DeviceTensor addInPlace(int thisOffset, DeviceTensor that, int thatOffset, int size) {
        return mapWithIndexInPlace(thisOffset, size, (value, index) -> value + that.getLong(index - thisOffset + thatOffset));
    }
    DeviceTensor addInPlace(DeviceTensor that) {
        return addInPlace(0, that, 0, size());
    }
    DeviceTensor multiplyInPlace(int thisOffset, DeviceTensor that, int thatOffset, int size) {
        return mapWithIndexInPlace(thisOffset, size, (value, index) -> value * that.getLong(index - thisOffset + thatOffset));
    }
    DeviceTensor multiplyInPlace(DeviceTensor that) {
        return multiplyInPlace(0, that, 0, size());
    }
    DeviceTensor divideInPlace(int thisOffset, int size, Long value) {
        return mapInPlace(thisOffset, size, f -> f / value);
    }

	public static DeviceTensor allocate(int... dims) {
		int numberOfElements = DeviceTensor.numberOfElements(dims);
		return new DeviceTensor(new long[numberOfElements]);
	}

	public int size() {
		return (int) (memorySegment.byteSize() / Long.BYTES);
	}

	public long getLong(int index) {
		return memorySegment.getAtIndex(ValueLayout.JAVA_LONG, index);
	}

	public void setLong(int index, long value) {
		setModified();
		memorySegment.setAtIndex(ValueLayout.JAVA_LONG, index, value);
	}

	protected long totalBytes() { 
		return size() * (long) Long.BYTES; 
	}

	public boolean isImmutable() {
		return false;
	}   
	
	public DeviceTensor fillInPlace(int thisOffset, int size, long value) {
		setModified();
		for(int index = thisOffset; index < thisOffset+size; index++)
			memorySegment.setAtIndex(ValueLayout.JAVA_LONG, index, value);
		return this;
	}

	public LongVector getLongVector(VectorSpecies<Long> species, int index) {
		if (!USE_VECTOR_API) {
			throw new UnsupportedOperationException();
		}
		return LongVector.fromMemorySegment(species, memorySegment, (long) index * Long.BYTES, ByteOrder.nativeOrder());
	}

	public Arena getArena() {
		return Llama3.autoArena;
	}
	
	public MemorySegment getSegment() {
		return memorySegment;
	}

	public long getOffsetBytes(long elementOffset) {
		long blockIndex = elementOffset / GGMLType.I64.getBlockSize();
		return blockIndex * GGMLType.I64.getTypeSize();
	}

	public long getLengthBytes(long elementCount, long elementOffset) {
		long startBlock = elementOffset / GGMLType.I64.getBlockSize();
		long endBlock   = (elementOffset + elementCount - 1) / GGMLType.I64.getBlockSize();
		long blocks     = (endBlock - startBlock + 1);
		return blocks * GGMLType.I64.getTypeSize();
	}

	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		out.writeInt(size());
		for(int i = 0; i < size(); i++)
			out.writeLong(getLong(i));
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
		int vsize = in.readInt();
		// allocate off-heap space for headSize floats
		memorySegment = getArena().allocate(ValueLayout.JAVA_LONG, vsize);
		for(int i = 0; i < vsize; i++)
			setLong(i, in.readLong());
	}

	@Override
	public int compareTo(Object o) {
		return Arrays.compare(memorySegment.toArray(ValueLayout.JAVA_LONG),((DeviceTensor)o).getSegment().toArray(ValueLayout.JAVA_LONG));
	}
	
	@Override   
    public String toString() {
    	StringBuilder sb = new StringBuilder("[");
    	for(int i = 0; i < size(); i++) {
    		sb.append(getLong(i));
    		if(i == (size()-1)) 
    			sb.append("]");
    		else
    			sb.append(",");
    	}
    	return sb.toString();
    }
}
