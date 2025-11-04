package com.llama4j;

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import com.neocoretechs.cublas.DeviceBuffer;
import com.neocoretechs.cublas.Gemm;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

final class Q8_0FloatTensor extends FloatTensor implements Externalizable, Comparable {
	private static final long serialVersionUID = -1L;
	private transient DeviceBuffer device;      // device residency
	
	int size;
    transient MemorySegment memorySegment;

    public Q8_0FloatTensor() {
        //this.device = new DeviceBuffer(memorySegment.asByteBuffer(), GGMLType.Q8_0.getBlockSize(), GGMLType.Q8_0.getTypeSize(), GGMLType.FLOAT16_BYTES, DeviceBuffer.GGUFQ.Q8_0.ordinal());
    }
    
    public Q8_0FloatTensor(int size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
        //this.device = new DeviceBuffer(memorySegment.asByteBuffer(), GGMLType.Q8_0.getBlockSize(), GGMLType.Q8_0.getTypeSize(), GGMLType.FLOAT16_BYTES, DeviceBuffer.GGUFQ.Q8_0.ordinal());
    }

    @Override
    public int size() {
        return size;
    }

	@Override
	public MemorySegment asSlice(long offSet1, long offSet2) {
		return memorySegment.asSlice(offSet1, offSet2);
	}
	
	@Override
	public MemorySegment getSegment() {
		return memorySegment;
	}
	
    @Override
    public void setFloat(int index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, int index) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    @Override
    public GGMLType type() {
        return GGMLType.Q8_0;
    }

    @Override
    public float getFloat(int index) {
        assert 0 <= index && index < size;
        int blockIndex = index / GGMLType.Q8_0.getBlockSize();
        int withinBlockIndex = index % GGMLType.Q8_0.getBlockSize();
        int blockOffset = blockIndex * GGMLType.Q8_0.getTypeSize();
        byte quant = readByte(memorySegment, blockOffset + GGMLType.FLOAT16_BYTES + withinBlockIndex);
        float scale = Float.float16ToFloat(readShort(memorySegment, blockOffset));
        return quant * scale;
    }

    public static final ValueLayout.OfShort JAVA_SHORT_LE = ValueLayout.JAVA_SHORT.withOrder(ByteOrder.LITTLE_ENDIAN);

    @Override
    public float dot(long cublasHandle, int thisOffset, FloatTensor that, int thatOffset, int size) {
    	if(FloatTensor.USE_CUDA) {
    		try {
				return cuBLASdotSlice(cublasHandle, thisOffset, that, thatOffset, size);
			} catch (Throwable e) {
				//e.printStackTrace();
				System.out.println("Failed to invoke sdotSliceQ8Handle:"+e.getMessage()+" default to CPU...");
			   	if (FloatTensor.USE_VECTOR_API) {
	        		return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
	        	}
	        	return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
			}
    		//return cuBLASdot(thisOffset, (ArrayFloatTensor) that, thatOffset, size);
    		//boolean success = device.upload() && that.getDevice().upload();
    		//if(success) {
    			//return cuBLASdotDevice(thisOffset, (ArrayFloatTensor) that, thatOffset, size);
    		//}
    		// default to CPU
    	   	//if (FloatTensor.USE_VECTOR_API) {
        	//	return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        	//}
        	//return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    	}
    	if (FloatTensor.USE_VECTOR_API) {
    		return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
    	}
    	return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }
    /**
     * dot product using GPU
     * @param thisOffset
     * @param that
     * @param thatOffset
     * @param size
     * @return
    public float cuBLASdot(long cublasHandle, int thisOffset, FloatTensor that, int thatOffset, int size) {
    	// 1. Export both slices into float[]
    	float[] a = this.exportSlice(new float[size], 0, thisOffset, size);
    	float[] b = that.exportSlice(new float[size], 0, thatOffset, size);
    	// 2. Prepare result buffer
    	float[] r = new float[1];
    	int rc = Gemm.sdot(cublasHandle, size, a, 1, b, 1, r);
    	if (rc != 0) throw new RuntimeException("JNI error " + rc);
    	return r[0];
    }*/
    
    @Override
    public long devicePtr() { return device.devicePtr; }
    
	@Override
	public DeviceBuffer getDevice() {
		return device;
	}
    @Override
    public long getOffsetBytes(long elementOffset) {
        long blockIndex = elementOffset / GGMLType.Q8_0.getBlockSize();
        return blockIndex * GGMLType.Q8_0.getTypeSize();
    }

    @Override
    public long getLengthBytes(long elementCount, long elementOffset) {
        long startBlock = elementOffset / GGMLType.Q8_0.getBlockSize();
        long endBlock   = (elementOffset + elementCount - 1) / GGMLType.Q8_0.getBlockSize();
        long blocks     = (endBlock - startBlock + 1);
        return blocks * GGMLType.Q8_0.getTypeSize();
    }

    /**
     * dot product using vector extensions
     * @param thiz
     * @param thisOffset
     * @param that
     * @param thatOffset
     * @param size
     * @return
     */
    private static float vectorDot(Q8_0FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;

        // Align thisOffset + startIndex to type().getBlockSize().
        assert Integer.bitCount(GGMLType.Q8_0.getBlockSize()) == 1 : "power of 2";
        int alignmentBound = Math.min(size, -thisOffset & (GGMLType.Q8_0.getBlockSize() - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }
        assert (thisOffset + j) % GGMLType.Q8_0.getBlockSize() == 0;

        FloatVector val = FloatVector.zero(F_SPECIES);
        int blockOffset = (thisOffset + j) / GGMLType.Q8_0.getBlockSize() * GGMLType.Q8_0.getTypeSize();
        int upperBound = size / GGMLType.Q8_0.getBlockSize() * GGMLType.Q8_0.getBlockSize();
        for (; j < upperBound; j += GGMLType.Q8_0.getBlockSize(), blockOffset += GGMLType.Q8_0.getTypeSize()) {
            float wScaleValue = Float.float16ToFloat(readShort(thiz.memorySegment, blockOffset));
            var wScale = FloatVector.broadcast(F_SPECIES, wScaleValue);
            switch (F_SPECIES.vectorBitSize()) {
                case 512 -> {
                    var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_256, thiz.memorySegment, blockOffset + GGMLType.FLOAT16_BYTES, ByteOrder.LITTLE_ENDIAN);
                    var sum0 = that.getFloatVector(F_SPECIES, thatOffset + j + 0 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 0));
                    var sum1 = that.getFloatVector(F_SPECIES, thatOffset + j + 1 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 1));
                    val = sum0.add(sum1).fma(wScale, val);
                }
                case 256 -> {
                    var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_256, thiz.memorySegment, blockOffset + GGMLType.FLOAT16_BYTES, ByteOrder.LITTLE_ENDIAN);
                    var sum0 = that.getFloatVector(F_SPECIES, thatOffset + j + 0 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 0));
                    var sum1 = that.getFloatVector(F_SPECIES, thatOffset + j + 1 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 1));
                    var sum2 = that.getFloatVector(F_SPECIES, thatOffset + j + 2 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 2));
                    var sum3 = that.getFloatVector(F_SPECIES, thatOffset + j + 3 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 3));
                    val = sum0.add(sum1).add(sum2).add(sum3).fma(wScale, val);
                }
                case 128 -> {
                    // This loop cannot be unrolled, why?
                    for (int i = 0; i < 2; ++i) {
                        var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment, blockOffset + GGMLType.FLOAT16_BYTES + i * ByteVector.SPECIES_128.vectorByteSize(), ByteOrder.LITTLE_ENDIAN);
                        var sum0 = that.getFloatVector(F_SPECIES, thatOffset + j + i * 16 + 0 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 0));
                        var sum1 = that.getFloatVector(F_SPECIES, thatOffset + j + i * 16 + 1 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 1));
                        var sum2 = that.getFloatVector(F_SPECIES, thatOffset + j + i * 16 + 2 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 2));
                        var sum3 = that.getFloatVector(F_SPECIES, thatOffset + j + i * 16 + 3 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 3));
                        val = sum0.add(sum1).add(sum2).add(sum3).fma(wScale, val);
                    }
                }
                default -> throw new UnsupportedOperationException(F_SPECIES.toString());
            }
        }
        result += val.reduceLanes(VectorOperators.ADD);

        // Remaining entries.
        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }
    
    public float cuBLASdotSlice(int thisOffset, FloatTensor that, int thatOffset, int size) throws Throwable {
        MemorySegment qSeg = this.getSegment();//.sliceElements(thisOffset, size);
        MemorySegment kSeg = that.getSegment();//.sliceElements(thatOffset, size);
        //System.out.printf("sdotSliceQ8Handle( qSeq=%s, kSeg=%s, size=%d, blockSize=%d, thisOffset=%d, typeSize=%d, FLOAT16_BYTES=%d);%n",
        //		qSeg,kSeg,size,GGMLType.Q8_0.getBlockSize(),
        //        thisOffset,
        //        GGMLType.Q8_0.getTypeSize(),
        //        GGMLType.FLOAT16_BYTES);
        float result = (float) Llama3.sdotSliceQ8Handle.invokeExact(Llama3.cublasHandle,
            qSeg, kSeg, size,
            GGMLType.Q8_0.getBlockSize(),
            thisOffset,
            GGMLType.Q8_0.getTypeSize(),
            GGMLType.FLOAT16_BYTES
        );
        return result;
    }
    
	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		out.writeInt(size);
		out.writeLong(memorySegment.byteSize());
		out.write(memorySegment.toArray(ValueLayout.JAVA_BYTE));
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
		size = in.readInt();
		long bs = in.readLong();
		memorySegment = Arena.ofAuto().allocate(bs, 1);
		for(int i = 0; i < bs; i++)
			memorySegment.set(ValueLayout.JAVA_BYTE, i, (byte)(in.read() & 0xFF));
	}

	@Override
	public int compareTo(Object o) {
		for(int i = 0; i < memorySegment.byteSize(); i++) {
			byte b;
			if(i >= ((Q8_0FloatTensor)o).memorySegment.byteSize())
				return 1;
			else
				b = ((Q8_0FloatTensor)o).memorySegment.get(ValueLayout.JAVA_BYTE, i);
			if(memorySegment.get(ValueLayout.JAVA_BYTE,i) > b)
				return 1;
			if(memorySegment.get(ValueLayout.JAVA_BYTE,i) < b)
				return -1;
		}
		return 0;
	}
	@Override
	public FloatSliceView sliceView(int offset, int length) {
		return new Q8_0SliceView(memorySegment, offset, length);
	}
	@Override
	public float[] exportSlice(float[] dst, int dstOffset, int offset, int length) {
		// Dequantize block-wise to dst; matches your getFloat math but vectorized
		final int BS = GGMLType.Q8_0.getBlockSize();     // e.g., 32
		final int TS = GGMLType.Q8_0.getTypeSize();      // bytes per block
		int i = 0;
		// Head: align to block boundary
		int head = Math.min(length, (-offset) & (BS - 1));
		for (; i < head; i++) {
			dst[dstOffset + i] = getFloat(offset + i);
		}
		// Middle: block-wise fast path
		int blockIndex = (offset + i) / BS;
		int withinOffset = (offset + i) % BS;
		if (withinOffset != 0) {
			// Shouldn’t happen because we aligned above, but guard anyway
			int toBoundary = BS - withinOffset;
			int bound = Math.min(length - i, toBoundary);
			for (int k = 0; k < bound; k++) 
				dst[dstOffset + i + k] = getFloat(offset + i + k);
			i += bound;
			blockIndex = (offset + i) / BS;
		}
		int remaining = length - i;
		int fullBlocks = remaining / BS;
		int tail = remaining % BS;
		int blockOffsetBytes = blockIndex * TS;
		for (int b = 0; b < fullBlocks; b++, blockOffsetBytes += TS) {
			float scale = Float.float16ToFloat(readShort(memorySegment, blockOffsetBytes));
			// Read BS quant bytes after the scale
			for (int q = 0; q < BS; q++) {
				byte quant = readByte(memorySegment, blockOffsetBytes + GGMLType.FLOAT16_BYTES + q);
				dst[dstOffset + i + q] = quant * scale;
			}
			i += BS;
		}
		// Tail: remainder
		for (int t = 0; t < tail; t++) {
			dst[dstOffset + i + t] = getFloat(offset + i + t);
		}
		return dst;
	}
	final class Q8_0SliceView implements FloatSliceView {
		final MemorySegment seg; 
		final int base; 
		final int len;
		Q8_0SliceView(MemorySegment seg, int base, int len) { 
			this.seg=seg;
			this.base=base; 
			this.len=len; 
		}
		public int length() { 
			return len; 
		}
		public float get(int i) {
			int idx = base + i;
			int blockIndex = idx / GGMLType.Q8_0.getBlockSize();
			int withinBlockIndex = idx % GGMLType.Q8_0.getBlockSize();
			int blockOffset = blockIndex * GGMLType.Q8_0.getTypeSize();
			byte quant = FloatTensor.readByte(seg, blockOffset + GGMLType.FLOAT16_BYTES + withinBlockIndex);
			float scale = Float.float16ToFloat(FloatTensor.readShort(seg, blockOffset));
			return quant * scale;
		}
	}
}

