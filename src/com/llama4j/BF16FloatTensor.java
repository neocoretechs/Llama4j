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

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

final class BF16FloatTensor extends FloatTensor implements Externalizable, Comparable {
	private static final long serialVersionUID = -1L;

    int size;
    transient MemorySegment memorySegment;
	private transient DeviceBuffer device;      // device residency
    
    public BF16FloatTensor() {
    	//this.device = new DeviceBuffer(memorySegment.asByteBuffer(), GGMLType.BF16.getBlockSize(), GGMLType.BF16.getTypeSize(), GGMLType.BFLOAT16_BYTES, DeviceBuffer.GGUFQ.BF16.ordinal());
    }
    
    public BF16FloatTensor(int size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    	//this.device = new DeviceBuffer(memorySegment.asByteBuffer(), GGMLType.BF16.getBlockSize(), GGMLType.BF16.getTypeSize(), GGMLType.BFLOAT16_BYTES, DeviceBuffer.GGUFQ.BF16.ordinal());
    }

    @Override
    public int size() {
        return size;
    }

	@Override
	public MemorySegment getSegment() {
		return memorySegment;
	}
	
	@Override
	public MemorySegment asSlice(long offSet1, long offSet2) {
		return memorySegment.asSlice(offSet1, offSet2);
	}

    @Override
    public long getOffsetBytes(long elementOffset) {
        long blockIndex = elementOffset / GGMLType.BF16.getBlockSize();
        return blockIndex * GGMLType.BF16.getTypeSize();
    }
    
    @Override
    public long getLengthBytes(long elementCount, long elementOffset) {
        long startBlock = elementOffset / GGMLType.BF16.getBlockSize();
        long endBlock   = (elementOffset + elementCount - 1) / GGMLType.BF16.getBlockSize();
        long blocks     = (endBlock - startBlock + 1);
        return blocks * GGMLType.BF16.getTypeSize();
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
        return GGMLType.BF16;
    }

    @Override
    public float getFloat(int index) {
        assert 0 <= index && index < size;
        return bfloat16ToFloat(readShort(memorySegment, index * GGMLType.BFLOAT16_BYTES));
    }

    private float bfloat16ToFloat(short bfloat16) {
        return Float.intBitsToFloat(bfloat16 << 16);
    }
    
    public long devicePtr() { return device.devicePtr; }
    
	@Override
	public DeviceBuffer getDevice() {
		return device;
	}
    
    @Override
    public float dot(long cublasHandle, int thisOffset, FloatTensor that, int thatOffset, int size) {
    	if(FloatTensor.USE_CUDA) {
    		//return cuBLASdot(thisOffset, (ArrayFloatTensor) that, thatOffset, size);
    		//return cuBLASdotDevice(thisOffset, (ArrayFloatTensor) that, thatOffset, size);
      		try {
    			return cuBLASdotSlice(cublasHandle, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
    		} catch (Throwable e) {
    			if (FloatTensor.USE_VECTOR_API) {
    				return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
    			} else {
    				return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    			}
    		}
    	} else
    		if (FloatTensor.USE_VECTOR_API) {
    			return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
    		} else {
    			return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    		}
    }

    private static float vectorDot(BF16FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        assert S_SPECIES_HALF.length() == F_SPECIES.length();
        FloatVector val = FloatVector.zero(F_SPECIES);
        int upperBound = F_SPECIES.loopBound(size);
        for (int i = 0; i < upperBound; i += F_SPECIES.length()) {
            FloatVector thatVector = that.getFloatVector(F_SPECIES, thatOffset + i);
            ShortVector bfloat16 = ShortVector.fromMemorySegment(S_SPECIES_HALF, thiz.memorySegment, (thisOffset + i) * (long) GGMLType.BFLOAT16_BYTES, ByteOrder.LITTLE_ENDIAN);
            // BFloat16 to Float32 Conversion:
            //
            // ┌─[15]─┬─[14]───····───[7]─┬─[6]────····────[0]─┐
            // │ Sign │ Exponent (8 bits) │ Mantissa (7 bits)  │ BFloat16 Layout (16 bits)
            // └──────┴───────────────────┴────────────────────┘
            //    │             │                    │
            //    ▼             ▼                    ▼
            // ┌─[31]─┬─[30]───···───[23]─┬─[22]────···────[0]─┐
            // │ Sign │ Exponent (8 bits) │ Mantissa (23 bits) │ Float32 Layout (32 bits)
            // └──────┴───────────────────┴────────────────────┘
            FloatVector thizVector = bfloat16
                    .castShape(I_SPECIES, 0) // (int) vi
                    .lanewise(VectorOperators.LSHL, 16) // vi <<= 16
                    .reinterpretAsFloats(); // Float.intBitsToFloat(vi)
            val = thizVector.fma(thatVector, val);
        }
        float result = val.reduceLanes(VectorOperators.ADD);
        // Remaining entries.
        if (upperBound < size) {
            result += scalarDot(thiz, thisOffset + upperBound, that, thatOffset + upperBound, size - upperBound);
        }

        return result;
    }
    
    public float cuBLASdotSlice(int thisOffset, FloatTensor that, int thatOffset, int size) throws Throwable {
        MemorySegment qSeg = this.sliceElements(thisOffset, size);
        MemorySegment kSeg = that.sliceElements(thatOffset, size);
        float result = (float) Llama3.sdotSliceBF16Handle.invokeExact(
            qSeg, kSeg, size,
            GGMLType.BF16.getBlockSize(),
            thisOffset
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
			if(i >= ((BF16FloatTensor)o).memorySegment.byteSize())
				return 1;
			else
				b = ((BF16FloatTensor)o).memorySegment.get(ValueLayout.JAVA_BYTE, i);
			if(memorySegment.get(ValueLayout.JAVA_BYTE,i) > b)
				return 1;
			if(memorySegment.get(ValueLayout.JAVA_BYTE,i) < b)
				return -1;
		}
		return 0;
	}

	    // existing fields and methods…

	    @Override
	    public FloatSliceView sliceView(int offset, int length) {
	        return new BF16SliceView(memorySegment, offset, length);
	    }

	    @Override
	    public float[] exportSlice(float[] dst, int dstOffset, int offset, int length) {
	        // Vectorized widening: ShortVector -> IntVector (<<16) -> reinterpret as Float
	        final int bytesPerElem = GGMLType.BFLOAT16_BYTES; // 2
	        final int upper = F_SPECIES.loopBound(length);
	        int i = 0;

	        if (FloatTensor.USE_VECTOR_API) {
	            // Short lanes count equals float lanes count in your dot code
	            assert S_SPECIES_HALF.length() == F_SPECIES.length();

	            long baseBytes = (long) offset * bytesPerElem;
	            for (; i < upper; i += F_SPECIES.length(), baseBytes += (long) F_SPECIES.length() * bytesPerElem) {
	                ShortVector sv = ShortVector.fromMemorySegment(S_SPECIES_HALF, memorySegment, baseBytes, ByteOrder.LITTLE_ENDIAN);
	                // widen: short -> int << 16 -> reinterpret as float
	                FloatVector fv = sv
	                        .castShape(I_SPECIES, 0)
	                        .lanewise(VectorOperators.LSHL, 16)
	                        .reinterpretAsFloats();
	                fv.intoArray(dst, dstOffset + i);
	            }
	        }

	        // Tail
	        for (; i < length; i++) {
	            short s = readShort(memorySegment, (long) (offset + i) * bytesPerElem);
	            dst[dstOffset + i] = Float.intBitsToFloat((s & 0xFFFF) << 16);
	        }
	        return dst;
	    }
	    
	    @Override
	    public FloatTensor fillInPlace(int thisOffset, int size, float value) {
	        // Convert once, then splat
	        int bits = Float.floatToIntBits(value);
	        short bf16 = (short) (bits >>> 16);

	        long base = (long) thisOffset * GGMLType.BFLOAT16_BYTES;
	        for (int i = 0; i < size; i++) {
	            long addr = base + (long) i * GGMLType.BFLOAT16_BYTES;
	            memorySegment.set(ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), addr, bf16);
	        }
	        return this;
	    }
	    
	    final class BF16SliceView implements FloatSliceView {
	    	final MemorySegment seg; 
	    	final int base; final 
	    	int len;
	    	BF16SliceView(MemorySegment s, int base, int len) { 
	    		this.seg = s; 
	    		this.base = base; 
	    		this.len = len; 
	    	}
	    	public int length(){ return len; }
	    	public float get(int i){
	    		short s = readShort(seg, (long) (base + i) * GGMLType.BFLOAT16_BYTES);
	    		return Float.intBitsToFloat((s & 0xFFFF) << 16);
	    	}
	    }
}

