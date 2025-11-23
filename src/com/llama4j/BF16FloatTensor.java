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

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

final class BF16FloatTensor extends FloatTensor implements Externalizable, Comparable {
	private static final long serialVersionUID = -1L;

    int size;
    transient MemorySegment memorySegment;
    
    public BF16FloatTensor() {
    }
    
    public BF16FloatTensor(int size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public Arena getArena() {
    	return Llama3.autoArena;
    }
    
	@Override
	public MemorySegment getSegment() {
		return memorySegment;
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
    int getHeadSize() {
    	return GGMLType.FLOAT16_BYTES;
    }
    protected long totalBytes() { 
    	return size(); 
    }
    @Override
    int getFormatType() {
    	return 4;
    }
    @Override
    public float getFloat(int index) {
        assert 0 <= index && index < size;
        return bfloat16ToFloat(readShort(memorySegment, index * GGMLType.BFLOAT16_BYTES));
    }

    private float bfloat16ToFloat(short bfloat16) {
        return Float.intBitsToFloat(bfloat16 << 16);
    }
    @Override
    public boolean isImmutable() {
    	return true;
    }
    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
    	if(FloatTensor.USE_CUDA) 
    		return cudaDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
    	if (FloatTensor.USE_VECTOR_API)
    		return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
    	return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
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
		memorySegment = getArena().allocate(bs, 1);
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
	    
	
}

