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
import java.util.concurrent.TimeUnit;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

final class Q8_0FloatTensor extends FloatTensor implements Externalizable, Comparable {
	private static final long serialVersionUID = -1L;
	
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
    public Arena getArena() {
    	return Llama3.autoArena;
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
    int getHeadSize() {
    	return GGMLType.FLOAT16_BYTES;
    }
    @Override
    int getFormatType() {
    	return 1;
    }
    protected long totalBytes() { 
    	return size(); 
    }
    @Override
    public float getFloat(int index) {
        assert 0 <= index && index < size;
        int blockIndex = index / GGMLType.Q8_0.getBlockSize();
        int withinBlockIndex = index % GGMLType.Q8_0.getBlockSize();
        int blockOffset = blockIndex * GGMLType.Q8_0.getTypeSize();
        byte quant = readByte(memorySegment, blockOffset + GGMLType.FLOAT16_BYTES + withinBlockIndex);
        float scale = Float.float16ToFloat(readShort(memorySegment, blockOffset));
        //System.out.printf("CPU index=%d quant=%d scale=%.6f%n", index, quant,scale);
        return quant * scale;
    }

    public static final ValueLayout.OfShort JAVA_SHORT_LE = ValueLayout.JAVA_SHORT.withOrder(ByteOrder.LITTLE_ENDIAN);

    @Override
    public boolean isImmutable() {
    	return true;
    }
    
    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
    	//if(FloatTensor.USE_CUDA) 
      	//	return cudaDot(this, thisOffset, that, thatOffset, size);
    	if (FloatTensor.USE_VECTOR_API)
    		return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
    	return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
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

}

