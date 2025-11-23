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

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * {@link FloatTensor} quantized in the {@link GGMLType#Q4_0} format.
 * <p>
 * This tensor implementation is not compatible with {@link FloatTensor}, but
 * {@link #dot(int, FloatTensor, int, int)} has a vectorized implementation that is used when
 * the second argument implements {@link FloatTensor}.
 */
final class Q4_0FloatTensor extends FloatTensor implements Externalizable, Comparable {
	private static final long serialVersionUID = -1L;
	
	int size;
    transient MemorySegment memorySegment;

    public Q4_0FloatTensor() {
	       //this.device = new DeviceBuffer(memorySegment.asByteBuffer(), GGMLType.Q4_0.getBlockSize(), GGMLType.Q4_0.getTypeSize(), GGMLType.FLOAT16_BYTES, DeviceBuffer.GGUFQ.Q4_0.ordinal());
    }
    
    public Q4_0FloatTensor(int size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
        //this.device = new DeviceBuffer(memorySegment.asByteBuffer(), GGMLType.Q4_0.getBlockSize(), GGMLType.Q4_0.getTypeSize(), GGMLType.FLOAT16_BYTES, DeviceBuffer.GGUFQ.Q4_0.ordinal());
    }

    @Override
    public int size() {
        return size;
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
        return GGMLType.Q4_0;
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
    	return 2;
    }
    @Override
    public Arena getArena() {
    	return Llama3.autoArena;
    }
	
    @Override
    public long getOffsetBytes(long elementOffset) {
        long blockIndex = elementOffset / GGMLType.Q4_0.getBlockSize();
        return blockIndex * GGMLType.Q4_0.getTypeSize();
    }

    @Override
    public long getLengthBytes(long elementCount, long elementOffset) {
        long startBlock = elementOffset / GGMLType.Q4_0.getBlockSize();
        long endBlock   = (elementOffset + elementCount - 1) / GGMLType.Q4_0.getBlockSize();
        long blocks     = (endBlock - startBlock + 1);
        return blocks * GGMLType.Q4_0.getTypeSize();
    }
	@Override
	public MemorySegment getSegment() {
		return memorySegment;
	}
	
    @Override
    public float getFloat(int index) {
        assert 0 <= index && index < size;
        int blockIndex = index / GGMLType.Q4_0.getBlockSize();
        int blockOffset = blockIndex * GGMLType.Q4_0.getTypeSize();
        float scale = Float.float16ToFloat(readShort(memorySegment, blockOffset));
        byte quant;
        int modIndex = index % GGMLType.Q4_0.getBlockSize();
        if (modIndex < GGMLType.Q4_0.getBlockSize() / 2) {
            quant = (byte) (readByte(memorySegment, blockOffset + GGMLType.FLOAT16_BYTES + modIndex) & 0x0F);
        } else {
            quant = (byte) ((readByte(memorySegment, blockOffset + GGMLType.FLOAT16_BYTES + modIndex - GGMLType.Q4_0.getBlockSize() / 2) >>> 4) & 0x0F);
        }
        quant -= 8;
        return quant * scale;
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

    private static float vectorDot(Q4_0FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;

        // Align thisOffset + j to type().getBlockSize().
        assert Integer.bitCount(GGMLType.Q4_0.getBlockSize()) == 1 : "power of 2";
        int alignmentBound = Math.min(size, -thisOffset & (GGMLType.Q4_0.getBlockSize() - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }
        assert (thisOffset + j) % GGMLType.Q4_0.getBlockSize() == 0;

        FloatVector val = FloatVector.zero(F_SPECIES);
        int blockOffset = (thisOffset + j) / GGMLType.Q4_0.getBlockSize() * GGMLType.Q4_0.getTypeSize();
        int upperBound = size / GGMLType.Q4_0.getBlockSize() * GGMLType.Q4_0.getBlockSize();
        for (; j < upperBound; j += GGMLType.Q4_0.getBlockSize(), blockOffset += GGMLType.Q4_0.getTypeSize()) {
            float wScaleValue = Float.float16ToFloat(readShort(thiz.memorySegment, blockOffset));
            var wScale = FloatVector.broadcast(F_SPECIES, wScaleValue);
            var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment, blockOffset + GGMLType.FLOAT16_BYTES, ByteOrder.LITTLE_ENDIAN);
            var loBytes = wBytes.and((byte) 0xF).sub((byte) 8);
            var hiBytes = wBytes.lanewise(VectorOperators.LSHR, 4).sub((byte) 8);
            switch (F_SPECIES.vectorBitSize()) {
                case 512 -> {
                    var sum0 = that.getFloatVector(F_SPECIES, thatOffset + j + 0 * F_SPECIES.length()).mul(loBytes.castShape(F_SPECIES, 0));
                    var sum2 = that.getFloatVector(F_SPECIES, thatOffset + j + 1 * F_SPECIES.length()).mul(hiBytes.castShape(F_SPECIES, 0));
                    val = sum0.add(sum2).fma(wScale, val);
                }
                case 256 -> {
                    var sum0 = that.getFloatVector(F_SPECIES, thatOffset + j + 0 * F_SPECIES.length()).mul(loBytes.castShape(F_SPECIES, 0));
                    var sum1 = that.getFloatVector(F_SPECIES, thatOffset + j + 1 * F_SPECIES.length()).mul(loBytes.castShape(F_SPECIES, 1));
                    var sum2 = that.getFloatVector(F_SPECIES, thatOffset + j + 2 * F_SPECIES.length()).mul(hiBytes.castShape(F_SPECIES, 0));
                    var sum3 = that.getFloatVector(F_SPECIES, thatOffset + j + 3 * F_SPECIES.length()).mul(hiBytes.castShape(F_SPECIES, 1));
                    val = sum0.add(sum1).add(sum2).add(sum3).fma(wScale, val);
                }
                case 128 -> {
                    // This loop cannot be unrolled, why?
                    for (int i = 0; i < 2; ++i) {
                        var tmp = i == 0 ? loBytes : hiBytes;
                        var sum0 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 0) * F_SPECIES.length()).mul(tmp.castShape(F_SPECIES, 0));
                        var sum1 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 1) * F_SPECIES.length()).mul(tmp.castShape(F_SPECIES, 1));
                        var sum2 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 2) * F_SPECIES.length()).mul(tmp.castShape(F_SPECIES, 2));
                        var sum3 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 3) * F_SPECIES.length()).mul(tmp.castShape(F_SPECIES, 3));
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
		memorySegment = getArena().allocate(bs, 1);
		for(int i = 0; i < bs; i++)
			memorySegment.set(ValueLayout.JAVA_BYTE, i, (byte)(in.read() & 0xFF));
	}

	@Override
	public int compareTo(Object o) {
		for(int i = 0; i < memorySegment.byteSize(); i++) {
			byte b;
			if(i >= ((Q4_0FloatTensor)o).memorySegment.byteSize())
				return 1;
			else
				b = ((Q4_0FloatTensor)o).memorySegment.get(ValueLayout.JAVA_BYTE, i);
			if(memorySegment.get(ValueLayout.JAVA_BYTE,i) > b)
				return 1;
			if(memorySegment.get(ValueLayout.JAVA_BYTE,i) < b)
				return -1;
		}
		return 0;
	}
}

