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
import jdk.incubator.vector.VectorSpecies;

final class F32FloatTensor extends FloatTensor implements Externalizable, Comparable {
	private static final long serialVersionUID = -1L;

	int size;
	transient MemorySegment memorySegment;
	
	public F32FloatTensor() {
	}
	
	public F32FloatTensor(int size, MemorySegment memorySegment) {
		this.size = size;
		this.memorySegment = memorySegment;
	}

	@Override
	public int size() {
		return size;
	}
	@Override
	int getHeadSize() {
		return GGMLType.FLOAT16_BYTES; // although not used
	}
	@Override
	int getFormatType() {
		return 5;
	}
	@Override
	public float getFloat(int index) {
		assert 0 <= index && index < size;
		return readFloat(memorySegment, index * 4);
	}

	@Override
	public void setFloat(int index, float value) {
		throw new UnsupportedOperationException("setFloat");	
	}

	@Override
	FloatVector getFloatVector(VectorSpecies<Float> species, int offset) {
		 throw new UnsupportedOperationException("getFloatVector");
	}

	@Override
	GGMLType type() {
		return GGMLType.F32;
	}
	
	
	@Override
	public long getOffsetBytes(long elementOffset) {
		long blockIndex = elementOffset / GGMLType.F32.getBlockSize();
		return blockIndex * GGMLType.F32.getTypeSize();
	}

	@Override
	public long getLengthBytes(long elementCount, long elementOffset) {
		long startBlock = elementOffset / GGMLType.F32.getBlockSize();
		long endBlock   = (elementOffset + elementCount - 1) / GGMLType.F32.getBlockSize();
		long blocks     = (endBlock - startBlock + 1);
		return blocks * GGMLType.F32.getTypeSize();
	}
	@Override
	public boolean isImmutable() {
		return true;
	}
    @Override
    public Arena getArena() {
    	return Llama3.autoArena;
    }
	@Override
	public MemorySegment getSegment() {
		return memorySegment;
	}
	protected long totalBytes() { 
	    return memorySegment.byteSize(); 
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
			if(i >= ((F32FloatTensor)o).memorySegment.byteSize())
				return 1;
			else
				b = ((F32FloatTensor)o).memorySegment.get(ValueLayout.JAVA_BYTE, i);
			if(memorySegment.get(ValueLayout.JAVA_BYTE,i) > b)
				return 1;
			if(memorySegment.get(ValueLayout.JAVA_BYTE,i) < b)
				return -1;
		}
		return 0;
	}
	
	@Override
	public FloatTensor fillInPlace(int thisOffset, int size, float value) {
	    long base = (long) thisOffset * Float.BYTES;
	    for (int i = 0; i < size; i++) {
	        long addr = base + (long) i * Float.BYTES;
	        memorySegment.set(ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.LITTLE_ENDIAN), addr, value);
	    }
	    return this;
	}
	

}



