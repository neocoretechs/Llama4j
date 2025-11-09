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
	       //this.device = new DeviceBuffer(memorySegment.asByteBuffer(), GGMLType.F32.getBlockSize(), GGMLType.F32.getTypeSize(), GGMLType.FLOAT16_BYTES, DeviceBuffer.GGUFQ.F32.ordinal());
	}
	
	public F32FloatTensor(int size, MemorySegment memorySegment) {
		this.size = size;
		this.memorySegment = memorySegment;
	    //this.device = new DeviceBuffer(memorySegment.asByteBuffer(), GGMLType.F32.getBlockSize(), GGMLType.F32.getTypeSize(), GGMLType.FLOAT16_BYTES, DeviceBuffer.GGUFQ.F32.ordinal());
	}

	@Override
	public int size() {
		return size;
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
	public MemorySegment asSlice(long offSet1, long offSet2) {
		return memorySegment.asSlice(offSet1, offSet2);
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
    public Arena getArena() {
    	return Llama3.autoArena;
    }
	@Override
	public MemorySegment getSegment() {
		return memorySegment;
	}
	
    public float cuBLASdotSlice(int thisOffset, FloatTensor that, int thatOffset, int size) throws Throwable {
        MemorySegment qSeg = this.sliceElements(thisOffset, size);
        MemorySegment kSeg = that.sliceElements(thatOffset, size);
        float result = (float) Llama3.sdotSliceHandle.invokeExact(qSeg, kSeg, size);
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
	public FloatSliceView sliceView(int offset, int length) {
		return new F32SliceView(memorySegment, offset, length);
	}
	
	@Override
	public float[] exportSlice(float[] dst, int dstOffset, int offset, int length) {
		int i = 0;
		if (FloatTensor.USE_VECTOR_API) {
			final int upper = F_SPECIES.loopBound(length);
			long baseBytes = ((long) offset) * Float.BYTES;
			for (; i < upper; i += F_SPECIES.length(), baseBytes += (long) F_SPECIES.length() * Float.BYTES) {
				FloatVector fv = FloatVector.fromMemorySegment(F_SPECIES, memorySegment, baseBytes, ByteOrder.LITTLE_ENDIAN);
				fv.intoArray(dst, dstOffset + i);
			}
		} else
			for(; i < length; i++) {
				dst[dstOffset + i] = readFloat(memorySegment, ((long) (offset + i)) * Float.BYTES);
			}
		return dst;
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
	
	final class F32SliceView implements FloatSliceView {
		final MemorySegment seg; 
		final int base; 
		final int len;
		F32SliceView(MemorySegment s, int base, int len){ 
			seg = s;
			this.base = base; 
			this.len = len; 
		}
		public int length(){ 
			return len; 
		}
	    public float get(int i) {
	    	return FloatTensor.readFloat(seg, ((long) (base + i)) * Float.BYTES);
	    }
	}

}



