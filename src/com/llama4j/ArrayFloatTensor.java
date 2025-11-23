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

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
/**
 * ArrayFloatTensor maintaining array of floats under CPU inference and backed by a MemorySegment under GPU inference.
 * Controlled by FloatTensor.USE_CUDA flag.
 */
final class ArrayFloatTensor extends FloatTensor implements Externalizable, Comparable {
	public static boolean DEBUG = false;
    private float[] values;
	MemorySegment memorySegment;

	public ArrayFloatTensor() {
	}

	ArrayFloatTensor(float[] values) {
		if(FloatTensor.USE_CUDA) {
			// allocate off-heap space for headSize floats
			memorySegment = getArena().allocate(ValueLayout.JAVA_FLOAT, values.length);
			// bulk copy from heap arrays if you have them
			MemorySegment.copy(values, 0, memorySegment, ValueLayout.JAVA_FLOAT, 0, values.length);
		} else {
			this.values = values;
		}
	}

	public static FloatTensor allocate(int... dims) {
		int numberOfElements = FloatTensor.numberOfElements(dims);
		return new ArrayFloatTensor(new float[numberOfElements]);
	}

	@Override
	public int size() {
		if(FloatTensor.USE_CUDA) 
			return (int) (memorySegment.byteSize() / Float.BYTES);
		return values.length;
	}

	@Override
	public float getFloat(int index) {
		if(FloatTensor.USE_CUDA)
			return memorySegment.getAtIndex(ValueLayout.JAVA_FLOAT, index);
		return values[index];
	}

	@Override
	public void setFloat(int index, float value) {
		if(FloatTensor.USE_CUDA)
			memorySegment.setAtIndex(ValueLayout.JAVA_FLOAT, index, value);
		else
			values[index] = value;
	}

	@Override
	public GGMLType type() {
		return GGMLType.F32;
	}
	@Override
	int getHeadSize() {
		return GGMLType.FLOAT16_BYTES; // even though its not used
	}
	@Override
	int getFormatType() {
		return 5;
	}
	protected long totalBytes() { 
		return size() * (long) Float.BYTES; 
	}
	@Override
	public boolean isImmutable() {
		return false;
	}   
	@Override
	public FloatTensor fillInPlace(int thisOffset, int size, float value) {
		if(FloatTensor.USE_CUDA) {
			for(int index = thisOffset; index < thisOffset+size; index++)
				memorySegment.setAtIndex(ValueLayout.JAVA_FLOAT, index, value);
		} else {
			Arrays.fill(values, thisOffset, thisOffset + size, value);
		}
		return this;
	}

	@Override
	public FloatVector getFloatVector(VectorSpecies<Float> species, int index) {
		if (!USE_VECTOR_API) {
			throw new UnsupportedOperationException();
		}
		if(FloatTensor.USE_CUDA)
			return FloatVector.fromMemorySegment(species, memorySegment, (long) index * Float.BYTES, ByteOrder.nativeOrder());
		return FloatVector.fromArray(species, values, index);
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
	public void writeExternal(ObjectOutput out) throws IOException {
		out.writeInt(size());
		for(int i = 0; i < size(); i++)
			out.writeFloat(getFloat(i));
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
		int vsize = in.readInt();
		if(FloatTensor.USE_CUDA) {
			// allocate off-heap space for headSize floats
			memorySegment = getArena().allocate(ValueLayout.JAVA_FLOAT, vsize);
			for(int i = 0; i < vsize; i++)
				setFloat(i, in.readFloat());
		} else {
			values = new float[vsize];
			for(int i = 0; i < vsize; i++)
				values[i]= in.readFloat();
		}
	}

	@Override
	public int compareTo(Object o) {
		if(FloatTensor.USE_CUDA)
			return Arrays.compare(memorySegment.toArray(ValueLayout.JAVA_FLOAT),((ArrayFloatTensor)o).getSegment().toArray(ValueLayout.JAVA_FLOAT));
		return Arrays.compare(values,((ArrayFloatTensor)o).values);
	}
	
	
	@Override
	public String toString() {
    	if(FloatTensor.USE_CUDA)
    		return getSegment().toString();//Arrays.toString(getSegment().toArray(ValueLayout.JAVA_FLOAT));
    	return Arrays.toString(values);
	}
}