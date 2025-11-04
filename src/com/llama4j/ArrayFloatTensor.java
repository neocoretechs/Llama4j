package com.llama4j;

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.util.Arrays;

import com.neocoretechs.cublas.DeviceBuffer;
import com.neocoretechs.cublas.Gemm;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

final class ArrayFloatTensor extends FloatTensor implements Externalizable, Comparable {
	private transient DeviceBuffer device;      // device residency
	MemorySegment memorySegment;
	public static boolean DEBUG = false;
    float[] values;
    
    public ArrayFloatTensor() {
    	Arena arena = Arena.ofShared();
        // allocate off-heap space for headSize floats
        memorySegment = arena.allocate(ValueLayout.JAVA_FLOAT, values.length);
        // bulk copy from heap arrays if you have them
        MemorySegment.copy(values, 0, memorySegment, ValueLayout.JAVA_FLOAT, 0, values.length);
    	//ByteBuffer bb = ByteBuffer.allocateDirect(values.length * Float.BYTES);
    	//bb.asFloatBuffer().put(values);
        //this.device = new DeviceBuffer(bb, GGMLType.F32.getBlockSize(), GGMLType.F32.getTypeSize(), GGMLType.FLOAT16_BYTES, DeviceBuffer.GGUFQ.F32.ordinal());
    }
    
    ArrayFloatTensor(float[] values) {
        this.values = values;
       	Arena arena = Arena.ofShared();
        // allocate off-heap space for headSize floats
        memorySegment = arena.allocate(ValueLayout.JAVA_FLOAT, values.length);
        // bulk copy from heap arrays if you have them
        MemorySegment.copy(values, 0, memorySegment, ValueLayout.JAVA_FLOAT, 0, values.length);
       	//memorySegment = MemorySegment.ofArray(values);
    	//ByteBuffer bb = ByteBuffer.allocateDirect(values.length * Float.BYTES);
    	//bb.asFloatBuffer().put(values);
        //this.device = new DeviceBuffer(bb, GGMLType.F32.getBlockSize(), GGMLType.F32.getTypeSize(), GGMLType.FLOAT16_BYTES, DeviceBuffer.GGUFQ.F32.ordinal());
    }

    public static FloatTensor allocate(int... dims) {
        int numberOfElements = FloatTensor.numberOfElements(dims);
        return new ArrayFloatTensor(new float[numberOfElements]);
    }

    @Override
    public int size() {
        return values.length;
    }

    @Override
    public float getFloat(int index) {
        return values[index];
    }

    @Override
    public void setFloat(int index, float value) {
        values[index] = value;
    }

    @Override
    public GGMLType type() {
        return GGMLType.F32;
    }

    @Override
    public FloatTensor fillInPlace(int thisOffset, int size, float value) {
        Arrays.fill(values, thisOffset, thisOffset + size, value);
        return this;
    }

    @Override
    public FloatVector getFloatVector(VectorSpecies<Float> species, int index) {
        if (!USE_VECTOR_API) {
            throw new UnsupportedOperationException();
        }
        return FloatVector.fromArray(species, values, index);
    }

    @Override
    public long devicePtr() { return device.devicePtr; }
    
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
	public DeviceBuffer getDevice() {
		return device;
	}
    
	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		out.writeInt(values.length);
		for(float v: values)
			out.writeFloat(v);	
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
		int vsize = in.readInt();
		values = new float[vsize];
		for(int i = 0; i < vsize; i++)
			values[i]= in.readFloat();
	}

	@Override
	public int compareTo(Object o) {
		return Arrays.compare(values,((ArrayFloatTensor)o).values);
	}
	@Override
    public FloatSliceView sliceView(int offset, int length) {
    	// Zero-copy if you store (data, baseOffset, length) in the view
    	return new ArrayFloatSliceView(values, offset, length);
    }
    @Override
    public float[] exportSlice(float[] dst, int dstOffset, int offset, int length) {
    	System.arraycopy(values, offset, dst, dstOffset, length);
       	if(DEBUG)
    		System.out.println(this.getClass().getName()+".exportSlice dst="+(dst == null ? " arrayCopy dst length="+length+" FAIL, null!": dst.length));
        return dst;
    }
    
    final class ArrayFloatSliceView implements FloatSliceView {
    	final float[] data; 
    	final int base; 
    	final int len;
        ArrayFloatSliceView(float[] data, int base, int len) { 
        	this.data=data; 
        	this.base=base; 
        	this.len=len;
        }
        public int length() { return len; }
        public float get(int i) { return data[base + i]; }
    }

	@Override
	public MemorySegment asSlice(long offSet1, long offSet2) {
		return memorySegment.asSlice(offSet1, offSet2);
	}
}


