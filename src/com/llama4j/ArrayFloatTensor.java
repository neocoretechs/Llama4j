package com.llama4j;

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.Arrays;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

final class ArrayFloatTensor extends FloatTensor implements Externalizable, Comparable {
	public static boolean DEBUG = true;
    float[] values;
    
    public ArrayFloatTensor() {}
    
    ArrayFloatTensor(float[] values) {
        this.values = values;
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
}


