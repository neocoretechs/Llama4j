package com.llama4j;

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

final class F16FloatTensor extends FloatTensor implements Externalizable, Comparable {
	private static final long serialVersionUID = -1L;
    int size;
    transient MemorySegment memorySegment;

    public F16FloatTensor() {}
    
    public F16FloatTensor(int size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override
    int size() {
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
        return GGMLType.F16;
    }

    @Override
    public float getFloat(int index) {
        assert 0 <= index && index < size;
        return Float.float16ToFloat(readShort(memorySegment, index * GGMLType.FLOAT16_BYTES));
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        } else {
            return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    private static float vectorDot(F16FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        assert S_SPECIES_HALF.length() == F_SPECIES.length();
        FloatVector val = FloatVector.zero(F_SPECIES);
        int upperBound = F_SPECIES.loopBound(size);
        for (int i = 0; i < upperBound; i += F_SPECIES.length()) {
            FloatVector thatVector = that.getFloatVector(F_SPECIES, thatOffset + i);
            ShortVector bits16 = ShortVector.fromMemorySegment(S_SPECIES_HALF, thiz.memorySegment, (thisOffset + i) * (long) GGMLType.FLOAT16_BYTES, ByteOrder.LITTLE_ENDIAN);

            var bits32 = bits16.castShape(I_SPECIES, 0).reinterpretAsInts(); // (int) bits16
            // Does not support infinities nor NaNs, preserves sign, emulate DAZ (denormals-are-zero).
            // Expects well-formed float16 values only (e.g. model weights).
            // Fast Float16 to Float32 Conversion:
            //
            // ┌─[15]─┬─[14]───···───[10]─┬─[9]────····────[0]─┐
            // │ Sign │ Exponent (5 bits) │ Mantissa (10 bits) │ Float16 Layout (16 bits)
            // └──────┴───────────────────┴────────────────────┘
            //    │             │                    │
            //    ▼             ▼                    ▼
            // ┌─[31]─┬─[30]───···───[23]─┬─[22]────···────[0]─┐
            // │ Sign │ Exponent (8 bits) │ Mantissa (23 bits) │ Float32 Layout (32 bits)
            // └──────┴───────────────────┴────────────────────┘
            //
            // Shifts and adjustments:
            // - Sign:       float16[15] -> float32[31] (shift 16 bits up)
            // - Exponent:   float16[10-14] -> float32[23-30] (+ bias adjustment)
            // - Mantissa:   float16[0-9] -> float32[13-22] (shift 13 bits up)
            //
            // exp = bits32 & 0x7C00
            // zeroExponentMask = exp == 0 ? 0 : ~0
            var zeroExponentMask = bits32.and(0x7C00).neg().lanewise(VectorOperators.ASHR, 31); // = (-exp) >> 31
            bits32 = bits32.and(0x8000).lanewise(VectorOperators.LSHL, 16) // sign
                    .or(
                            // exponent and mantissa combined
                            bits32.and(0x7FFF).add(0x1C000).lanewise(VectorOperators.LSHL, 13)
                                    .and(zeroExponentMask) // -0, +0 and DAZ (denormals-are-zero)

                    );

            FloatVector thizVector = bits32.reinterpretAsFloats(); // Float.intBitsToFloat(vi)
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
		memorySegment = Arena.ofAuto().allocate(bs, 1);
		for(int i = 0; i < bs; i++)
			memorySegment.set(ValueLayout.JAVA_BYTE, i, (byte)(in.read() & 0xFF));
	}

	@Override
	public int compareTo(Object o) {
		for(int i = 0; i < memorySegment.byteSize(); i++) {
			byte b;
			if(i >= ((F16FloatTensor)o).memorySegment.byteSize())
				return 1;
			else
				b = ((F16FloatTensor)o).memorySegment.get(ValueLayout.JAVA_BYTE, i);
			if(memorySegment.get(ValueLayout.JAVA_BYTE,i) > b)
				return 1;
			if(memorySegment.get(ValueLayout.JAVA_BYTE,i) < b)
				return -1;
		}
		return 0;
	}

	   @Override
	    public FloatSliceView sliceView(int offset, int length) {
	        return new F16SliceView(memorySegment, offset, length);
	    }

	    @Override
	    public float[] exportSlice(float[] dst, int dstOffset, int offset, int length) {
	        final int bytesPerElem = GGMLType.FLOAT16_BYTES; // 2
	        int i = 0;

	        if (FloatTensor.USE_VECTOR_API) {
	            // Short lanes == float lanes (as in your vectorDot)
	            assert S_SPECIES_HALF.length() == F_SPECIES.length();
	            final int upper = F_SPECIES.loopBound(length);
	            long baseBytes = (long) offset * bytesPerElem;

	            for (; i < upper; i += F_SPECIES.length(), baseBytes += (long) F_SPECIES.length() * bytesPerElem) {
	                ShortVector bits16 = ShortVector.fromMemorySegment(
	                        S_SPECIES_HALF, memorySegment, baseBytes, ByteOrder.LITTLE_ENDIAN);

	                // Fast f16→f32 widening with DAZ (denormals-are-zero), no NaN/Inf handling
	                IntVector bits32 = bits16
	                        .castShape(I_SPECIES, 0)
	                        .reinterpretAsInts();

	                // zeroExponentMask = (exp == 0) ? 0 : ~0
	                IntVector zeroExponentMask = bits32.and(0x7C00).neg().lanewise(VectorOperators.ASHR, 31);

	                bits32 = bits32.and(0x8000).lanewise(VectorOperators.LSHL, 16) // sign << 16
	                        .or(
	                            bits32.and(0x7FFF)                 // strip sign
	                                  .add(0x1C000)               // bias/mantissa adjust
	                                  .lanewise(VectorOperators.LSHL, 13)
	                                  .and(zeroExponentMask)      // DAZ + preserve zeros
	                        );

	                FloatVector fv = bits32.reinterpretAsFloats();
	                fv.intoArray(dst, dstOffset + i);
	            }
	        }

	        // Tail: correctness over speed
	        for (; i < length; i++) {
	            dst[dstOffset + i] = Float.float16ToFloat(
	                    readShort(memorySegment, (long) (offset + i) * bytesPerElem));
	        }
	        return dst;
	    }
	    private static short floatToF16(float f) {
	        return Float.floatToFloat16(f);
	    }

	    @Override
	    public FloatTensor fillInPlace(int thisOffset, int size, float value) {
	        short h = floatToF16(value);
	        long base = (long) thisOffset * GGMLType.FLOAT16_BYTES;
	        for (int i = 0; i < size; i++) {
	            long addr = base + (long) i * GGMLType.FLOAT16_BYTES;
	            memorySegment.set(ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), addr, h);
	        }
	        return this;
	    }
	 // Read-only view for CPU paths
	    final class F16SliceView implements FloatSliceView {
	        final MemorySegment seg; 
	        final int base; 
	        final int len;
	        F16SliceView(MemorySegment s, int base, int len) { 
	        	this.seg = s; 
	        	this.base = base; 
	        	this.len = len; 
	        }
	        public int length(){ return len; }
	        public float get(int i){
	            return Float.float16ToFloat(readShort(seg, (long) (base + i) * GGMLType.FLOAT16_BYTES));
	        }
	    }
}
