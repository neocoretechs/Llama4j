package com.llama4j;

import java.util.concurrent.ConcurrentHashMap;

public final class DeviceManager {
	public static enum TensorState {
		ON_DEVICE,
		ON_HOST
	}
	static class DeviceStatus {
		TensorState state;
		DeviceStatus(TensorState state) {
			this.state = state;
		}
	}
	private static ConcurrentHashMap<Long, DeviceStatus> deviceMap = new ConcurrentHashMap<>();
	public static void offer(FloatTensor t, String id, boolean reupload) {
		if(t.devicePtrOr0() == 0L) {
			deviceMap.put(FloatTensor.allocDevice(t.totalBytes()), new DeviceStatus(TensorState.ON_HOST));
		}
		DeviceStatus status = deviceMap.get(t.devicePtrOr0());
		if(status == null) { // allocated outside device manager, deal with it
			deviceMap.put(t.devicePtrOr0(), new DeviceStatus(TensorState.ON_HOST));
			System.out.println("WARNING - Tensor "+id+" allocated outside device manager.");
		} else {
			if(status.state != TensorState.ON_DEVICE) {
				if(t.isUploaded())
					System.out.println("WARNING - Tensor "+id+" uploaded outside device manager.");
				t.copyHostToDevice(id);
				status.state = TensorState.ON_DEVICE;
				deviceMap.put(t.devicePtrOr0(), status);
			} else {
				if(reupload)
					t.copyHostToDevice(id);
			}
		}
	}

	/**
	 * called from FlotTensor.freeDevice()
	 * @param devicePtr
	 */
	public static void reclaim(long devicePtr) {
		deviceMap.remove(devicePtr);
	}
	/**
	 * Copy data from device to tensor
	 * @param t
	 */
	public static void reclaim(FloatTensor t, String id) {
		if(t.devicePtrOr0() == 0L) {
			throw new RuntimeException("Tensor "+id+" was already freed. Cannot download from device.");
		}
		DeviceStatus status = deviceMap.get(t.devicePtrOr0());
		if(status == null) { // allocated outside device manager, deal with it
			deviceMap.put(t.devicePtrOr0(), new DeviceStatus(TensorState.ON_HOST));
			System.out.println("WARNING - Tensor "+id+" was previously unknown to device manager.");
		} else {
			if(status.state != TensorState.ON_DEVICE) { // not on device for device manager, but uploaded to tensor?
				if(t.isUploaded()) {
					System.out.println("WARNING - Tensor "+id+" uploaded outside device manager.");
					t.copyDeviceToHost(id);
				}
			} else { // device, download
				t.copyDeviceToHost(id);
			}
		}
	}
	
	static FloatTensor softmax(FloatTensor thiz, int thisOffset, int size, int columns) {
		try {
			return (FloatTensor) Llama3.launchSoftmaxInplace.invokeExact(thiz.devicePtrOr0(), size, columns, thisOffset);
		} catch (Throwable e) {
			throw new RuntimeException(e);
		}
	}
	
	//launch_rmsnorm_fp32_rowmajor(uint8_t* qA, int indexA, int formatA, int blockSizeA, int typeSizeA, int headerBytesA,
    //uint8_t* qB, int indexB, int formatB, int blockSizeB, int typeSizeB, int headerBytesB,
    //float* out, int size, float eps) 
    static void rmsnormGpu(FloatTensor out, FloatTensor x, FloatTensor weight, int size, float eps) {
        try {
			Llama3.launchRmsnorm.invokeExact(x.devicePtrOr0(), 0, x.getFormatType(), x.type().getBlockSize(), x.type().getTypeSize(), x.getHeadSize(),
					weight.devicePtrOr0(), 0, weight.getFormatType(), weight.type().getBlockSize(), weight.type().getTypeSize(), weight.getHeadSize(),
					out.devicePtrOr0(), size, eps);
		} catch (Throwable e) {
			throw new RuntimeException(e);
		}
    }
    
    static void weightedSum(FloatTensor att, FloatTensor xb, FloatTensor vCache, int h, int headSize, int attOffset, int xbOffset, int kvDim, int kvMul, int position, int token) {
    	//void launch_weighted_sum(uint8_t* Att, uint8_t* xb, uint8_t* vCache, int h, int headSize, 
		// int attOffset, int xbOffset, int vcOffset, int kvDim, int kvMul, int position, int token, int size) 
    	try {
    		Llama3.launchAV.invokeExact(att.devicePtrOr0(), xb.devicePtrOr0(), vCache.devicePtrOr0(), 
    				h, headSize, attOffset, xbOffset, kvDim, kvMul, position, token);
    	} catch (Throwable e) {
    		throw new RuntimeException(e);
    	}
    }
    
    static void qkScores(FloatTensor q, int qOffset, FloatTensor keyCache,  
	    FloatTensor Att, int attOffset, int position, int token, int h, int headSize, int kvDim, int kvMul ) {
    	try {
			Llama3.launchQK.invokeExact(q.devicePtrOr0(), qOffset, q.getFormatType(), q.type().getBlockSize(), q.type().getTypeSize(), q.getHeadSize(),
					keyCache.devicePtrOr0(), keyCache.getFormatType(), keyCache.type().getBlockSize(), keyCache.type().getTypeSize(), keyCache.getHeadSize(),
					Att.devicePtrOr0(), attOffset, position, token, h, headSize, kvDim, kvMul);
		} catch (Throwable e) {
			throw new RuntimeException(e);
		}
    }
    
    static float sdot(FloatTensor thiz, int thisOffset, FloatTensor that, int thatOffset, int size) {
    	float result = 0.0f;
		try {
			result = (float) Llama3.sdotSliceDeviceHandle.invokeExact(
					thiz.devicePtrOr0(), thisOffset, thiz.getFormatType(), thiz.type().getBlockSize(), thiz.type().getTypeSize(), thiz.getHeadSize(),
					that.devicePtrOr0(), thatOffset, that.getFormatType(), that.type().getBlockSize(), that.type().getTypeSize(), that.getHeadSize(),
					size);
		} catch (Throwable e) {
			throw new RuntimeException(e);
		}
		return result;
    }
    
    static void matmul(FloatTensor thiz, FloatTensor that, FloatTensor out, int dim0, int dim1) {
		try {
			Llama3.launchMatmul.invokeExact(thiz.devicePtrOr0(), 0, thiz.getFormatType(), thiz.type().getBlockSize(), thiz.type().getTypeSize(), thiz.getHeadSize(),
					that.devicePtrOr0(), 0, that.getFormatType(), that.type().getBlockSize(), that.type().getTypeSize(), that.getHeadSize(), 
					out.devicePtrOr0(), dim0, dim1);
		} catch (Throwable e) {
			throw new RuntimeException(e);
		}
    }
    protected static void printParallelMatmul() {
        /*System.out.println("Parallel matmul print start:");
        Parallel.parallelForLong(0, (long) nTokens * (long) config.numberOfHeads, ht -> {
            final int token = (int) (ht / config.numberOfHeads);
            final int h     = (int) (ht % config.numberOfHeads);
            // Offsets for this head
            final int qOffset   = h * headSize;
            final int attOffset = h * config.contextLength;
            // Time horizon for this token (inclusive of current position)
            final int T = position + token + 1;
            System.out.println(Thread.currentThread().getName()+"| (T, token, h, qOffset, attOffset) T="+T+" token="+token+" h="+h+" qOffset="+qOffset+" attOffset="+attOffset);
            for (int t = 0; t < T; t++) {
                final int keyCacheOffset = t * kvDim + (h / kvMul) * headSize;
                //System.out.println(Thread.currentThread().getName()+"|t loop (curLayer, keyCacheOffset, headSize) - float[] kVec = state.keyCache["+curLayer+"].exportSlicePooled(Llama3.poolHead,"+ keyCacheOffset+","+ headSize+")");
            }
            // 4) Weighted sum over values → xb
            final int xbOffset = h * headSize;
            for (int t = 0; t < T; t++) {
                final int vOffset = t * kvDim + (h / kvMul) * headSize;
                //System.out.println(Thread.currentThread().getName()+"|t loop (token, attOffset, t) float a = state.att["+token+"].getFloat("+(attOffset + t)+")");
                //System.out.println(Thread.currentThread().getName()+"|t loop (token xbOffset, curLayer, vOffset, headSize) state.xb["+token+"].saxpyInPlace("+xbOffset+",state.valueCache["+curLayer+"],("+vOffset+","+ headSize+", a)");
            }
        });
        System.out.println("Parallel matmul print end");*/		
    }

}
