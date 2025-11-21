package com.llama4j;

import java.util.Iterator;
import java.util.concurrent.ConcurrentHashMap;

public final class DeviceManager {
	public static enum TensorState {
		ON_DEVICE,
		ON_HOST
	}
	static class DeviceStatus {
		TensorState state;
		String id;
		long uploads = 0L;
		long downloads = 0L;
		long uploadBytes = 0L;
		long downloadBytes = 0L;
		DeviceStatus(TensorState state, String id) {
			this.state = state;
			this.id = id;
		}
		public String toString() {
			return String.format("Tensor %s uploads=%d downloads=%d upload bytes = %d download bytes = %d", id, uploads, downloads, uploadBytes, downloadBytes);
		}
	}
	
	private static ConcurrentHashMap<Long, DeviceStatus> deviceMap = new ConcurrentHashMap<>();
	
	public static synchronized void offer(FloatTensor t, String id, boolean reupload) {
		if(t.devicePtrOr0() == 0L) {
			t.allocDevice();
			deviceMap.put(t.devicePtrOr0(), new DeviceStatus(TensorState.ON_HOST, id));
		}
		DeviceStatus status = deviceMap.get(t.devicePtrOr0());
		if(status == null) { // allocated outside device manager, deal with it
			deviceMap.put(t.devicePtrOr0(), new DeviceStatus(TensorState.ON_HOST, id));
			System.out.println("WARNING - Tensor "+id+" allocated outside device manager.");
		} else {
			if(status.state != TensorState.ON_DEVICE) {
				if(t.isUploaded())
					System.out.println("WARNING - Tensor "+id+" uploaded outside device manager.");
				t.copyHostToDevice(id);
				status.state = TensorState.ON_DEVICE;
				status.uploadBytes += t.totalBytes();
				++status.uploads;
			} else {
				if(reupload) {
					t.copyHostToDevice(id);
					status.uploadBytes += t.totalBytes();
					++status.uploads;
				}
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
	public static synchronized void reclaim(FloatTensor t, String id) {
		if(t.isImmutable()) {
			throw new RuntimeException("Attempt to reclaim immutable tensor "+id);
		}
		if(t.devicePtrOr0() == 0L) {
			throw new RuntimeException("Tensor "+id+" was already freed. Cannot download from device.");
		}
		DeviceStatus status = deviceMap.get(t.devicePtrOr0());
		if(status == null) { // allocated outside device manager, deal with it
			deviceMap.put(t.devicePtrOr0(), new DeviceStatus(TensorState.ON_HOST, id));
			System.out.println("WARNING - Tensor "+id+" was previously unknown to device manager.");
		} else {
			if(status.state != TensorState.ON_DEVICE) { // not on device for device manager, but uploaded to tensor?
				if(t.isUploaded()) {
					status.state = TensorState.ON_DEVICE;
					System.out.println("WARNING - Tensor "+id+" uploaded outside device manager.");
					t.copyDeviceToHost(id);
					status.downloadBytes += t.totalBytes();
					++status.downloads;
				}
			} else { // its on device, download
				t.copyDeviceToHost(id);
				status.downloadBytes += t.totalBytes();
				++status.downloads;
			}
		}
	}
	
	public static synchronized void reclaimTest(FloatTensor t, String id) {
		if(t.isImmutable())
			return;
		if(t.devicePtrOr0() == 0L) {
			//throw new RuntimeException("Tensor "+id+" was already freed. Cannot download from device.");
			return;
		}
		DeviceStatus status = deviceMap.get(t.devicePtrOr0());
		if(status == null) { // allocated outside device manager, deal with it
			deviceMap.put(t.devicePtrOr0(), new DeviceStatus(TensorState.ON_HOST, id));
			//System.out.println("WARNING - Tensor "+id+" was previously unknown to device manager.");
		} else {
			if(status.state != TensorState.ON_DEVICE) { // not on device for device manager, but uploaded to tensor?
				if(t.isUploaded()) {
					//System.out.println("WARNING - Tensor "+id+" uploaded outside device manager.");
					status.state = TensorState.ON_DEVICE;
					t.copyDeviceToHost(id);
					status.downloadBytes += t.totalBytes();
					++status.downloads;
				}
			} else { // device, download
				t.copyDeviceToHost(id);
				status.downloadBytes += t.totalBytes();
				++status.downloads;
			}
		}
	}
	public static void report() {
	    for (DeviceStatus ds : deviceMap.values()) {
	        System.out.println(ds);
	    }
	}
	public static void reset() {
	    for (DeviceStatus ds : deviceMap.values()) {
	        ds.uploads = ds.downloads = 0L;
	        ds.uploadBytes = ds.downloadBytes = 0L;
	    }
	}
	
	static FloatTensor softmax(FloatTensor thiz, int thisOffset, int size, int columns) {
		try {
			offer(thiz, "softmax", false);
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
        	offer(x, "rmsnorm x", false);
        	offer(weight, "rmsnorm weight", false);
        	offer(out, "rmsnorm out", false);
			Llama3.launchRmsnorm.invokeExact(x.devicePtrOr0(), 0, x.getFormatType(), x.type().getBlockSize(), x.type().getTypeSize(), x.getHeadSize(),
					weight.devicePtrOr0(), 0, weight.getFormatType(), weight.type().getBlockSize(), weight.type().getTypeSize(), weight.getHeadSize(),
					out.devicePtrOr0(), size, eps);
		} catch (Throwable e) {
			throw new RuntimeException(e);
		}
    }
    
    static void rmsnormCpu(FloatTensor out, FloatTensor x, FloatTensor weight, int size, float eps) {
    	// calculate sum of squares
     	reclaimTest(x, "rmsnorm x");
    	reclaimTest(weight, "rmsnorm weight");
    	reclaimTest(out, "rmsnorm out");
    	float ss = x.reduce(0, size, 0f, (acc, xi) -> acc + xi * xi);
    	ss /= size;
    	ss += eps;
    	ss = (float) (1.0 / Math.sqrt(ss));
    	// normalize and scale
    	final float finalss = ss; // for the lambda
    	out.mapWithIndexInPlace(0, size, (value, index) -> weight.getFloat(index) * (finalss * x.getFloat(index)));
      	offer(x, "rmsnorm x", true); // re-upload
    	offer(weight, "rmsnorm weight", true);
    	offer(out, "rmsnorm out", true);
    }
    
    static void weightedSum(FloatTensor att, FloatTensor xb, FloatTensor vCache, int h, int headSize, int attOffset, int xbOffset, int kvDim, int kvMul, int position, int token) {
    	//void launch_weighted_sum(uint8_t* Att, uint8_t* xb, uint8_t* vCache, int h, int headSize, 
		// int attOffset, int xbOffset, int vcOffset, int kvDim, int kvMul, int position, int token, int size) 
    	try {
    		offer(att, "weightedSum att", false);
    		offer(xb, "weightedSum xb", false);
    		offer(vCache, "weightedSum vCache", false);
    		Llama3.launchAV.invokeExact(att.devicePtrOr0(), xb.devicePtrOr0(), vCache.devicePtrOr0(), 
    				h, headSize, attOffset, xbOffset, kvDim, kvMul, position, token);
    	} catch (Throwable e) {
    		throw new RuntimeException(e);
    	}
    }
    
    static void qkScores(FloatTensor q, int qOffset, FloatTensor keyCache,  
	    FloatTensor Att, int attOffset, int position, int token, int h, int headSize, int kvDim, int kvMul ) {
    	try {
    		offer(q, "qkScore q", false);
    		offer(keyCache, "qkScore keyCache", false);
    		offer(Att, "qkScore Att", false);
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
			offer(thiz, "sdot thiz", false);
			offer(that, "sdot that", false);
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
			offer(thiz, "matmul this", false);
			offer(that, "matmul that", false);
			offer(out, "matmul out", false);
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
