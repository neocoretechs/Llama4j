package com.llama4j.ffi;

import java.io.File;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.atomic.AtomicReference;

import com.llama4j.FloatTensor;
import com.llama4j.Llama3;

public final class NativeLoader {
    private static volatile boolean loaded = false;
    private NativeLoader() {}
    private enum LibraryState {
		NOT_LOADED,
		LOADING,
		LOADED
	}
	private static final AtomicReference<LibraryState> libraryLoaded = new AtomicReference<>(LibraryState.NOT_LOADED);

	static {
		if(FloatTensor.USE_CUDA)
			NativeLoader.loadLibrary(new File(System.getProperty("java.library.path")).list());
	}

	public static void load() {
		if(FloatTensor.USE_CUDA)
			NativeLoader.loadLibrary(new File(System.getProperty("java.library.path")).list());
	}
	
	/**
	 * Tries to load the necessary library files from the given list of
	 * directories.
	 *
	 * @param paths a list of strings where each describes a directory of a library.
	 */
	public static void loadLibrary(final String[] paths) {
		if (libraryLoaded.get() == LibraryState.LOADED) {
			return;
		}
		if(libraryLoaded.compareAndSet(LibraryState.NOT_LOADED,LibraryState.LOADING)) {
			synchronized (NativeLoader.class) {
				//.out.println("Loading from paths list of length:"+paths.size());
				for (final String path : paths) {
					//System.out.println(path);
					if(path.endsWith(".so") || path.endsWith(".dll")) {
						String fname = new File(path).getName();
						fname = fname.substring(0,fname.indexOf("."));
						System.out.println("Trying load for:"+fname);
						System.loadLibrary(fname);
					}
				}
			}
			libraryLoaded.set(LibraryState.LOADED);
		}
		while (libraryLoaded.get() == LibraryState.LOADING) {
			try {
				System.out.println("Waiting for load, retry..");
				Thread.sleep(10);
			} catch(final InterruptedException e) {}
		}
	}

	public static void loadMethods() {
		if(!FloatTensor.USE_CUDA)
			return;
		Linker linker = Linker.nativeLinker();
		System.out.println("linker:"+linker);
		SymbolLookup lookup = SymbolLookup.loaderLookup();
		System.out.println("Loader:"+lookup);
		Llama3.sdotSliceDeviceHandle = linker.downcallHandle(
				lookup.find("sdotSliceDevice").get(),
				FunctionDescriptor.of(
						ValueLayout.JAVA_FLOAT,   // return float
						ValueLayout.JAVA_LONG,      // device
						ValueLayout.JAVA_LONG,      // device
						ValueLayout.JAVA_LONG,      // offset1
						ValueLayout.JAVA_LONG,      // offset2
						ValueLayout.JAVA_INT      // headSize
						));
		System.out.println("sdotSliceDevice:"+Llama3.sdotSliceDeviceHandle);
		Llama3.sdotSliceQ8DeviceHandle = linker.downcallHandle(
				lookup.find("sdotSliceQ8Device").get(),
				FunctionDescriptor.of(
						ValueLayout.JAVA_FLOAT,   // return float
						ValueLayout.JAVA_LONG,      // device
						ValueLayout.JAVA_LONG,      // device
						ValueLayout.JAVA_LONG,      // offset1
						ValueLayout.JAVA_LONG,      // offset2
						ValueLayout.JAVA_INT,     // headSize
						ValueLayout.JAVA_INT,     // blockSize
						ValueLayout.JAVA_INT,     // typeSize
						ValueLayout.JAVA_INT      // headerBytes
						));
		System.out.println("sdotSliceQ8Device:"+Llama3.sdotSliceQ8DeviceHandle);
		Llama3.sdotSliceQ4DeviceHandle = linker.downcallHandle(
				lookup.find("sdotSliceQ4Device").get(),
				FunctionDescriptor.of(
						ValueLayout.JAVA_FLOAT,   // return float
						ValueLayout.JAVA_LONG,      // device
						ValueLayout.JAVA_LONG,      // device
						ValueLayout.JAVA_INT,     // headSize
						ValueLayout.JAVA_INT,     // blockSize
						ValueLayout.JAVA_INT,     // index
						ValueLayout.JAVA_INT,     // typeSize
						ValueLayout.JAVA_INT      // headerBytes
						));
		System.out.println("sdotSliceQ4Device:"+Llama3.sdotSliceQ4DeviceHandle);
		Llama3.sdotSliceF16DeviceHandle = linker.downcallHandle(
				lookup.find("sdotSliceF16Device").get(),
				FunctionDescriptor.of(
						ValueLayout.JAVA_FLOAT,   // return float
						ValueLayout.JAVA_LONG,      // device
						ValueLayout.JAVA_LONG,      // device
						ValueLayout.JAVA_INT,     // headSize
						ValueLayout.JAVA_INT,     // index
						ValueLayout.JAVA_INT      // typeSize
						));
		System.out.println("sdotSliceF16Device:"+Llama3.sdotSliceF16DeviceHandle);
		//sdotSliceF16(const uint8_t* q, const float* k, int headSize, int blocks, int typeSize) 
		Llama3.sdotSliceBF16DeviceHandle = linker.downcallHandle(
				lookup.find("sdotSliceBF16Device").get(),
				FunctionDescriptor.of(
						ValueLayout.JAVA_FLOAT,   // return float
						ValueLayout.JAVA_LONG,      // device
						ValueLayout.JAVA_LONG,      // device
						ValueLayout.JAVA_INT,     // headSize
						ValueLayout.JAVA_INT,     // index
						ValueLayout.JAVA_INT      // typeSize
						));
		System.out.println("sdotSliceBF16Device:"+Llama3.sdotSliceBF16DeviceHandle);
		Llama3.sdotSliceHandle = linker.downcallHandle(
				lookup.find("sdotSlice").get(),
				FunctionDescriptor.of(
						ValueLayout.JAVA_FLOAT,   // return float
						ValueLayout.ADDRESS,      // const float* q
						ValueLayout.ADDRESS,      // const float* k
						ValueLayout.JAVA_INT      // headSize
						));
		System.out.println("sdotSlice:"+Llama3.sdotSliceHandle);
		//sdotSliceQ8(const uint8_t*, const float*, int, int, int, int, int);
		Llama3.sdotSliceQ8Handle = linker.downcallHandle(
				lookup.find("sdotSliceQ8").get(),
				FunctionDescriptor.of(
						ValueLayout.JAVA_FLOAT,   // return float
						ValueLayout.ADDRESS,      // const float* q
						ValueLayout.ADDRESS,      // const float* k
						ValueLayout.JAVA_INT,     // headSize
						ValueLayout.JAVA_INT,     // blockSize
						ValueLayout.JAVA_INT,     // index
						ValueLayout.JAVA_INT,     // typeSize
						ValueLayout.JAVA_INT      // headerBytes
						));
		System.out.println("sdotSliceQ8:"+Llama3.sdotSliceQ8Handle);
		//sdotSliceQ4(const uint8_t*, const float*, int, int, int, int, int);
		Llama3.sdotSliceQ4Handle = linker.downcallHandle(
				lookup.find("sdotSliceQ4").get(),
				FunctionDescriptor.of(
						ValueLayout.JAVA_FLOAT,   // return float
						ValueLayout.ADDRESS,      // const float* q
						ValueLayout.ADDRESS,      // const float* k
						ValueLayout.JAVA_INT,     // headSize
						ValueLayout.JAVA_INT,     // blockSize
						ValueLayout.JAVA_INT,     // index
						ValueLayout.JAVA_INT,     // typeSize
						ValueLayout.JAVA_INT      // headerBytes
						));
		System.out.println("sdotSliceQ4:"+Llama3.sdotSliceQ4Handle);
		//sdotSliceF16(const uint8_t* q, const float* k, int headSize, int blocks, int typeSize) 
		Llama3.sdotSliceF16Handle = linker.downcallHandle(
				lookup.find("sdotSliceF16").get(),
				FunctionDescriptor.of(
						ValueLayout.JAVA_FLOAT,   // return float
						ValueLayout.ADDRESS,      // const float* q
						ValueLayout.ADDRESS,      // const float* k
						ValueLayout.JAVA_INT,     // headSize
						ValueLayout.JAVA_INT,     // index
						ValueLayout.JAVA_INT      // typeSize
						));
		System.out.println("sdotSliceF16:"+Llama3.sdotSliceF16Handle);
		//sdotSliceF16(const uint8_t* q, const float* k, int headSize, int blocks, int typeSize) 
		Llama3.sdotSliceBF16Handle = linker.downcallHandle(
				lookup.find("sdotSliceBF16").get(),
				FunctionDescriptor.of(
						ValueLayout.JAVA_FLOAT,   // return float
						ValueLayout.ADDRESS,      // const float* q
						ValueLayout.ADDRESS,      // const float* k
						ValueLayout.JAVA_INT,     // headSize
						ValueLayout.JAVA_INT,     // index
						ValueLayout.JAVA_INT      // typeSize
						));
		System.out.println("sdotSliceBF16:"+Llama3.sdotSliceBF16Handle);
		Llama3.cublasGetHandle = linker.downcallHandle(
				lookup.find("cublasHandle").get(),
				FunctionDescriptor.of(
						ValueLayout.JAVA_LONG  // return long handle
						));
		System.out.println("cublasHandle:"+Llama3.cublasGetHandle);
		Llama3.cublasFreeHandle = linker.downcallHandle(
				lookup.find("cublasHandleDestroy").get(),
				FunctionDescriptor.ofVoid(
						// return void
						ValueLayout.JAVA_LONG  // pass long handle
						));
		System.out.println("cublasHandleDestroy:"+Llama3.cublasFreeHandle);
		Llama3.cudaInit = linker.downcallHandle(
				lookup.find("cudaInit").get(),
				FunctionDescriptor.ofVoid());
		Llama3.cudaGetMemInfo = linker.downcallHandle(
				lookup.find("cudaGetMemInfo").get(),
				FunctionDescriptor.ofVoid(
						ValueLayout.ADDRESS,    // size_t* free
						ValueLayout.ADDRESS     // size_t* total
						));
		System.out.println("cudaGetMemInfo:"+Llama3.cudaGetMemInfo);
		// Signature: void launch_rmsnorm_fp32_rowmajor(const float* x, const float* weight, float* out, int size, float eps)
		Llama3.launchRmsnorm = linker.downcallHandle(
				lookup.find("launch_rmsnorm_fp32_rowmajor").get(),
				FunctionDescriptor.ofVoid(
						ValueLayout.ADDRESS, // x
						ValueLayout.ADDRESS, // weight
						ValueLayout.ADDRESS, // out
						ValueLayout.JAVA_INT,   // size
						ValueLayout.JAVA_FLOAT  // eps
						)
				);
		System.out.println("launch_rmsnorm_fp32_rowmajor:"+Llama3.launchRmsnorm);
		//launch_qk_scores_fp32_rowmajor(
		// const float* Q, const uint8_t* K, float* S,
		// int h, int nHeads, int headSize, int contextLength,
		// int kvDim, int kvMul, int tMaxInclusive, int tensorSize, float sqrtHeadSize,
		// int format, int blockSize, int typeSize, int headerBytes)
		Llama3.launchQK = linker.downcallHandle(
				lookup.find("launch_qk_scores_fp32_rowmajor").get(),
				FunctionDescriptor.ofVoid(
						ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG, // Q, K, S
						ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, // h, nHeads, headSize, contextLen
						ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_FLOAT,// kvDim, kvMul, tMaxInclusive, tensorSize, sqrtSize
						ValueLayout.JAVA_INT, // format
						ValueLayout.JAVA_INT ,ValueLayout.JAVA_INT,ValueLayout.JAVA_INT  // blocksize, typesize, headerbytes
						)
				);
		System.out.println("launch_qk_scores_fp32_rowmajor:"+Llama3.launchQK);
		Llama3.launchSoftmax = linker.downcallHandle(
				lookup.find("launch_row_softmax_fp32").get(),
				FunctionDescriptor.ofVoid(
						ValueLayout.ADDRESS, ValueLayout.ADDRESS, // S, A
						ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, // rows, cols
						ValueLayout.JAVA_INT, ValueLayout.JAVA_INT  // ldS, ldA strides
						)
				);
		System.out.println("launch_row_softmax_fp32:"+Llama3.launchSoftmax);
		Llama3.launchSoftmaxInplace = linker.downcallHandle(
				lookup.find("launch_row_softmax_inplace_fp32").get(),
				FunctionDescriptor.ofVoid(
						ValueLayout.JAVA_LONG, // S device address
						ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, // rows, cols
						ValueLayout.JAVA_INT  // offset
						)
				);
		System.out.println("launch_row_softmax_inplace_fp32:"+Llama3.launchSoftmaxInplace);
		Llama3.launchAV = linker.downcallHandle(
			    lookup.find("launch_attention_av_weighted_sum").get(),
			    FunctionDescriptor.ofVoid(
			        ValueLayout.ADDRESS, // attTok
			        ValueLayout.ADDRESS, // vCacheRaw quantized/raw bytes for V: [contextLen*kvTypeSizeTotal]
			        ValueLayout.ADDRESS, // xbTok [nHeads*headSize]
			        //int nHeads, int headSize, int kvDim, int kvMul,int contextLen, int tMax,
			        ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
			        // Quantization params for V. vBlockSize elements per quant block, vTypeSize bytes per block (header + payload), vheaderBytes bytes before payload, format: 1 for Q8, 2 for Q4, 3 for F16, 4 for BF16, 5 for F32
			        ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT
			    )
			);
		System.out.println("launch_attention_av_weighted_sum:"+Llama3.launchAV);
	    Llama3.allocDevicePtr = linker.downcallHandle(
		        lookup.find("allocDevicePtr").get(),
		        FunctionDescriptor.of(ValueLayout.JAVA_LONG, // uint64_t device ptr
		        					ValueLayout.JAVA_LONG) // size 
		    );
		System.out.println("allocDevicePtr:"+Llama3.allocDevicePtr);
	    Llama3.freeDevicePtr = linker.downcallHandle(
		        lookup.find("freeDevicePtr").get(),
		        FunctionDescriptor.ofVoid(ValueLayout.JAVA_LONG) // uint64_t device ptr
		    );
		System.out.println("freeDevicePtr:"+Llama3.freeDevicePtr);
		 // copyHostToDevice
	    Llama3.copyHostToDeviceMH = linker.downcallHandle(
	        lookup.find("copyHostToDevice").get(),
	        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS,   // uint8_t* tensor
	                              ValueLayout.JAVA_LONG, // uint64_t device ptr
	                              ValueLayout.JAVA_LONG) // int bytes
	    );
		System.out.println("copyHostToDevice:"+Llama3.copyHostToDeviceMH);
	    // copyDeviceToHost
	    Llama3.copyDeviceToHostMH = linker.downcallHandle(
	        lookup.find("copyDeviceToHost").get(),
	        FunctionDescriptor.ofVoid(ValueLayout.JAVA_LONG, // uint64_t device pt
	                                  ValueLayout.ADDRESS,   // uint8_t* tensor
	                                  ValueLayout.JAVA_LONG) // size_t bytes
	    );
		System.out.println("copyDeviceToHost:"+Llama3.copyDeviceToHostMH);
	}
}
