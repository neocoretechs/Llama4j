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
		//System.out.println("linker:"+linker);
		SymbolLookup lookup = SymbolLookup.loaderLookup();
		//System.out.println("Loader:"+lookup);
		//
		//float getFloat(const uint64_t q, int index, int blockSize, int typeSize, int headerBytes) {
		//    const float* d_q = reinterpret_cast<const float*>(q);
		Llama3.getFloatQ8 = linker.downcallHandle(
				lookup.find("getFloatQ8").get(),
				FunctionDescriptor.of(
						ValueLayout.JAVA_FLOAT,   // return float
						ValueLayout.JAVA_LONG,    // device A
						ValueLayout.JAVA_INT ,    // offset A
						ValueLayout.JAVA_INT,     // blockSize A for format
						ValueLayout.JAVA_INT,     // typeSize A for format
						ValueLayout.JAVA_INT      // headerBytes A for format
						));
		System.out.println("getFloatQ8:"+Llama3.getFloatQ8);
		//float getFloat(const uint64_t q, int index)
		Llama3.getFloat = linker.downcallHandle(
				lookup.find("getFloat").get(),
				FunctionDescriptor.of(
						ValueLayout.JAVA_FLOAT,   // return float
						ValueLayout.JAVA_LONG,    // device A
						ValueLayout.JAVA_INT    // offset A
						));
		System.out.println("getFloat:"+Llama3.getFloat);
		//
		Llama3.sdotSliceDeviceHandle = linker.downcallHandle(
				lookup.find("sdotSliceDevice").get(),
				FunctionDescriptor.of(
						//float sdotSliceDevice(const uint8_t* qA, int indexA, int formatA, int blockSizeA, int typeSizeA, int headerBytesA,
						//	    const uint8_t* qB, int indexB, int formatB, int blockSizeB, int typeSizeB, int headerBytesB,
						//	    int N)
						ValueLayout.JAVA_FLOAT,   // return float
						ValueLayout.JAVA_LONG,    // device A
						ValueLayout.JAVA_INT ,    // offset A
						ValueLayout.JAVA_INT,     // format A (1-5 = Q8,Q4,F16,BF16,F32)
						ValueLayout.JAVA_INT,     // blockSize A for format
						ValueLayout.JAVA_INT,     // typeSize A for format
						ValueLayout.JAVA_INT,     // headerBytes A for format
						ValueLayout.JAVA_LONG,    // device B
						ValueLayout.JAVA_INT,     // offset B
						ValueLayout.JAVA_INT,     // format B
						ValueLayout.JAVA_INT,     // blocksize B
						ValueLayout.JAVA_INT,     // typeSize B
						ValueLayout.JAVA_INT,     // headerBytes B
						ValueLayout.JAVA_INT	  // Number of elements in tensor
						));
		System.out.println("sdotSliceDevice:"+Llama3.sdotSliceDeviceHandle);
		//System.out.println("cublasHandle:"+Llama3.cublasGetHandle);
		//Llama3.cublasFreeHandle = linker.downcallHandle(
		//		lookup.find("cublasHandleDestroy").get(),
		//		FunctionDescriptor.ofVoid(
		//				// return void
		//				ValueLayout.JAVA_LONG  // pass long handle
		//				));
		//System.out.println("cublasHandleDestroy:"+Llama3.cublasFreeHandle);
		Llama3.cudaInit = linker.downcallHandle(
				lookup.find("cudaInit").get(),
				FunctionDescriptor.ofVoid());
		Llama3.cudaGetMemInfo = linker.downcallHandle(
				lookup.find("cudaGetMemInfo").get(),
				FunctionDescriptor.ofVoid(
						ValueLayout.ADDRESS,    // size_t* free, writes to memorysegments
						ValueLayout.ADDRESS     // size_t* total
						));
		System.out.println("cudaGetMemInfo:"+Llama3.cudaGetMemInfo);
		//launch_rmsnorm_fp32_rowmajor(uint8_t* qA, int indexA, int formatA, int blockSizeA, int typeSizeA, int headerBytesA,
	    //uint8_t* qB, int indexB, int formatB, int blockSizeB, int typeSizeB, int headerBytesB,
	    //float* out, int size, float eps) {
		Llama3.launchRmsnorm = linker.downcallHandle(
				lookup.find("launch_rmsnorm_fp32_rowmajor").get(),
				FunctionDescriptor.ofVoid(
						ValueLayout.JAVA_LONG, // deviceptr x
						ValueLayout.JAVA_INT, // offset into x
						ValueLayout.JAVA_INT, // format x
						ValueLayout.JAVA_INT, // blocksize x
						ValueLayout.JAVA_INT, // typesiez x
						ValueLayout.JAVA_INT, // headerbytes s
						ValueLayout.JAVA_LONG, // deviceptr weights
						ValueLayout.JAVA_INT, // offset into weights
						ValueLayout.JAVA_INT, // format weights
						ValueLayout.JAVA_INT, // blocksize weights
						ValueLayout.JAVA_INT, // typesize weights
						ValueLayout.JAVA_INT, // headerbytes weights
						ValueLayout.JAVA_LONG, // deviceptr out
						ValueLayout.JAVA_INT,   // size
						ValueLayout.JAVA_FLOAT  // eps
						)
				);
		System.out.println("launch_rmsnorm_fp32_rowmajor:"+Llama3.launchRmsnorm);
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
		//void launch_weighted_sum(uint8_t* Att, uint8_t* xb, uint8_t* vCache, int h, int headSize, 
		// int attOffset, int xbOffset, int vcOffset, int kvDim, int kvMul, int position, int token, int size) 
		Llama3.launchAV = linker.downcallHandle(
			    lookup.find("launch_weighted_sum").get(),
			    FunctionDescriptor.ofVoid(
			        ValueLayout.JAVA_LONG, // Att
			        ValueLayout.JAVA_LONG, // xb
			        ValueLayout.JAVA_LONG, // vCache
			        //int h, int headSize, int attOffset, int xbOffset,  
			        ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
			        ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, //int kvDim, int kvMul
			        ValueLayout.JAVA_INT, ValueLayout.JAVA_INT //int position, int token
			    )
			);
		System.out.println("launch_weighted_sum:"+Llama3.launchAV);
		// void launchMatmul(const uint8_t* qA, int indexA, int formatA, int blockSizeA, int typeSizeA, int headerBytesA,
		//	    const uint8_t* qB, int indexB, int formatB, int blockSizeB, int typeSizeB, int headerBytesB,
		//	    uint8_t* out, int dim0, int dim1) {
		Llama3.launchMatmul = linker.downcallHandle(
			    lookup.find("launch_Matmul").get(),
			    FunctionDescriptor.ofVoid(
			        ValueLayout.JAVA_LONG, // qA
			        //int indexA, formatA, blockSizeA, typeSizeA, headerBytesA
			        ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
			        ValueLayout.JAVA_LONG, // qB
			        //int indexA, formatA, blockSizeA, typeSizeA, headerBytesA
			        ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
			        ValueLayout.JAVA_LONG, // out
			        ValueLayout.JAVA_INT, // dim0
			        ValueLayout.JAVA_INT // dim1
			    )
			);
		System.out.println("launch_Matmul:"+Llama3.launchMatmul);
		//float launch_cpu_scalar_Dot(const uint8_t* d_q, int indexA, int formatA, int blockSizeA, int typeSizeA, int headerBytesA,
		//	    const uint8_t* d_k, int indexB, int formatB, int blockSizeB, int typeSizeB, int headerBytesB, int size)
		Llama3.sdotSimple = linker.downcallHandle(
				lookup.find("launch_cpu_scalar_Dot").get(),
				FunctionDescriptor.of(
						ValueLayout.JAVA_FLOAT,   // return float
						ValueLayout.JAVA_LONG,    // device A
						ValueLayout.JAVA_INT ,    // offset A
						ValueLayout.JAVA_INT,     // format A (1-5 = Q8,Q4,F16,BF16,F32)
						ValueLayout.JAVA_INT,     // blockSize A for format
						ValueLayout.JAVA_INT,     // typeSize A for format
						ValueLayout.JAVA_INT,     // headerBytes A for format
						ValueLayout.JAVA_LONG,    // device B
						ValueLayout.JAVA_INT,     // offset B
						ValueLayout.JAVA_INT,     // format B
						ValueLayout.JAVA_INT,     // blocksize B
						ValueLayout.JAVA_INT,     // typeSize B
						ValueLayout.JAVA_INT,     // headerBytes B
						ValueLayout.JAVA_INT	  // Number of elements in tensor
						));
		System.out.println("launch_cpu_scalar_Dot:"+Llama3.sdotSimple);
		//void launch_qkscores(uint8_t* q, int qOffset, int formatA, int blockSizeA, int typeSizeA, int headerBlockA,
	    //uint8_t* keyCache, int keyCacheOffset, int formatB, int blockSizeB, int typeSizeB, int headerBlockB, 
	    //uint8_t* Att, int attOffset, 
	    //int position, int token, int h, int headSize, int kvDim, int kvMul )
		Llama3.launchQK = linker.downcallHandle(
				lookup.find("launch_qkscores").get(),
				FunctionDescriptor.ofVoid(
						ValueLayout.JAVA_LONG,    // q
						ValueLayout.JAVA_INT ,    // offset q
						ValueLayout.JAVA_INT,     // format q (1-5 = Q8,Q4,F16,BF16,F32)
						ValueLayout.JAVA_INT,     // blockSize q for format
						ValueLayout.JAVA_INT,     // typeSize q for format
						ValueLayout.JAVA_INT,     // headerBytes q for format
						ValueLayout.JAVA_LONG,    // keyCache (keyCacheOffset computed in device kernel)
						ValueLayout.JAVA_INT,     // format 
						ValueLayout.JAVA_INT,     // blocksize 
						ValueLayout.JAVA_INT,     // typeSize 
						ValueLayout.JAVA_INT,     // headerBytes keyCache
						ValueLayout.JAVA_LONG,    // Att
						ValueLayout.JAVA_INT,     // attOffset 
						ValueLayout.JAVA_INT,	  // position
						ValueLayout.JAVA_INT,     // token 
						ValueLayout.JAVA_INT,     // h
						ValueLayout.JAVA_INT,     // headSize
						ValueLayout.JAVA_INT,     // numHeads
						ValueLayout.JAVA_INT,     // contextLength
						ValueLayout.JAVA_INT,     // kvDim
						ValueLayout.JAVA_INT	  // kvMul
						));
		System.out.println("launch_qkscores:"+Llama3.launchQK);
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
	                                  ValueLayout.JAVA_LONG,   // uint8_t* tensor
	                                  ValueLayout.JAVA_LONG) // size_t bytes
	    );
		System.out.println("copyDeviceToHost:"+Llama3.copyDeviceToHostMH);
	}
}
