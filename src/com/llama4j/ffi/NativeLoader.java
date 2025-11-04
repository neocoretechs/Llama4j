package com.llama4j.ffi;

import java.io.File;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.atomic.AtomicReference;

import com.llama4j.Llama3;

public class NativeLoader {
    private static volatile boolean loaded = false;
    private NativeLoader() {}
    private enum LibraryState {
		NOT_LOADED,
		LOADING,
		LOADED
	}
	private static final AtomicReference<LibraryState> libraryLoaded = new AtomicReference<>(LibraryState.NOT_LOADED);

	static {
		NativeLoader.loadLibrary(new File(System.getProperty("java.library.path")).list());
	}

	public static void load() {
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
        Linker linker = Linker.nativeLinker();
        System.out.println("linker:"+linker);
        SymbolLookup lookup = SymbolLookup.loaderLookup();
        System.out.println("Loader:"+lookup);
        Llama3.sdotSliceHandle = linker.downcallHandle(
            lookup.find("sdotSlice").get(),
            FunctionDescriptor.of(
                ValueLayout.JAVA_FLOAT,   // return float
                ValueLayout.JAVA_LONG,	  // cublasHandle
                ValueLayout.ADDRESS,      // const float* q
                ValueLayout.ADDRESS,      // const float* k
                ValueLayout.JAVA_INT      // headSize
            )
        );
        System.out.println("sdotSlice:"+Llama3.sdotSliceHandle);
        //sdotSliceQ8(const uint8_t*, const float*, int, int, int, int, int);
        Llama3.sdotSliceQ8Handle = linker.downcallHandle(
                lookup.find("sdotSliceQ8").get(),
                FunctionDescriptor.of(
                    ValueLayout.JAVA_FLOAT,   // return float
                    ValueLayout.JAVA_LONG,	  // cublasHandle
                    ValueLayout.ADDRESS,      // const float* q
                    ValueLayout.ADDRESS,      // const float* k
                    ValueLayout.JAVA_INT,     // headSize
                    ValueLayout.JAVA_INT,     // blockSize
                    ValueLayout.JAVA_INT,     // blocks
                    ValueLayout.JAVA_INT,     // typeSize
                    ValueLayout.JAVA_INT      // headerBytes
                )
            );
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
                    ValueLayout.JAVA_INT,     // blocks
                    ValueLayout.JAVA_INT,     // typeSize
                    ValueLayout.JAVA_INT      // headerBytes
                )
            );
        System.out.println("sdotSliceQ4:"+Llama3.sdotSliceQ4Handle);
        //sdotSliceF16(const uint8_t* q, const float* k, int headSize, int blocks, int typeSize) 
        Llama3.sdotSliceF16Handle = linker.downcallHandle(
                lookup.find("sdotSliceF16").get(),
                FunctionDescriptor.of(
                    ValueLayout.JAVA_FLOAT,   // return float
                    ValueLayout.ADDRESS,      // const float* q
                    ValueLayout.ADDRESS,      // const float* k
                    ValueLayout.JAVA_INT,     // headSize
                    ValueLayout.JAVA_INT,     // blocks
                    ValueLayout.JAVA_INT      // typeSize
                )
            );
        System.out.println("sdotSliceF16:"+Llama3.sdotSliceF16Handle);
        //sdotSliceF16(const uint8_t* q, const float* k, int headSize, int blocks, int typeSize) 
        Llama3.sdotSliceBF16Handle = linker.downcallHandle(
                lookup.find("sdotSliceBF16").get(),
                FunctionDescriptor.of(
                    ValueLayout.JAVA_FLOAT,   // return float
                    ValueLayout.ADDRESS,      // const float* q
                    ValueLayout.ADDRESS,      // const float* k
                    ValueLayout.JAVA_INT,     // headSize
                    ValueLayout.JAVA_INT,     // blocks
                    ValueLayout.JAVA_INT      // typeSize
                )
            );
        System.out.println("sdotSliceBF16:"+Llama3.sdotSliceBF16Handle);
        Llama3.cublasGetHandle = linker.downcallHandle(
                lookup.find("cublasHandle").get(),
                FunctionDescriptor.of(
                    ValueLayout.JAVA_LONG  // return long handle
                )
            );
        Llama3.cublasFreeHandle = linker.downcallHandle(
                lookup.find("cublasHandleDestroy").get(),
                FunctionDescriptor.ofVoid(
                	// return void
                	ValueLayout.JAVA_LONG  // pass long handle
                )
            );
        Llama3.cudaGetMemInfo = linker.downcallHandle(
                lookup.find("cudaGetMemInfo").get(),
                FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,   // return int
                    ValueLayout.ADDRESS,    // size_t* free
                    ValueLayout.ADDRESS     // size_t* total
                )
            );
	}
}
