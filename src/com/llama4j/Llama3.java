///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 21+
//PREVIEW
//COMPILE_OPTIONS --add-modules=jdk.incubator.vector
//RUNTIME_OPTIONS --add-modules=jdk.incubator.vector -Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0
//MAIN com.llama4j.Llama3

// Practical Llama 3 (and 3.1) inference in a single Java file
// Author: Alfonso² Peterssen
// Based on Andrej Karpathy's llama2.c and minbpe projects
//
// Supports llama.cpp's GGUF format, restricted to Q4_0 and Q8_0 quantized models
// Multi-threaded matrix vector multiplication routines implemented using Java's Vector API
// Simple CLI with --chat and --instruct mode
//
// To run just:
// jbang Llama3.java --help
//
// Remember: Llama models use GPT2 vocabulary while non-Llama models use Llama vocabulary!
// Enjoy!
package com.llama4j;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import java.io.PrintStream;
import java.io.PrintWriter;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.function.IntConsumer;
import java.util.function.IntFunction;
import java.util.function.LongConsumer;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.LongStream;
import java.util.stream.Stream;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import com.neocoretechs.relatrix.client.asynch.AsynchRelatrixClientTransaction;
import com.neocoretechs.relatrix.Result;
import com.neocoretechs.rocksack.TransactionId;

import com.neocoretechs.rocksack.Alias;
import com.llama4j.ffi.NativeLoader;

public class Llama3 {
	private static final Log log = LogFactory.getLog(Llama3.class);
    // Batch-size used in prompt evaluation.
    private static int BATCH_SIZE = Integer.getInteger("llama.BatchSize", 16);
    public final static boolean DEBUG = false;
    public final static boolean DISPLAY_METADATA = true;
    public static AsynchRelatrixClientTransaction dbClient = null;
    public static TransactionId xid = null;
    public static Alias tensorAlias = null;
    // metadata dump
	public static BufferedWriter outputStream = null;
	public static PrintWriter output = null;
	public static FileWriter fileWriter = null;
	public static long[] cublasHandle;
    public static MethodHandle sdotSliceHandle;
    public static MethodHandle sdotSliceQ8Handle;
    public static MethodHandle sdotSliceQ4Handle;
    public static MethodHandle sdotSliceF16Handle;
    public static MethodHandle sdotSliceBF16Handle;
    public static MethodHandle cublasGetHandle;
    public static MethodHandle cublasFreeHandle;
    public static MethodHandle cudaGetMemInfo;
	public static BufferPool poolHead   = new BufferPool();
	public static BufferPool poolScalar = new BufferPool();

	static {
		NativeLoader.load();
	}
	
    static Sampler selectSampler(int vocabularySize, float temperature, float topp, long rngSeed) {
        Sampler sampler;
        if (temperature == 0.0f) {
            // greedy argmax sampling: take the token with the highest probability
            sampler = Sampler.ARGMAX;
        } else {
            // we sample from this distribution to get the next token
            RandomGenerator rng = RandomGeneratorFactory.getDefault().create(rngSeed);
            Sampler innerSampler;
            if (topp <= 0 || topp >= 1) {
                // simply sample from the predicted probability distribution
                innerSampler = new CategoricalSampler(rng);
            } else {
                // top-p (nucleus) sampling, clamping the least likely tokens to zero
                innerSampler = new ToppSampler(vocabularySize, topp, rng);
            }
            sampler = logits -> {
                // apply the temperature to the logits
                logits.divideInPlace(0, logits.size(), temperature);
                // apply softmax to the logits to get the probabilities for next token
                logits.softmaxInPlace(0, logits.size());
                return innerSampler.sampleToken(logits);
            };
        }
        return sampler;
    }

    static void runInteractive(Llama model, Sampler sampler, Options options) {
        Llama.State state = null;
        List<Integer> conversationTokens = new ArrayList<>();
        ChatFormatInterface chatFormat;
        if(DISPLAY_METADATA) {
        	Llama3.output.println("Begin Special tokens:");
        	Llama3.output.println(model.tokenizer().getSpecialTokens());
        	Llama3.output.println("End Special tokens.\r\n");
        }
        // Chat format seems solely based on individual model, so we extract a name in model loader from Metada general.name
        if(ModelLoader.name.equals("mistral")) {
        	chatFormat = new MistralChatFormat(model.tokenizer());
        } else {
        	if(ModelLoader.name.equals("llama")) {
        		chatFormat = new ChatFormat(model.tokenizer());
        	} else {
        		if(ModelLoader.name.equals("qwen")) {
        			BATCH_SIZE = 1;
        			chatFormat = new ChatMLFormat(model.tokenizer());
        		} else {
        			if(ModelLoader.name.equals("magistral")) {
        				chatFormat = new MistralChatFormat(model.tokenizer());
        			} else
        				throw new IllegalArgumentException("expected metadata general.name containing mistral, magistral, llama, or qwen but found "+ModelLoader.name);
        		}
        	}
        }
        conversationTokens.add(chatFormat.getBeginOfText());
        if (options.systemPrompt() != null) {
            conversationTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.SYSTEM, options.systemPrompt())));
        }
        if(options.localNode() != null) {
        	try {
        		dbClient = new AsynchRelatrixClientTransaction(options.localNode(), options.remoteNode(), options.remotePort());
        		xid = dbClient.getTransactionId();
        		tensorAlias = new Alias("Tensors");
        		try {
        			if(dbClient.getAlias(tensorAlias).get() == null)
        				dbClient.setRelativeAlias(tensorAlias);
        		} catch(ExecutionException | InterruptedException ie) {}
        		if(DEBUG)
        			System.out.println("Relatrix transaction Id:"+xid);
        	} catch(IOException ioe) {
        		ioe.printStackTrace();
        	}
        }
        int startPosition = 0;
        Scanner in = new Scanner(System.in);
        loop: while (true) {
        	boolean storeDb = true;
            System.out.print("> ");
            System.out.flush();
            String userText = in.nextLine();
            switch (userText) {
                case "/quit":
                case "/exit": break loop;
                case "/context": {
                    System.out.printf("%d out of %d context tokens used (%d tokens remaining)%n",
                            conversationTokens.size(),
                            model.configuration().contextLength,
                            model.configuration().contextLength - conversationTokens.size());
                    continue;
                }
            }
            if(userText.startsWith("www.") || userText.startsWith("http://") || userText.startsWith("https://")) {
            	String[] urlc = userText.split(" ");
            	Element result = parseLinks(urlc);
            	// replace userText
            	if(result == null)
            		continue;
            	userText = result.text();
            	System.out.println(userText);
            } else {
            	if(userText.startsWith("/recalltime")) {
            		storeDb = false;
            		String[] query = userText.split(" ");
            		String s = parseTime(query);
            		if(s == null)
            			continue;
            		userText = s;
                  	System.out.println(userText);
            	} else {
                  	if(userText.startsWith("/recallwords")) {
                  		storeDb = false;
                		String[] query = userText.split(" ");
                		String s = parseKeywords(query);
                		if(s == null)
                			continue;
                		userText = s;
                      	System.out.println(userText);
                  	}
            	}
            }
            if (state == null) {
                state = model.createNewState(BATCH_SIZE, chatFormat.getBeginOfText());
            }
            conversationTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, userText)));
            conversationTokens.addAll(chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
            Set<Integer> stopTokens = chatFormat.getStopTokens();
            List<Integer> responseTokens;
            if(ModelLoader.name.equals("qwen")) {
            	responseTokens = Llama.generateTokensQwen(model, state, startPosition, conversationTokens.subList(startPosition, conversationTokens.size()), stopTokens, options.maxTokens(), sampler, options.echo(), token -> {
            		if (options.stream()) {
            			int tokenType = model.tokenizer().getTokenType(token);
            			if (tokenType == 1 || tokenType == 6) {
            				System.out.print(model.tokenizer().decode(List.of(token)));
            			}
            		}
            	});
            } else {
            	responseTokens = Llama.generateTokens(model, state, startPosition, conversationTokens.subList(startPosition, conversationTokens.size()), stopTokens, options.maxTokens(), sampler, options.echo(), token -> {
            		if (options.stream()) {
            			if (!model.tokenizer().isSpecialToken(token)) {
            				System.out.print(model.tokenizer().decode(List.of(token)));
            			}
            		}
            	});
            }
            // Include stop token in the prompt history, but not in the response displayed to the user.
            conversationTokens.addAll(responseTokens);
            startPosition = conversationTokens.size();
            Integer stopToken = null;
            if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
                stopToken = responseTokens.getLast();
                responseTokens.removeLast();
            }
            if (!options.stream()) {
                String responseText = model.tokenizer().decode(responseTokens);
                System.out.println(responseText);
                if(dbClient != null && storeDb)
                	dbClient.store(xid, System.currentTimeMillis(), userText, responseText);//.thenAccept(result-> {
                		//System.out.println("Response from storage:"+result);
                	//});
            } else {
                if(dbClient != null && storeDb)
                	dbClient.store(xid, System.currentTimeMillis(), userText, model.tokenizer().decode(responseTokens));//.thenAccept(result-> {
                		//System.out.println("Response from storage:"+result);
                	//});
            }
            if (stopToken == null) {
                System.err.println("Ran out of context length...");
                break;
            }
        }
        if(dbClient != null) {
        	try {
        		dbClient.commit(xid).get();
        		dbClient.endTransaction(xid).get();
        		dbClient.close();
        	} catch(InterruptedException | ExecutionException ie) {}
        }
        if(Llama3.DISPLAY_METADATA) {
        	try {
        		Llama3.outputStream.flush();
        		Llama3.output.close();
        	} catch (final IOException e) {
        		System.err.println("Could not flush metadata file "+e);
        	} finally {
        		try {
        			if (Llama3.outputStream != null) {
        				Llama3.outputStream.close();
        			}
        			if (Llama3.output != null) {
        				Llama3.output.close();
        			}
        		} catch (final IOException e) {
        			System.err.println("Failed to close file: "+e);
        		}
        	}
        }
    }

    static void runInstructOnce(Llama model, Sampler sampler, Options options) {
        ChatFormatInterface chatFormat;
        // Chat format seems solely based on individual model, so we extract a name in model loader from Metada general.name
        if(ModelLoader.name.equals("mistral")) {
        	chatFormat = new MistralChatFormat(model.tokenizer());
        } else {
        	if(ModelLoader.name.equals("llama")) {
        		chatFormat = new ChatFormat(model.tokenizer());
        	} else {
        		if(ModelLoader.name.equals("qwen")) {
        			chatFormat = new ChatMLFormat(model.tokenizer());
        		} else {
        			throw new IllegalArgumentException("expected metadata general.name containing mistral, llama, or qwen but found "+ModelLoader.name);
        		}
        	}
        }
        Llama.State state = model.createNewState(BATCH_SIZE, chatFormat.getBeginOfText());
        List<Integer> promptTokens = new ArrayList<>();
        promptTokens.add(chatFormat.getBeginOfText());
        if (options.systemPrompt() != null) {
            promptTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.SYSTEM, options.systemPrompt())));
        }
        promptTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, options.prompt())));
        promptTokens.addAll(chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));

        Set<Integer> stopTokens = chatFormat.getStopTokens();
        List<Integer> responseTokens = Llama.generateTokens(model, state, 0, promptTokens, stopTokens, options.maxTokens(), sampler, options.echo(), token -> {
            if (options.stream()) {
                if (!model.tokenizer().isSpecialToken(token)) {
                    System.out.print(model.tokenizer().decode(List.of(token)));
                }
            }
        });
        if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
            responseTokens.removeLast();
        }
        if (!options.stream()) {
            String responseText = model.tokenizer().decode(responseTokens);
            System.out.println(responseText);
        }
    }
    /**
     * Parse the command line for url and xpath directive
     * @param urlc array of cmdl args, link at 0
     * @return The Element that matches directive
     */
    private static Element parseLinks(String[] urlc) {
    	//try {	
    		Document doc = null;
    		if(urlc == null || urlc.length < 2)
    			return null;
    		try {
    	 		doc = Jsoup.connect(urlc[0])
        			.userAgent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        			.get();
    		} catch(IOException ioe) {
    			ioe.printStackTrace();
    			return null;
    		}
    		Element result = null;
    		Elements results = null;
    		//for(int i = 1; i < urlc.length; i++) {
    		//	results = doc.select(urlc[i]);
    		//}
    		results = doc.selectXpath(urlc[1]);
    		if(results == null)
    			return null;
    		result = results.first();
    		if(result == null)
    			return null;
    		if(result.is("a"))
    			return parseLinks(new String[] {result.attr("href"),"//a"});
    		return result;
    		//System.out.printf("toString:%s text:%s wholeText:%s%n", result.toString(),result.text(),result.wholeText());
    		//System.out.printf("result is a:%b result is a[href]:%b%n",result.is("a"),result.is("a[href]"));
    	//} catch(MalformedURLException e) {
    	//	e.printStackTrace();
    	//}
    	//return null;
    }
    
    /**
     * element 0 is command <br> /recalltime 
     * arg day time to end day time
     * @param query the command line with command times
     * @return String of Result instances from db that contain 2 elements of question/answer string in time range
     */
    private static String parseTime(String[] query) {
    	CompletableFuture<Stream> s;
		String tq,tqe;
		LocalDateTime localDateTime;
		long millis,millise;
    	if(query == null)
    		return null;
    	if(query.length == 5) {
    		// day time to end day time
    		tq = String.format("%s %s", query[1], query[2]);
    		tqe = String.format("%s %s", query[3], query[4]);
    		localDateTime = LocalDateTime.parse(tq, DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss") );
    		millis = localDateTime.atZone(ZoneId.systemDefault()).toInstant().toEpochMilli();
    		localDateTime = LocalDateTime.parse(tqe, DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss") );
    		millise = localDateTime.atZone(ZoneId.systemDefault()).toInstant().toEpochMilli();
    		s = dbClient.findSubStream(xid,'*','?','?',millis,millise,String.class,String.class);
    		StringBuilder sb = new StringBuilder();
    		try {
    			s.get().forEach(e->{
    				sb.append(((Result)e).get(0));
    				sb.append(((Result)e).get(1));
    			});
    		} catch(InterruptedException | ExecutionException ie) {}
    		return sb.toString();
    	}
    	return null;
    }
    /**
     * Element 0 is command /recallwords
     * @param query the command line with command keywords
     * @return the string of question/answer containing keywords
     */
    private static String parseKeywords(String[] query) {
      	if(query == null || query.length < 2)
    		return null;
     	StringBuilder sb = new StringBuilder();
      	CompletableFuture<Stream> s = dbClient.findStream(xid, '*', '?', '?');
      	try {
      		s.get().forEach(e->{
      			String s1 = (String)((Result)e).get(0);
      			for(int i = 1; i < query.length; i++) {
      				if(s1.contains(query[i])) {
      					sb.append(s1);
      					break;
      				}
      			}
      			s1 = (String)((Result)e).get(1);
      			for(int i = 1; i < query.length; i++) {
      				if(s1.contains(query[i])) {
      					sb.append(s1);
      					break;
      				}
      			}
      		});
      	} catch(InterruptedException | ExecutionException ie) {}
      	return sb.toString();
    }
    
    record Options(Path modelPath, String prompt, String systemPrompt, boolean interactive,
                   float temperature, float topp, long seed, int maxTokens, boolean stream, boolean echo,
                   String localNode, String remoteNode, int remotePort) {

        static final int DEFAULT_MAX_TOKENS = 512;

        Options {
            require(modelPath != null, "Missing argument: --model <path> is required");
            require(interactive || prompt != null, "Missing argument: --prompt is required in --instruct mode e.g. --prompt \"Why is the sky blue?\"");
            require(0 <= temperature, "Invalid argument: --temperature must be non-negative");
            require(0 <= topp && topp <= 1, "Invalid argument: --top-p must be within [0, 1]");
        }

        static void require(boolean condition, String messageFormat, Object... args) {
            if (!condition) {
                System.out.println("ERROR " + messageFormat.formatted(args));
                System.out.println();
                printUsage(System.out);
                System.exit(-1);
            }
        }

        static void printUsage(PrintStream out) {
            out.println("Usage:  jbang Llama3.java [options]");
            out.println();
            out.println("Options:");
            out.println("  --model, -m <path>            required, path to .gguf file");
            out.println("  --interactive, --chat, -i     run in chat mode");
            out.println("  --instruct                    run in instruct (once) mode, default mode");
            out.println("  --prompt, -p <string>         input prompt");
            out.println("  --system-prompt, -sp <string> (optional) system prompt");
            out.println("  --temperature, -temp <float>  temperature in [0,inf], default 0.1");
            out.println("  --top-p <float>               p value in top-p (nucleus) sampling in [0,1] default 0.95");
            out.println("  --seed <long>                 random seed, default System.nanoTime()");
            out.println("  --max-tokens, -n <int>        number of steps to run for < 0 = limited by context length, default " + DEFAULT_MAX_TOKENS);
            out.println("  --stream <boolean>            print tokens during generation; may cause encoding artifacts for non ASCII text, default true");
            out.println("  --echo <boolean>              print ALL tokens to stderr, if true, recommended to set --stream=false, default false");
            out.println("  --localNode <string>          local database client node");
            out.println("  --remoteNode <string>         remote database client node");
            out.println("  --remotePort <int>            remote database port");
            out.println();
            out.println("Examples:");
            out.println("  jbang Llama3.java --model llama3.2-1b-q4_0.gguf --prompt \"Tell me a joke\"");
            out.println("  jbang Llama3.java --model llama3.2-1b-q4_0.gguf --system-prompt \"Reply concisely, in French\" --prompt \"Who was Marie Curie?\"");
            out.println("  jbang Llama3.java --model llama3.2-1b-q4_0.gguf --system-prompt \"Answer concisely\" --chat");
            out.println("  jbang Llama3.java --model llama3.2-1b-q4_0.gguf --chat");
            out.println("  jbang Llama3.java --model llama3.2-1b-q4_0.gguf --prompt \"Print 5 emojis\" --stream=false");
        }

        static Options parseOptions(String[] args) {
            String prompt = null;
            String systemPrompt = null;
            float temperature = 0.1f;
            float topp = 0.95f;
            Path modelPath = null;
            long seed = System.nanoTime();
            // Keep max context length small for low-memory devices.
            int maxTokens = DEFAULT_MAX_TOKENS;
            boolean interactive = false;
            boolean stream = true;
            boolean echo = false;
            String localNode = null;
            String remoteNode = null;
            int remotePort = 0;

            for (int i = 0; i < args.length; i++) {
                String optionName = args[i];
                require(optionName.startsWith("-"), "Invalid option %s", optionName);
                switch (optionName) {
                    case "--interactive", "--chat", "-i" -> interactive = true;
                    case "--instruct" -> interactive = false;
                    case "--help", "-h" -> {
                        printUsage(System.out);
                        System.exit(0);
                    }
                    default -> {
                        String nextArg;
                        if (optionName.contains("=")) {
                            String[] parts = optionName.split("=", 2);
                            optionName = parts[0];
                            nextArg = parts[1];
                        } else {
                            require(i + 1 < args.length, "Missing argument for option %s", optionName);
                            nextArg = args[i + 1];
                            i += 1; // skip arg
                        }
                        switch (optionName) {
                            case "--prompt", "-p" -> prompt = nextArg;
                            case "--system-prompt", "-sp" -> systemPrompt = nextArg;
                            case "--temperature", "--temp" -> temperature = Float.parseFloat(nextArg);
                            case "--top-p" -> topp = Float.parseFloat(nextArg);
                            case "--model", "-m" -> modelPath = Paths.get(nextArg);
                            case "--seed", "-s" -> seed = Long.parseLong(nextArg);
                            case "--max-tokens", "-n" -> maxTokens = Integer.parseInt(nextArg);
                            case "--stream" -> stream = Boolean.parseBoolean(nextArg);
                            case "--echo" -> echo = Boolean.parseBoolean(nextArg);
                            case "--localNode" -> localNode = nextArg;
                            case "--remoteNode" -> remoteNode = nextArg;
                            case "--remotePort" -> remotePort = Integer.parseInt(nextArg);
                            default -> require(false, "Unknown option: %s", optionName);
                        }
                    }
                }
            }
            return new Options(modelPath, prompt, systemPrompt, interactive, temperature, topp, seed, maxTokens, stream, echo, localNode, remoteNode, remotePort);
        }
    }

    public static void main(String[] args) throws IOException {
        Options options = Options.parseOptions(args);
        if(Llama3.DISPLAY_METADATA) {
        	try {
        		Llama3.fileWriter = new FileWriter(options.modelPath.toString()+".metadata", false);
        		Llama3.outputStream = new BufferedWriter(fileWriter);
        		Llama3.output = new PrintWriter(outputStream);
        	} catch (final IOException e) {
        		System.err.println("Could not open file " + options.modelPath.toString()+".metadata\r\n"+e);
        	}
        }
        NativeLoader.loadMethods();
        Llama model = AOT.tryUsePreLoaded(options.modelPath(), options.maxTokens());
        if(model == null)
        	model = ModelLoader.loadModel(options.modelPath(), options.maxTokens(), true);
        if(FloatTensor.USE_CUDA) {
        	Llama3.cublasHandle = new long[model.configuration().numberOfHeads];
        	try {
        		for(int i = 0; i < model.configuration().numberOfHeads; i++) {
        			Llama3.cublasHandle[i] = (long) Llama3.cublasGetHandle.invokeExact();
        		}
        	} catch(Throwable t) {
        		t.printStackTrace();
        		System.exit(1);
        	}
        }
        Sampler sampler = selectSampler(model.configuration().vocabularySize, options.temperature(), options.topp(), options.seed());
        if (options.interactive()) {
            runInteractive(model, sampler, options);
        } else {
            runInstructOnce(model, sampler, options);
        }
    }
}

final class GGUF {
    private static final int GGUF_MAGIC = 0x46554747;
    private static final int DEFAULT_ALIGNMENT = 32; // must be a power of 2
    private static final List<Integer> SUPPORTED_GGUF_VERSIONS = List.of(2, 3);
    private int magic;
    private int version;
    private int tensorCount; // uint64_t
    private int alignment;
    private int metadata_kv_count; // uint64_t
    private Map<String, Object> metadata;

    public Map<String, GGUFTensorInfo> getTensorInfos() {
        return tensorInfos;
    }

    private Map<String, GGUFTensorInfo> tensorInfos;

    private long tensorDataOffset;

    public long getTensorDataOffset() {
        return tensorDataOffset;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    private final ByteBuffer BB_1 = ByteBuffer.allocate(Byte.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    private final ByteBuffer BB_2 = ByteBuffer.allocate(Short.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    private final ByteBuffer BB_4 = ByteBuffer.allocate(Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    private final ByteBuffer BB_8 = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);

    public static GGUF loadModel(Path modelPath) throws IOException {
        try (FileChannel fileChannel = FileChannel.open(modelPath);
            var ignored = Timer.log("Parse " + modelPath)) {
            GGUF gguf = new GGUF();
            gguf.loadModelImpl(fileChannel);
            return gguf;
        }
    }

    enum MetadataValueType {
        // The value is a 8-bit unsigned integer.
        UINT8(1),
        // The value is a 8-bit signed integer.
        INT8(1),
        // The value is a 16-bit unsigned little-endian integer.
        UINT16(2),
        // The value is a 16-bit signed little-endian integer.
        INT16(2),
        // The value is a 32-bit unsigned little-endian integer.
        UINT32(4),
        // The value is a 32-bit signed little-endian integer.
        INT32(4),
        // The value is a 32-bit IEEE754 floating point number.
        FLOAT32(4),
        // The value is a boolean.
        // 1-byte value where 0 is false and 1 is true.
        // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
        BOOL(1),
        // The value is a UTF-8 non-null-terminated string, with length prepended.
        STRING(-8),
        // The value is an array of other values, with the length and type prepended.
        // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
        ARRAY(-8),
        // The value is a 64-bit unsigned little-endian integer.
        UINT64(8),
        // The value is a 64-bit signed little-endian integer.
        INT64(8),
        // The value is a 64-bit IEEE754 floating point number.
        FLOAT64(8);
        private final int byteSize;

        MetadataValueType(int byteSize) {
            this.byteSize = byteSize;
        }

        private static final MetadataValueType[] VALUES = values();

        public static MetadataValueType fromIndex(int index) {
            return VALUES[index];
        }

        public int byteSize() {
            return byteSize;
        }
    }

    private void loadModelImpl(FileChannel fileChannel) throws IOException {
        // The header of the file.
        readHeader(fileChannel); // gguf_header_t header;
        // Tensor infos, which can be used to locate the tensor data.
        // gguf_tensor_info_t tensor_infos[header.tensor_count];
        this.tensorInfos = HashMap.newHashMap(tensorCount);
        for (int i = 0; i < tensorCount; ++i) {
            GGUF.GGUFTensorInfo ti = readTensorInfo(fileChannel);
            assert !tensorInfos.containsKey(ti.name);
            tensorInfos.put(ti.name, ti);
        }
        // Padding to the nearest multiple of `ALIGNMENT`.
        // uint8_t _padding[ALIGNMENT - (sizeof(header + tensor_infos) % ALIGNMENT)];
        //long _padding = -fileChannel.position() & (ALIGNMENT - 1);
        long _padding = getAlignment() - (fileChannel.position() % getAlignment());
        fileChannel.position(fileChannel.position() + _padding);
        // Tensor data.
        //
        // This is arbitrary binary data corresponding to the weights of the model. This data should be close
        // or identical to the data in the original model file, but may be different due to quantization or
        // other optimizations for inference. Any such deviations should be recorded in the metadata or as
        // part of the architecture definition.
        //
        // Each tensor's data must be stored within this array, and located through its `tensor_infos` entry.
        // The offset of each tensor's data must be a multiple of `ALIGNMENT`, and the space between tensors
        // should be padded to `ALIGNMENT` bytes.
        // uint8_t tensor_data[];
        this.tensorDataOffset = fileChannel.position();
    }

    public static Map<String, GGMLTensorEntry> loadTensors(FileChannel fileChannel, long tensorDataOffset, Map<String, GGUFTensorInfo> tensorInfos) throws IOException {
        Arena arena = Arena.ofAuto();
        MemorySegment tensorData = fileChannel.map(FileChannel.MapMode.READ_ONLY, tensorDataOffset, fileChannel.size() - tensorDataOffset, arena);
        Map<String, GGMLTensorEntry> tensorEntries = HashMap.newHashMap(tensorInfos.size());
        if(Llama3.DISPLAY_METADATA)
        	Llama3.output.println("Begin Tensors:");
        for (Map.Entry<String, GGUFTensorInfo> entry : tensorInfos.entrySet()) {
            GGUFTensorInfo ti = entry.getValue();
            int numberOfElements = FloatTensor.numberOfElements(ti.dimensions());
            int sizeInBytes = Math.toIntExact(ti.ggmlType().byteSizeFor(numberOfElements));
            if(Llama3.DISPLAY_METADATA)
            	Llama3.output.println("Tensor:"+entry.getKey()+"="+ti.name+" offset:"+ti.offset+" dims:"+Arrays.toString(ti.dimensions)+" number elems:"+numberOfElements+" size:"+sizeInBytes);
            MemorySegment memorySegment = tensorData.asSlice(ti.offset(), sizeInBytes);
            tensorEntries.put(ti.name(), new GGMLTensorEntry(tensorData, ti.name(), ti.ggmlType(), ti.dimensions(), memorySegment));
        }
        if(Llama3.DISPLAY_METADATA)
        	Llama3.output.println("End Tensors.\r\n");
        return tensorEntries;
    }

    public record GGUFTensorInfo(String name, int[] dimensions, GGMLType ggmlType, long offset) {
    }

    private GGMLType readGGMLType(FileChannel fileChannel) throws IOException {
        int ggmlTypeId = readInt(fileChannel); // ggml_type type;
        return GGMLType.fromId(ggmlTypeId);
    }

    private GGUF.GGUFTensorInfo readTensorInfo(FileChannel fileChannel) throws IOException {
        // The name of the tensor. It is a standard GGUF string, with the caveat that
        // it must be at most 64 bytes long.
        String name = readString(fileChannel); // gguf_string_t name;
        assert name.length() <= 64;
        // The number of dimensions in the tensor.
        // Currently at most 4, but this may change in the future.
        int n_dimensions = readInt(fileChannel); // uint32_t n_dimensions;
        assert n_dimensions <= 4;
        // The dimensions of the tensor.
        int[] dimensions = new int[n_dimensions]; // uint64_t dimensions[n_dimensions];
        for (int i = 0; i < n_dimensions; ++i) {
            dimensions[i] = Math.toIntExact(readLong(fileChannel));
        }
        // The type of the tensor.
        GGMLType ggmlType = readGGMLType(fileChannel); // ggml_type type;
        // The offset of the tensor's data in this file in bytes.
        // This offset is relative to `tensor_data`, not to the start
        // of the file, to make it easier for writers to write the file.
        // Readers should consider exposing this offset relative to the
        // file to make it easier to read the data.
        // Must be a multiple of `ALIGNMENT`.
        long offset = readLong(fileChannel); // uint64_t offset;
        assert offset % getAlignment() == 0;
        return new GGUF.GGUFTensorInfo(name, dimensions, ggmlType, offset);
    }

    private String readString(FileChannel fileChannel) throws IOException {
        // A string in GGUF.
        // The length of the string, in bytes.
        int len = Math.toIntExact(readLong(fileChannel)); // uint64_t len;
        // The string as a UTF-8 non-null-terminated string.
        byte[] bytes = new byte[len]; // char string[len];
        int bytesRead = fileChannel.read(ByteBuffer.wrap(bytes));
        assert len == bytesRead;
        return new String(bytes, StandardCharsets.UTF_8);
    }

    private Pair<String, Object> readKeyValuePair(FileChannel fileChannel) throws IOException {
        // The key of the metadata. It is a standard GGUF string, with the following caveats:
        // - It must be a valid ASCII string.
        // - It must be a hierarchical key, where each segment is `lower_snake_case` and separated by a `.`.
        // - It must be at most 2^16-1/65535 bytes long.
        // Any keys that do not follow these rules are invalid.
        String key = readString(fileChannel); // gguf_string_t key;
        assert key.length() < (1 << 16);
        assert key.codePoints().allMatch(cp -> ('a' <= cp && cp <= 'z') || ('0' <= cp && cp <= '9') || cp == '_' || cp == '.');
        Object value = readMetadataValue(fileChannel);
        return new Pair<>(key, value);
    }

    private Object readMetadataValue(FileChannel fileChannel) throws IOException {
        // The type of the value.
        // Must be one of the `gguf_metadata_value_type` values.
        MetadataValueType value_type = readMetadataValueType(fileChannel); // gguf_metadata_value_type value_type;
        // The value.
        return readMetadataValueOfType(value_type, fileChannel); // gguf_metadata_value_t value;
    }

    void readHeader(FileChannel fileChannel) throws IOException {
        // Magic number to announce that this is a GGUF file.
        // Must be `GGUF` at the byte level: `0x47` `0x47` `0x55` `0x46`.
        // Your executor might do little-endian byte order, so it might be
        // check for 0x46554747 and letting the endianness cancel out.
        // Consider being *very* explicit about the byte order here.
        this.magic = readInt(fileChannel); //    uint32_t magic;
        if (magic != GGUF_MAGIC) {
            throw new IllegalArgumentException("unsupported header.magic " + magic);
        }
        // The version of the format implemented.
        // Must be `3` for version described in this spec.
        //
        // This version should only be increased for structural changes to the format.
        // Changes that do not affect the structure of the file should instead update the metadata
        // to signify the change.
        this.version = readInt(fileChannel); // uint32_t version;
        if (!SUPPORTED_GGUF_VERSIONS.contains(version)) {
            throw new IllegalArgumentException("unsupported header.version " + version);
        }
        // The number of tensors in the file.
        // This is explicit, instead of being included in the metadata, to ensure it is always present
        // for loading the tensors.
        this.tensorCount = Math.toIntExact(readLong(fileChannel)); // uint64_t tensor_count;
        // The number of metadata key-value pairs.
        this.metadata_kv_count = Math.toIntExact(readLong(fileChannel)); // uint64_t metadata_kv_count;
        // The metadata key-value pairs.
        // gguf_metadata_kv_t metadata_kv[metadata_kv_count];
        this.metadata = HashMap.newHashMap(metadata_kv_count);
        for (int i = 0; i < metadata_kv_count; ++i) {
            Pair<String, Object> keyValue = readKeyValuePair(fileChannel);
            assert !metadata.containsKey(keyValue.first());
            metadata.put(keyValue.first(), keyValue.second());
        }
    }

    private Object readArray(FileChannel fileChannel) throws IOException {
        // Any value type is valid, including arrays.
        MetadataValueType value_type = readMetadataValueType(fileChannel); // gguf_metadata_value_type type;
        // Number of elements, not bytes
        int len = Math.toIntExact(readLong(fileChannel)); // uint64_t len;
        // The array of values.
        // gguf_metadata_value_t array[len];
        switch (value_type) {
            case UINT8, INT8 -> {
                byte[] bytes = new byte[len];
                for (int i = 0; i < len; ++i) {
                    bytes[i] = readByte(fileChannel);
                }
                return bytes;
            }
            case UINT16, INT16 -> {
                short[] shorts = new short[len];
                for (int i = 0; i < len; ++i) {
                    shorts[i] = readShort(fileChannel);
                }
                return shorts;
            }
            case UINT32, INT32 -> {
                int[] ints = new int[len];
                for (int i = 0; i < len; ++i) {
                    ints[i] = readInt(fileChannel);
                }
                return ints;
            }
            case FLOAT32 -> {
                float[] floats = new float[len];
                for (int i = 0; i < len; ++i) {
                    floats[i] = readFloat(fileChannel);
                }
                return floats;
            }
            case BOOL -> {
                boolean[] booleans = new boolean[len];
                for (int i = 0; i < len; ++i) {
                    booleans[i] = readBoolean(fileChannel);
                }
                return booleans;
            }
            case STRING -> {
                String[] strings = new String[len];
                for (int i = 0; i < len; ++i) {
                    strings[i] = readString(fileChannel);
                }
                return strings;
            }
            case ARRAY -> {
                Object[] arrays = new Object[len];
                for (int i = 0; i < len; ++i) {
                    arrays[i] = readArray(fileChannel);
                }
                return arrays;
            }
            default -> throw new UnsupportedOperationException("read array of " + value_type);
        }
    }

    private Object readMetadataValueOfType(MetadataValueType valueType, FileChannel fileChannel) throws IOException {
        return switch (valueType) {
            case UINT8, INT8 -> readByte(fileChannel);
            case UINT16, INT16 -> readShort(fileChannel);
            case UINT32, INT32 -> readInt(fileChannel);
            case FLOAT32 -> readFloat(fileChannel);
            case UINT64, INT64 -> readLong(fileChannel);
            case FLOAT64 -> readDouble(fileChannel);
            case BOOL -> readBoolean(fileChannel);
            case STRING -> readString(fileChannel);
            case ARRAY -> readArray(fileChannel);
        };
    }

    private byte readByte(FileChannel fileChannel) throws IOException {
        int bytesRead = fileChannel.read(BB_1);
        assert bytesRead == 1;
        return BB_1.clear().get(0);
    }

    private boolean readBoolean(FileChannel fileChannel) throws IOException {
        return readByte(fileChannel) != 0;
    }

    private short readShort(FileChannel fileChannel) throws IOException {
        int bytesRead = fileChannel.read(BB_2);
        assert bytesRead == 2;
        return BB_2.clear().getShort(0);
    }

    private int readInt(FileChannel fileChannel) throws IOException {
        int bytesRead = fileChannel.read(BB_4);
        assert bytesRead == 4;
        return BB_4.clear().getInt(0);
    }

    private long readLong(FileChannel fileChannel) throws IOException {
        int bytesRead = fileChannel.read(BB_8);
        assert bytesRead == 8;
        return BB_8.clear().getLong(0);
    }

    private float readFloat(FileChannel fileChannel) throws IOException {
        return Float.intBitsToFloat(readInt(fileChannel));
    }

    private double readDouble(FileChannel fileChannel) throws IOException {
        return Double.longBitsToDouble(readLong(fileChannel));
    }

    private MetadataValueType readMetadataValueType(FileChannel fileChannel) throws IOException {
        int index = readInt(fileChannel);
        return MetadataValueType.fromIndex(index);
    }

    public int getAlignment() {
        if (alignment != 0) {
            return alignment;
        }
        alignment = (int) metadata.getOrDefault("general.alignment", DEFAULT_ALIGNMENT);
        assert Integer.bitCount(alignment) == 1 : "alignment must be a power of two";
        return alignment;
    }
}

interface Timer extends AutoCloseable {
    @Override
    void close(); // no Exception

    static Timer log(String label) {
        return log(label, TimeUnit.MILLISECONDS);
    }

    static Timer log(String label, TimeUnit timeUnit) {
        return new Timer() {
            final long startNanos = System.nanoTime();

            @Override
            public void close() {
                long elapsedNanos = System.nanoTime() - startNanos;
                System.err.println(label + ": "
                        + timeUnit.convert(elapsedNanos, TimeUnit.NANOSECONDS) + " "
                        + timeUnit.toChronoUnit().name().toLowerCase());
            }
        };
    }
}
/**
 * Load model, get GGUF metadata, load vocabulary, create tokenizer, create config, if loadWeights - load tensors, load weights
 * create Llama with config, tokenizer, weights
 */
final class ModelLoader {
    static final String TOKENIZER_GPT2_MODEL = "gpt2"; // Llama3 uses gpt2!
    static final String TOKENIZER_LLAMA_MODEL = "llama"; // non Llama uses llama!
    public static String model = "gpt2"; // default for Llama models!
    public static String name = null; // Name is based solely on name of model, they all seem to have their own ChatFormat not based on model
    private static final String LLAMA_3_PATTERN = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    private static Vocabulary loadVocabulary(Map<String, Object> metadata) {
        model = (String) metadata.get("tokenizer.ggml.model");
        name = (String) metadata.get("general.name");
        if(name.toLowerCase().contains("llama")) // Meta Llama etc. etc.
        	name = "llama";
        else
        	if(name.toLowerCase().contains("mistral")) //models--mistralai etc. etc.
        		name="mistral";
        	else
        		if(name.toLowerCase().contains("qwen"))
        			name="qwen";
        		else
        			if(name.toLowerCase().contains("magistral"))
        				name="magistral";
        String[] tokens = (String[]) metadata.get("tokenizer.ggml.tokens");
        if(TOKENIZER_LLAMA_MODEL.equals(model)) {
        	float[] scores = (float[]) metadata.get("tokenizer.ggml.scores");
        	return new Vocabulary(tokens, scores);
        } else {
        	if(TOKENIZER_GPT2_MODEL.equals(model)) {
        		return new Vocabulary(tokens, null);
        	} else {
        		throw new IllegalArgumentException("expected " + TOKENIZER_GPT2_MODEL + " or "+ TOKENIZER_LLAMA_MODEL+ " but found " + model);
        	}
        }
    }

    public static Llama loadModel(Path ggufPath, int contextLength, boolean loadWeights) throws IOException {
        GGUF gguf = GGUF.loadModel(ggufPath);
        FileChannel fileChannel = FileChannel.open(ggufPath, StandardOpenOption.READ);
        return loadModel(fileChannel, gguf, contextLength, loadWeights);
    }

    public static Llama loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights) throws IOException {
        try (var ignored = Timer.log("Load model")) {
            Map<String, Object> metadata = gguf.getMetadata();
            if(Llama3.DISPLAY_METADATA) {
            	Llama3.output.println("Begin GGUF Metadata:");
            	metadata.forEach((k, v) -> {
            		String valueStr;
            		if (v != null && v.getClass().isArray()) {
            			Class<?> componentType = v.getClass().getComponentType();
            			if (componentType == int.class) {
            				valueStr = Arrays.toString((int[]) v);
            			} else if (componentType == byte.class) {
            				valueStr = Arrays.toString((byte[]) v);
            			} else if (componentType == double.class) {
            				valueStr = Arrays.toString((double[]) v);
            			} else if (componentType == boolean.class) {
            				valueStr = Arrays.toString((boolean[]) v);
            			} else if (componentType == char.class) {
            				valueStr = Arrays.toString((char[]) v);
            			} else if (componentType == long.class) {
            				valueStr = Arrays.toString((long[]) v);
            			} else if (componentType == float.class) {
            				valueStr = Arrays.toString((float[]) v);
            			} else if (componentType == short.class) {
            				valueStr = Arrays.toString((short[]) v);
            			} else {
            				valueStr = Arrays.toString((Object[]) v); // for Object arrays
            			}
            		} else {
            			valueStr = String.valueOf(v);
            		}
            		Llama3.output.println(k + "=" + valueStr);
            	});
            	Llama3.output.println("End GGUF Metadata.\r\n");
            }
            Vocabulary vocabulary = loadVocabulary(metadata);
            TokenizerInterface tokenizer;
            Llama.Configuration config;
            Llama.Weights weights = null;
            String arch = (String) metadata.get("general.architecture");
            if(ModelLoader.name.equals("mistral")) {
           		tokenizer = createLlamaTokenizer(metadata, vocabulary);
        		config = createConfig(arch, metadata, vocabulary, contextLength);
        		if (loadWeights) {
        			// loadTensors corresponds to getTensorEntries in old version
        			Map<String, GGMLTensorEntry> tensorEntries = GGUF.loadTensors(fileChannel, gguf.getTensorDataOffset(), gguf.getTensorInfos());
        			weights = loadLlamaWeights(tensorEntries, config);
        		}
            } else {
            	if(ModelLoader.name.equals("llama")) {
                   	tokenizer = createGPT2Tokenizer(metadata, vocabulary);
                	config = createConfig(arch, metadata, vocabulary, contextLength);
                    if (loadWeights) {
                    	// loadTensors corresponds to getTensorEntries in old version
                        Map<String, GGMLTensorEntry> tensorEntries = GGUF.loadTensors(fileChannel, gguf.getTensorDataOffset(), gguf.getTensorInfos());
                        weights = loadGPT2Weights(tensorEntries, config);
                    }
            	} else {
            		if(ModelLoader.name.equals("qwen")) {
                      	tokenizer = createQwen2Tokenizer(metadata, vocabulary);
                    	config = createConfig(arch, metadata, vocabulary, contextLength);
                        if (loadWeights) {
                        	// loadTensors corresponds to getTensorEntries in old version
                            Map<String, GGMLTensorEntry> tensorEntries = GGUF.loadTensors(fileChannel, gguf.getTensorDataOffset(), gguf.getTensorInfos());
                            weights = loadQwenWeights(tensorEntries, config);
                        }
            		} else {
            			if(ModelLoader.name.equals("magistral")) {
                          	tokenizer = createMagistralTokenizer(metadata, vocabulary);
                        	config = createConfig(arch, metadata, vocabulary, contextLength);
                            if (loadWeights) {
                            	// loadTensors corresponds to getTensorEntries in old version
                                Map<String, GGMLTensorEntry> tensorEntries = GGUF.loadTensors(fileChannel, gguf.getTensorDataOffset(), gguf.getTensorInfos());
                                weights = loadLlamaWeights(tensorEntries, config);
                            }
            			} else
            				throw new IllegalArgumentException("expected metadata general.name containing mistral, magistral, llama, or qwen but found "+ModelLoader.name);
            		}
            	}
            }
            return new Llama(config, tokenizer, weights);
        }
    }
    
    static Llama.Configuration createConfig(String arch, Map<String, Object> metadata, Vocabulary vocabulary, int contextLength) {
        Llama.Configuration config = new Llama.Configuration(
                (int) metadata.get(arch+".embedding_length"),
                (int) metadata.get(arch+".feed_forward_length"),
                (int) metadata.get(arch+".block_count"),
                (int) metadata.get(arch+".attention.head_count"),

                metadata.containsKey(arch+".attention.head_count_kv")
                        ? (int) metadata.get(arch+".attention.head_count_kv")
                        : (int) metadata.get(arch+".attention.head_count"),

                vocabulary.size(),
                (int) metadata.get(arch+".context_length"),
                (float) metadata.getOrDefault(arch+".attention.layer_norm_rms_epsilon", 1e-5f),
                (float) metadata.getOrDefault(arch+".rope.freq_base", 10000f)
        ).withContextLength(contextLength);
        return config;
    }
    

    /**
     * Called from AOT.tryUsePreloaded and ModelLoader.loadModel
     * @param tensorEntries
     * @param config
     * @return
     */
    static Llama.Weights loadGPT2Weights(Map<String, GGMLTensorEntry> tensorEntries, Llama.Configuration config) {
        boolean ropeScaling = tensorEntries.containsKey("rope_freqs");
        float scaleFactor = 8;
        float loFreqFactor = 1;
        float hiFreqFactor = 3;
        int oldContextLength = 8192;
        Pair<float[], float[]> ropeFreqs = RoPE.precomputeFreqsCis(config.contextLength, config.headSize, config.ropeTheta,
                ropeScaling, scaleFactor, loFreqFactor, hiFreqFactor, oldContextLength);
        float[] ropeFreqsReal = ropeFreqs.first();
        float[] ropeFreqsImag = ropeFreqs.second();

        GGMLTensorEntry tokenEmbeddings = tensorEntries.get("token_embd.weight");
        Llama.Weights qw = new Llama.Weights(
                loadQuantized(tokenEmbeddings),
                loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")), // w1
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_down.weight")), // w2
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_up.weight")), // w3
                toFloatBuffer(tensorEntries.get("output_norm.weight")),
                FloatBuffer.wrap(ropeFreqsReal),
                FloatBuffer.wrap(ropeFreqsImag),
                // If "output.weight" is not present then the embedding weights are tied/shared with the decoder.
                // This is commonly referred as "tie word embeddings".
                loadQuantized(tensorEntries.getOrDefault("output.weight", tokenEmbeddings))
        );
        return qw;
    }
    
    static Llama.Weights loadLlamaWeights(Map<String, GGMLTensorEntry> tensorEntries, Llama.Configuration config) {
    	   Pair<float[], float[]> ropeFreqs = RoPE.precomputeFreqsCis(config.contextLength, config.headSize, config.ropeTheta);
           float[] ropeFreqsReal = ropeFreqs.first();
           float[] ropeFreqsImag = ropeFreqs.second();

           Llama.Weights qw = new Llama.Weights(
                   loadQuantized(tensorEntries.get("token_embd.weight")),
                   loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                   loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                   loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                   loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                   loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                   loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                   loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")), // w1
                   loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_down.weight")), // w2
                   loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_up.weight")), // w3
                   toFloatBuffer(tensorEntries.get("output_norm.weight")),
                   FloatBuffer.wrap(ropeFreqsReal),
                   FloatBuffer.wrap(ropeFreqsImag),
                   loadQuantized(tensorEntries.get("output.weight"))
           );
           return qw;
    }
    
    static Llama.Weights loadQwenWeights(Map<String, GGMLTensorEntry> tensorEntries, Llama.Configuration config) {
   	   Pair<float[], float[]> ropeFreqs = RoPE.precomputeFreqsCis(config.contextLength, config.headSize, config.ropeTheta);
       float[] ropeFreqsReal = ropeFreqs.first();
       float[] ropeFreqsImag = ropeFreqs.second();

    	Llama.Weights qw = new Llama.Weights(
    			loadQuantized(tensorEntries.get("token_embd.weight")),
    			loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
    			loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
    			loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
    			loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
    			loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_q.bias")),
    			loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_k.bias")),
    			loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_v.bias")),
    			loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
    			loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
    			loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")), // w1
    			loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_down.weight")), // w2
    			loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_up.weight")), // w3
    			toFloatBuffer(tensorEntries.get("output_norm.weight")),
    			FloatBuffer.wrap(ropeFreqsReal),
    			FloatBuffer.wrap(ropeFreqsImag),
    			loadQuantized(tensorEntries.get("output.weight"))
    			);
    	return qw;
    }

    private final static String QWEN2_PATTERN = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    private static Tokenizer createQwen2Tokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        int[] tokenTypes = (int[]) metadata.get("tokenizer.ggml.token_type");
        String[] mergeLines = (String[]) metadata.get("tokenizer.ggml.merges");
        List<Pair<Integer, Integer>> merges = Arrays.stream(mergeLines)
                .map(line -> line.split(" "))
                .map(parts ->
                        new Pair<>(
                                vocabulary.getIndex(parts[0]).orElseThrow(),
                                vocabulary.getIndex(parts[1]).orElseThrow())
                ).toList();

        int allTokens = vocabulary.size();
        int baseTokens = vocabulary.getIndex("<|endoftext|>").orElseThrow(); // assume all tokens after the base ones are special.
        int reservedSpecialTokens = allTokens - baseTokens;
        List<String> specialTokensList = Arrays.stream(vocabulary.tokens(), baseTokens, allTokens).toList();

        assert specialTokensList.stream().allMatch(token -> vocabulary.getIndex(token).isPresent());

        Map<String, Integer> specialTokens =
                IntStream.range(0, specialTokensList.size())
                        .boxed()
                        .collect(Collectors.toMap(
                                i -> specialTokensList.get(i),
                                i -> baseTokens + i)
                        );

        return new Tokenizer(vocabulary, merges, QWEN2_PATTERN, specialTokens, tokenTypes);
    }

    private static Tokenizer createGPT2Tokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        String[] mergeLines = (String[]) metadata.get("tokenizer.ggml.merges");
        List<Pair<Integer, Integer>> merges = Arrays.stream(mergeLines)
                .map(line -> line.split(" "))
                .map(parts ->
                        new Pair<>(
                                vocabulary.getIndex(parts[0]).orElseThrow(),
                                vocabulary.getIndex(parts[1]).orElseThrow())
                ).toList();

        int allTokens = vocabulary.size();
        int baseTokens = 128000; // assume all tokens after the base ones are special.
        int reservedSpecialTokens = allTokens - baseTokens;
        List<String> specialTokensList = Arrays.stream(vocabulary.tokens(), baseTokens, allTokens).toList();

        assert specialTokensList.stream().allMatch(token -> vocabulary.getIndex(token).isPresent());

        Map<String, Integer> specialTokens =
                IntStream.range(0, specialTokensList.size())
                        .boxed()
                        .collect(Collectors.toMap(
                                i -> specialTokensList.get(i),
                                i -> baseTokens + i)
                        );

        return new Tokenizer(vocabulary, merges, LLAMA_3_PATTERN, specialTokens);
    }
    
    private static MistralTokenizer createLlamaTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        int[] tokenTypes = (int[]) metadata.get("tokenizer.ggml.token_type");
        List<Integer> specialTokensList = IntStream.range(0, vocabulary.size()).filter(t -> tokenTypes[t] != 1 && tokenTypes[t] != 6).boxed().toList();
        Map<String, Integer> specialTokens =
                IntStream.range(0, specialTokensList.size())
                        .boxed()
                        .collect(Collectors.toMap(
                                t -> vocabulary.get(t),
                                t -> t)
                        );
        return new MistralTokenizer(vocabulary, null, specialTokens, tokenTypes);
    }
    
    private static Tokenizer createMagistralTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        int[] tokenTypes = (int[]) metadata.get("tokenizer.ggml.token_type");
        List<Integer> specialTokensList = IntStream.range(0, vocabulary.size()).filter(t -> tokenTypes[t] != 1 && tokenTypes[t] != 6).boxed().toList();
        Map<String, Integer> specialTokens =
                IntStream.range(0, specialTokensList.size())
                        .boxed()
                        .collect(Collectors.toMap(
                                t -> vocabulary.get(t),
                                t -> t)
                        );
        String[] mergeLines = (String[]) metadata.get("tokenizer.ggml.merges");
        List<Pair<Integer, Integer>> merges = Arrays.stream(mergeLines)
                .map(line -> line.split(" "))
                .map(parts ->
                        new Pair<>(
                                vocabulary.getIndex(parts[0]).orElseThrow(),
                                vocabulary.getIndex(parts[1]).orElseThrow())
                ).toList();

        return new Tokenizer(vocabulary, merges, LLAMA_3_PATTERN, specialTokens, tokenTypes);
    }
    
    public static FloatTensor loadQuantized(GGMLTensorEntry entry) {
        GGMLType ggmlType = entry.ggmlType();
        return switch (ggmlType) {
            case F32 -> new F32FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case Q8_0 -> new Q8_0FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case Q4_0 -> new Q4_0FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case BF16 -> new BF16FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case F16 -> new F16FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            default -> throw new UnsupportedOperationException("Quantization format " + ggmlType);
        };
    }

    public static FloatTensor[] loadArrayOfQuantized(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatTensor[] array = new FloatTensor[size];
        for (int i = 0; i < size; i++) {
            array[i] = loadQuantized(getTensorEntry.apply(i));
        }
        return array;
    }

    public static FloatBuffer[] loadArrayOfFloatBuffer(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatBuffer[] array = new FloatBuffer[size];
        for (int i = 0; i < size; i++) {
            array[i] = toFloatBuffer(getTensorEntry.apply(i));
        }
        return array;
    }

    public static FloatBuffer toFloatBuffer(GGMLTensorEntry tensorEntry) {
        GGMLType ggmlType = tensorEntry.ggmlType();
        return switch (ggmlType) {
            case F32 -> tensorEntry.memorySegment().asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            default -> throw new UnsupportedOperationException("Conversion to " + ggmlType);
        };
    }
}

record Llama(Configuration configuration, TokenizerInterface tokenizer, Weights weights) {
    private static boolean DEBUG = false;;

	public State createNewState(int batchsize, int beginOfText) {
        State state = new State(configuration(), batchsize);
        state.latestToken = beginOfText; // was tokenizer.getSpecialTokens().get("<|begin_of_text|>");, now we get from ChatFormat.beginOfText() which does the same
        return state;
    }

    public static final class Configuration {
        public final int dim; // transformer dimension
        public final int hiddenDim; // for ffn layers
        public final int numberOfLayers; // number of layers
        public final int numberOfHeads; // number of query heads
        public final int numberOfKeyValueHeads; // number of key/value heads (can be < query heads because of multiquery)
        public final int vocabularySize; // vocabulary size, usually 256 (byte-level)
        public final int contextLength; // max sequence length
        public final float rmsNormEps;
        public final float ropeTheta;
        public final int headSize;

        Configuration withContextLength(int newContextLength) {
            if (newContextLength < 0) {
                return this; // no change
            }
            return new Configuration(this.dim, this.hiddenDim, this.numberOfLayers, this.numberOfHeads, this.numberOfKeyValueHeads, this.vocabularySize, newContextLength, this.rmsNormEps, this.ropeTheta);
        }

        public Configuration(int dim, int hiddenDim, int numberOfLayers, int numberOfHeads, int numberOfKeyValueHeads, int vocabularySize, int contextLength, float rmsNormEps, float ropeTheta) {
            this.dim = dim;
            this.hiddenDim = hiddenDim;
            this.numberOfLayers = numberOfLayers;
            this.numberOfHeads = numberOfHeads;
            this.numberOfKeyValueHeads = numberOfKeyValueHeads;
            this.vocabularySize = vocabularySize;
            this.contextLength = contextLength;
            this.rmsNormEps = rmsNormEps;
            this.ropeTheta = ropeTheta;
            this.headSize = dim / numberOfHeads;
        }
    }

    public static final class Weights {
        // token embedding table
        public final FloatTensor token_embedding_table; // (vocab_size, dim)
        // weights for rmsnorms
        public final FloatBuffer[] rms_att_weight; // (layer, dim) rmsnorm weights
        // weights for matmuls
        public final FloatTensor[] wq; // (layer, n_heads * head_size)
        public final FloatTensor[] wk; // (layer, n_kv_heads, head_size)
        public final FloatTensor[] wv; // (layer, n_kv_heads * head_size)
        public final FloatTensor[] wo; // (layer, n_heads * head_size, dim)
        
        // next 3: qwen - Groff from Qwen2.java
        public FloatTensor[] q_bias = null; // (layer, dim)
        public FloatTensor[] k_bias = null; // (layer, kv_dim)
        public FloatTensor[] v_bias = null; // (layer, kv_dim)
        
        public final FloatBuffer[] rms_ffn_weight; // (layer, dim)
        // weights for ffn
        public final FloatTensor[] w1; // (layer, hidden_dim, dim)
        public final FloatTensor[] w2; // (layer, dim, hidden_dim)
        public final FloatTensor[] w3; // (layer, hidden_dim, dim)
        // public final rmsnorm
        public final FloatBuffer rms_final_weight; // (dim,)
        // freq_cis for RoPE relatively positional embeddings
        public final FloatBuffer freq_cis_real; // (seq_len, head_size/2)
        public final FloatBuffer freq_cis_imag; // (seq_len, head_size/2)
        // (optional) classifier weights for the logits, on the last layer
        public final FloatTensor wcls; // (vocab_size, dim)

        public Weights(FloatTensor token_embedding_table, FloatBuffer[] rms_att_weight, FloatTensor[] wq, FloatTensor[] wk, FloatTensor[] wv, FloatTensor[] wo, FloatBuffer[] rms_ffn_weight, FloatTensor[] w1, FloatTensor[] w2, FloatTensor[] w3, FloatBuffer rms_final_weight, FloatBuffer freq_cis_real, FloatBuffer freq_cis_imag, FloatTensor wcls) {
            this.token_embedding_table = token_embedding_table;
            this.rms_att_weight = rms_att_weight;
            this.wq = wq;
            this.wk = wk;
            this.wv = wv;
            this.wo = wo;
            this.rms_ffn_weight = rms_ffn_weight;
            this.w1 = w1;
            this.w2 = w2;
            this.w3 = w3;
            this.rms_final_weight = rms_final_weight;
            this.freq_cis_real = freq_cis_real;
            this.freq_cis_imag = freq_cis_imag;
            this.wcls = wcls;
        }

        public Weights(FloatTensor token_embedding_table, FloatBuffer[] rms_att_weight, FloatTensor[] wq, FloatTensor[] wk, FloatTensor[] wv, FloatTensor[] wo, FloatTensor[] q, FloatTensor[] k, FloatTensor[] v, FloatBuffer[] rms_ffn_weight, FloatTensor[] w1, FloatTensor[] w2, FloatTensor[] w3, FloatBuffer rms_final_weight, FloatBuffer freq_cis_real, FloatBuffer freq_cis_imag, FloatTensor wcls) {
        	this.token_embedding_table = token_embedding_table;
        	this.rms_att_weight = rms_att_weight;
        	this.wq = wq;
        	this.wk = wk;
        	this.wv = wv;
        	this.wo = wo;
        	this.q_bias = q;
        	this.k_bias = k;
        	this.v_bias = v;
        	this.rms_ffn_weight = rms_ffn_weight;
        	this.w1 = w1;
        	this.w2 = w2;
        	this.w3 = w3;
        	this.rms_final_weight = rms_final_weight;
        	this.freq_cis_real = freq_cis_real;
        	this.freq_cis_imag = freq_cis_imag;
        	this.wcls = wcls;
        }
    }

    public static final class State {

        // current wave of activations
        public final int batchsize;
        public final FloatTensor[] x; // activation at current time stamp (dim,)
        public final FloatTensor[] xb; // same, but inside a residual branch (dim,)
        public final FloatTensor[] xb2; // an additional buffer just for convenience (dim,)
        public final FloatTensor[] hb; // buffer for hidden dimension in the ffn (hidden_dim,)
        public final FloatTensor[] hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
        public final FloatTensor[] q; // query (dim,)
        public final FloatTensor[] k; // key (dim,)
        public final FloatTensor[] v; // value (dim,)
        public final FloatTensor[] att; // buffer for scores/attention values (n_heads, seq_len)
        public final FloatTensor logits; // output logits

        // kv cache
        public final FloatTensor[] keyCache;   // (n_layer, seq_len, kv_dim)
        public final FloatTensor[] valueCache; // (n_layer, seq_len, kv_dim)
        
        /** last index in previous block */
        int idxPrevBlock;

        public int latestToken;

        State(Configuration config, int batchsize) {
            this.batchsize = batchsize;
            this.x = allocate(batchsize, config.dim);
            this.xb = allocate(batchsize, config.dim);
            this.xb2 = allocate(batchsize, config.dim);
            this.hb = allocate(batchsize, config.hiddenDim);
            this.hb2 = allocate(batchsize, config.hiddenDim);
            this.q = allocate(batchsize, config.dim);
            this.k = allocate(batchsize, config.dim);
            this.v = allocate(batchsize, config.dim);
            this.att = allocate(batchsize, config.numberOfHeads, config.contextLength);
            idxPrevBlock = -1;

            this.logits = ArrayFloatTensor.allocate(config.vocabularySize);
            int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;
            this.keyCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength, kvDim)).limit(config.numberOfLayers).toArray(FloatTensor[]::new);
            this.valueCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength, kvDim)).limit(config.numberOfLayers).toArray(FloatTensor[]::new);
        }
    }

    static FloatTensor[] allocate(int numTokens, int... dims) {
        return IntStream.range(0, numTokens)
                .mapToObj(i -> ArrayFloatTensor.allocate(dims))
                .toArray(FloatTensor[]::new);
    }

    static void rmsnorm(FloatTensor out, FloatTensor x, FloatBuffer weight, int size, float rmsNormEps) {
        // calculate sum of squares
        float ss = x.reduce(0, size, 0f, (acc, xi) -> acc + xi * xi);
        ss /= size;
        ss += rmsNormEps;
        ss = (float) (1.0 / Math.sqrt(ss));
        // normalize and scale
        final float finalss = ss; // for the lambda
        out.mapWithIndexInPlace(0, size, (value, index) -> weight.get(index) * (finalss * x.getFloat(index)));
    }
    
    static FloatTensor forward(Llama model, State state, int[] tokens, int position, boolean computeLogits) {
        // a few convenience variables
        Configuration config = model.configuration();
        Weights weights = model.weights();
        int dim = config.dim;
        int headSize = config.headSize;
        int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;
        int kvMul = config.numberOfHeads / config.numberOfKeyValueHeads; // integer multiplier of the kv sharing in multiquery
        float sqrtHeadSize = (float) Math.sqrt(headSize);
        final int nTokens = tokens.length;

        // copy the token embedding into x
        Parallel.parallelFor(0, nTokens, t ->
            weights.token_embedding_table.copyTo(tokens[t] * dim, state.x[t], 0, dim)
        );

        // forward all the layers
        for (int l = 0; l < config.numberOfLayers; l++) {
            // attention rmsnorm
            // rmsnorm(state.xb, state.x, weights.rms_att_weight[l], dim, config.rmsNormEps);
            final int curLayer = l;
            Parallel.parallelFor(0, nTokens, t ->
                rmsnorm(state.xb[t], state.x[t], weights.rms_att_weight[curLayer], dim, config.rmsNormEps)
            );
    		//try (Timer timer = Timer.log("qkv matmuls layer:"+l,TimeUnit.MICROSECONDS)) {
            // qkv matmuls for this position
            weights.wq[l].matmul(nTokens, state.xb, state.q, dim, dim);
            weights.wk[l].matmul(nTokens, state.xb, state.k, kvDim, dim);
            weights.wv[l].matmul(nTokens, state.xb, state.v, kvDim, dim);
    		//}
    		//try (Timer timer = Timer.log("RoPe layer:"+l,TimeUnit.MICROSECONDS)) {
            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            Parallel.parallelFor(0, nTokens, t -> {
                for (int i = 0; i < dim; i += 2) {
                    int head_dim = i % headSize;
                    float fcr = weights.freq_cis_real.get((position + t) * (headSize / 2) + (head_dim / 2));
                    float fci = weights.freq_cis_imag.get((position + t) * (headSize / 2) + (head_dim / 2));
                    int rotn = i < kvDim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                    for (int vi = 0; vi < rotn; vi++) {
                        FloatTensor vec = vi == 0 ? state.q[t] : state.k[t]; // the vector to rotate (query or key)
                        float v0 = vec.getFloat(i);
                        float v1 = vec.getFloat(i + 1);
                        vec.setFloat(i, v0 * fcr - v1 * fci);
                        vec.setFloat(i + 1, v0 * fci + v1 * fcr);
                    }
                }
            });
    		//}
    		//try (Timer timer = Timer.log("Save kv layer:"+l,TimeUnit.MICROSECONDS)) {
            // save key,value at this time step (position) to our kv cache
            //int loff = l * config.seq_len * kvDim; // kv cache layer offset for convenience
            Parallel.parallelFor(0, nTokens, t -> {
                state.k[t].copyTo(0, state.keyCache[curLayer], (position + t) * kvDim, kvDim);
                state.v[t].copyTo(0, state.valueCache[curLayer], (position + t) * kvDim, kvDim);
            });
    		//}
            // If the logits are not required, the attention and FFN of the last layer can be skipped entirely.
            if (!computeLogits && curLayer == config.numberOfLayers - 1) {
                state.idxPrevBlock = nTokens - 1;
                return null;
            }
            //
            //----- GPU multihead attention. iterate over all heads 
            //List<CompletableFuture<Void>> gpuQueue = new ArrayList<>();
            /*int totalHeads = nTokens * config.numberOfHeads;
            int hiddenSize = 4096;                  // from model config
            int H = state.q.length;                 // number of heads
            int d = hiddenSize / H;                 // per-head width (e.g. 4096/32 = 128)

            // infer how many rows (tokens) are in each head tensor
            int rowsQ = state.q[0].size() / d;
            int rowsK = state.k[0].size() / d;
            int rowsV = state.v[0].size() / d;
            // for a single-step test, you’ll see rowsQ = rowsK = rowsV = 1
            int Tq = 512;//rowsQ;
            int Tk = 512;//rowsK;
            if(DEBUG )
            	System.out.printf("Attn: Tq=%d, Tk=%d, d=%d, H=%d%n", Tq, Tk, d, H);
            Attn attn = new Attn(Llama3.cublasHandle, Tq, Tk, d, H);
            try {
                attn.packHeads(state.q, state.k, state.v);
                int rc = attn.attention();
                if (rc != 0) {
                    throw new RuntimeException("attentionBatchedHeads rc=" + rc);
                }
                attn.unpackHeads(state.xb);
            } finally {
                attn.close();
            }
            //float[] f = new float[state.xb[0].size()];
            //state.xb[0].exportSlice(f, 0, 0, state.xb[0].size());
            //System.out.println("layer:"+l+"="+Arrays.toString(f));
            */
            
            // original multihead attention. iterate over all heads
      		//try (Timer timer = Timer.log("CPU Multihead Attn layer:"+l,TimeUnit.MICROSECONDS)) {
            Parallel.parallelForLong(0, (long) nTokens * (long) config.numberOfHeads, ht -> {
                int token = (int) (ht / config.numberOfHeads);
                int h = (int) (ht % config.numberOfHeads);
                // get the query vector for this head
                // float* q = s.q + h * headSize;
                int qOffset = h * headSize;
                // attention scores for this head
                // float* att = s.att + h * config.seq_len;
                int attOffset = h * config.contextLength;
                // iterate over all timesteps, including the current one
               for (int t = 0; t <= position + token; t++) {
                    // get the key vector for this head and at this timestep
                    // float* k = s.key_cache + loff + t * dim + h * headSize;
                    int keyCacheOffset = t * kvDim + (h / kvMul) * headSize;
                    // calculate the attention score as the dot product of q and k
                    float score = state.q[token].dot(Llama3.cublasHandle[(int)ht], qOffset, state.keyCache[curLayer], keyCacheOffset, headSize);
                    score /= sqrtHeadSize;
                    // save the score to the attention buffer
                    state.att[token].setFloat(attOffset + t, score);
                }
                // softmax the scores to get attention weights, from 0..position inclusively
                state.att[token].softmaxInPlace(attOffset, position + token + 1);
                // weighted sum of the values, store back into xb
                // float* xb = s.xb + h * headSize;
                int xbOffset = h * headSize;
                // memset(xb, 0, headSize * sizeof(float));
                state.xb[token].fillInPlace(xbOffset, headSize, 0f);
                for (int t = 0; t <= position + token; t++) {
                    // get the value vector for this head and at this timestep
                    // float* v = s.value_cache + loff + t * dim + h * headSize;
                    int vOffset = t * kvDim + (h / kvMul) * headSize;
                    // get the attention weight for this timestep
                    float a = state.att[token].getFloat(attOffset + t);
                    // accumulate the weighted value into xb
                    state.xb[token].saxpyInPlace(xbOffset, state.valueCache[curLayer], vOffset, headSize, a);
                }
            });
      		//}
            //----end original multihead attention
            
            /*---attempted bulk processing attention
            int heads = config.numberOfHeads;
            int d = headSize;
            final int H = heads;
            final int D = d;
            final int TQ = nTokens;
            final int TK = position + nTokens; // fixed context horizon for this batch
            // Prepare inputs (fill with your model-projected Q/K/V)
            float[] Q = new float[H * TQ * D];
            float[] K = new float[H * TK * D];
            float[] V = new float[H * TK * D];
            // Pack Q: per token, per head
            for (int t = 0; t < TQ; t++) {
            	for (int h = 0; h < H; h++) {
            		int qOffset = h * headSize;
            		float[] qVec = state.q[t].exportSlicePooled(Llama3.poolHead, qOffset, headSize);
            		int dst = h * (TQ * D) + t * D;
            		System.arraycopy(qVec, 0, Q, dst, D);
            	}
            }
            // Pack K/V from cache across time 0..TK-1 (or your window start..end)
            for(int u = 0; u < TK; u++) {
            	for(int h = 0; h < H; h++) {
            		int kvHeadBase = (h / kvMul) * headSize; // shared KV head group
            		int keyCacheOffset = u * kvDim + kvHeadBase;
            		float[] kVec = state.keyCache[curLayer].exportSlicePooled(Llama3.poolHead, keyCacheOffset, headSize);
            		float[] vVec = state.valueCache[curLayer].exportSlicePooled(Llama3.poolHead, keyCacheOffset, headSize);
            		int kd = h * (TK * D) + u * D;
            		System.arraycopy(kVec, 0, K, kd, D);
            		System.arraycopy(vVec, 0, V, kd, D);
            	}
            }
            Attn ctx = new Attn(Llama3.cublasHandle, 1, heads, TQ, TK, d);
            AttentionRunner.Config cfg = new AttentionRunner.Config(heads, d, TQ, TK);
            float[] O = AttentionRunner.run(Llama3.cublasHandle, ctx, cfg, Q, K, V);
  
            // Scatter O back into state.xb per token
            for(int t = 0; t < TQ; t++) {
            	int base = t * heads * d;
            	int stride = heads * d;
            	for(int j = 0; j < stride; j++) {
            	   state.xb[t].setFloat(j, O[base + j]);
            	}
            }
            ctx.close();
            
            // Continue with wo, residual, FFN...
            if(Llama3.DEBUG) {
            		long[] mem = Gemm.cudaMemGetInfo();
            		System.out.println(Thread.currentThread().getName()+" queries="+(Q != null ? Q.length : " queries null!")+
            			" keys="+(K != null ? K.length : " keys null!")+
            			" values="+(V != null ? V.length : " values null!")+" headSize="+headSize+" mem free:"+mem[0]+" total:"+mem[1]);
            }
    		System.out.println(">>>Layer:"+l+" queries="+(Q != null ? Q.length : " queries null!")+
        			" keys="+(K != null ? K.length : " keys null!")+
        			" values="+(V != null ? V.length : " values null!")+" headSize="+headSize);
        	---end attempted bulk multihead attention*/
            
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
            
   
            // CUBLAS version of above parallel loop
            /*
            Parallel.parallelForLong(0, (long) nTokens * (long) config.numberOfHeads, ht -> {
                final int token = (int) (ht / config.numberOfHeads);
                final int h     = (int) (ht % config.numberOfHeads);
                // Offsets for this head
                final int qOffset   = h * headSize;
                final int attOffset = h * config.contextLength;
                // Time horizon for this token (inclusive of current position)
                final int T = position + token + 1;
                final ArrayList<float[]> queries = new ArrayList<>(T);
                final ArrayList<float[]> keys    = new ArrayList<>(T);
                final ArrayList<float[]> results = new ArrayList<>(T);

                final float[] qVec = state.q[token].exportSlicePooled(Llama3.poolHead, qOffset, headSize);
                for (int t = 0; t < T; t++) {
                    final int keyCacheOffset = t * kvDim + (h / kvMul) * headSize;
                    final float[] kVec = state.keyCache[curLayer].exportSlicePooled(Llama3.poolHead, keyCacheOffset, headSize);
                    queries.add(qVec);
                    keys.add(kVec);
                    results.add(Llama3.poolScalar.acquire(1));
                }
                if(Llama3.DEBUG) {
                	long[] mem = Gemm.cudaMemGetInfo();
                	System.out.println(Thread.currentThread().getName()+" queries="+(queries != null ? queries.size() : " queries null!")+
                			" keys="+(keys != null ? keys.size() : " keys null!")+
                			" results="+(results != null ? results.size() : " results null!")+" headSize="+headSize+" resultSize="+T+" mem free:"+mem[0]+" total:"+mem[1]);
                }
                AtomicInteger retCode = new AtomicInteger();
                retCode.set(Gemm.matrixDotProductF16Batch(Llama3.cublasHandle, 1, headSize, queries, headSize, 1, keys, results, T));
                if(retCode.get() != 0) {
    				throw new RuntimeException("matrixDotProductFBatch returned JNI error code"+retCode.get());
    			}
                // 3) Write attention scores and apply scaling
                for (int t = 0; t < T; t++) {
                	//System.out.println("Results "+t+"+.)="+Arrays.toString(results.get(t)));
                    final float score = results.get(t)[0] / sqrtHeadSize;
                    state.att[token].setFloat(attOffset + t, score);
                }
                state.att[token].softmaxInPlace(attOffset, T);
                // 4) Weighted sum over values → xb
                final int xbOffset = h * headSize;
                state.xb[token].fillInPlace(xbOffset, headSize, 0f);
                for (int t = 0; t < T; t++) {
                    final int vOffset = t * kvDim + (h / kvMul) * headSize;
                    final float a = state.att[token].getFloat(attOffset + t);
                    state.xb[token].saxpyInPlace(xbOffset, state.valueCache[curLayer], vOffset, headSize, a);
                }
                Llama3.poolHead.release(qVec);
                for (float[] k : keys) Llama3.poolHead.release(k);
                for (float[] r : results) Llama3.poolScalar.release(r);
            });
            //---end CUBLAS version*/
      		
            // CUBLAS version of above parallel loop with bigger batch 
            /*AtomicInteger TT = new AtomicInteger(0);
            final List<float[]> queries = Collections.synchronizedList(new ArrayList<float[]>());
            final List<float[]> keys    = Collections.synchronizedList(new ArrayList<float[]>());
            final List<float[]> results = Collections.synchronizedList(new ArrayList<float[]>());
            Parallel.parallelForLong(0, (long) nTokens * (long) config.numberOfHeads, ht -> {
                final int token = (int) (ht / config.numberOfHeads);
                final int h     = (int) (ht % config.numberOfHeads);
                // Offsets for this head
                final int qOffset   = h * headSize;
                final int attOffset = h * config.contextLength;
                // Time horizon for this token (inclusive of current position)
                final int T = position + token + 1;
                TT.getAndAdd(T);
                final float[] qVec = state.q[token].exportSlicePooled(Llama3.poolHead, qOffset, headSize);
                for (int t = 0; t < T; t++) {
                    final int keyCacheOffset = t * kvDim + (h / kvMul) * headSize;
                    final float[] kVec = state.keyCache[curLayer].exportSlicePooled(Llama3.poolHead, keyCacheOffset, headSize);
                    queries.add(qVec);
                    keys.add(kVec);
                    results.add(Llama3.poolScalar.acquire(1));
                }
                if(Llama3.DEBUG) {
                	long[] mem = Gemm.cudaMemGetInfo();
                	System.out.println(Thread.currentThread().getName()+" queries="+(queries != null ? queries.size() : " queries null!")+
                			" keys="+(keys != null ? keys.size() : " keys null!")+
                			" results="+(results != null ? results.size() : " results null!")+" headSize="+headSize+" resultSize="+T+" mem free:"+mem[0]+" total:"+mem[1]);
                }
            });
            AtomicInteger retCode = new AtomicInteger();
            FloatBuffer fb1 = FloatBuffer.;  
            ArrayList<float[]> ql = new ArrayList<>(queries);
            ArrayList<float[]> kl = new ArrayList<>(keys);
            ArrayList<float[]> rl = new ArrayList<>(results);
            //System.out.println("Q="+ql.size()+" K="+kl.size()+" R="+rl.size());
            retCode.set(Gemm.matrixDotProductF(Llama3.cublasHandle, 1, headSize, ql, headSize, 1, kl, rl));
            if(retCode.get() != 0) {
            	throw new RuntimeException("matrixDotProductFBatch returned JNI error code"+retCode.get());
            }
            for(float[] q : queries) Llama3.poolHead.release(q);
            for(float[] k : keys) Llama3.poolHead.release(k);
            //System.out.println("TT = "+TT.get());
            for(float[] rll: rl)
            	System.out.println(Arrays.toString(rll));
            List<float[]> rl2 = Collections.synchronizedList(rl);
            //Parallel.parallelForLong(0, (long) nTokens * (long) config.numberOfHeads, ht -> {
            for(long ht = 0; ht < (long) nTokens * (long) config.numberOfHeads; ht++) {
                final int token = (int) (ht / config.numberOfHeads);
                final int h     = (int) (ht % config.numberOfHeads);
                // Offsets for this head
                final int attOffset = h * config.contextLength;
                // Time horizon for this token (inclusive of current position)
                final int T = position + token + 1;
                // 3) Write attention scores and apply scaling
                for (int t = 0; t < T; t++) {
                	//System.out.println("Results "+t+"+.)="+Arrays.toString(results.get(t)));
                    final float score = rl2.get(t)[0] / sqrtHeadSize;
                    state.att[token].setFloat(attOffset + t, score);
                }
                state.att[token].softmaxInPlace(attOffset, T);
                // 4) Weighted sum over values → xb
                final int xbOffset = h * headSize;
                state.xb[token].fillInPlace(xbOffset, headSize, 0f);
                for (int t = 0; t < T; t++) {
                    final int vOffset = t * kvDim + (h / kvMul) * headSize;
                    final float a = state.att[token].getFloat(attOffset + t);
                    state.xb[token].saxpyInPlace(xbOffset, state.valueCache[curLayer], vOffset, headSize, a);
                }
            //});
            }
            for(float[] r : results) Llama3.poolScalar.release(r);
            //---end CUBLAS version with bigger batch*/
      		
    		//try (Timer timer = Timer.log("Final matmul and residual connection layer:"+l,TimeUnit.MICROSECONDS)) {
            // final matmul to get the output of the attention
            weights.wo[l].matmul(nTokens, state.xb, state.xb2, dim, dim);
            // residual connection back into x
            Parallel.parallelFor(0, nTokens, t -> {
                state.x[t].addInPlace(state.xb2[t]);
            });
    		//}
    		//try (Timer timer = Timer.log("FFN RMSNorm layer:"+l,TimeUnit.MICROSECONDS)) {
            // ffn rmsnorm
            Parallel.parallelFor(0, nTokens, t -> {
                rmsnorm(state.xb[t], state.x[t], weights.rms_ffn_weight[curLayer], dim, config.rmsNormEps);
            });
    		//}
    		//try (Timer timer = Timer.log("SwiGLU non-linearity  and final conns layer:"+l,TimeUnit.MICROSECONDS)) {
            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            weights.w1[l].matmul(nTokens, state.xb, state.hb, config.hiddenDim, dim);
            weights.w3[l].matmul(nTokens, state.xb, state.hb2, config.hiddenDim, dim);
            // SwiGLU non-linearity
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            Parallel.parallelFor(0, nTokens, t -> {
                state.hb[t].mapInPlace(value -> value / (float) (1.0 + Math.exp(-value)));
            });
            // elementwise multiply with w3(x)
            Parallel.parallelFor(0, nTokens, t -> {
                state.hb[t].multiplyInPlace(state.hb2[t]);
            });
            // final matmul to get the output of the ffn
            weights.w2[l].matmul(nTokens, state.hb, state.xb, dim, config.hiddenDim);
            // residual connection
            Parallel.parallelFor(0, nTokens, t -> {
                state.x[t].addInPlace(state.xb[t]);
            });
    		//}
        }
        //System.out.println("<<<END LAYER LOOP");
        // final rmsnorm
        Parallel.parallelFor(0, nTokens, t -> {
            rmsnorm(state.x[t], state.x[t], weights.rms_final_weight, dim, config.rmsNormEps);
        });
        
        if(false) {
        	try (Timer timer = Timer.log("Last vector:"+state.x[nTokens-1].size())) {
        	}
        	try (Timer timer = Timer.log("Signature")) {
        		
        	}
        	try (Timer timer = Timer.log("Store Tensor:")) {
        		Llama3.dbClient.storekv(Llama3.tensorAlias, Llama3.xid, "index" , state.x[nTokens-1]);
        	}
        }
        
        // classifier into logits
        weights.wcls.matmul(state.x[nTokens - 1], state.logits, config.vocabularySize, dim);
        state.idxPrevBlock = nTokens - 1;

        return state.logits;
    }
    
    static FloatTensor forwardQwen(Llama model, State state, int token, int position) {
    	// a few convenience variables
    	Llama.Configuration config = model.configuration();
    	Llama.Weights weights = model.weights();
    	int dim = config.dim;
    	int headSize = config.headSize;
    	int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;
    	int kvMul = config.numberOfHeads / config.numberOfKeyValueHeads; // integer multiplier of the kv sharing in multiquery
    	float sqrtHeadSize = (float) Math.sqrt(headSize);

    	// copy the token embedding into x
    	weights.token_embedding_table.copyTo(token * dim, state.x[0], 0, dim);

    	// forward all the layers
    	for (int l = 0; l < config.numberOfLayers; l++) {
    		// attention rmsnorm
    		rmsnorm(state.xb[0], state.x[0], weights.rms_att_weight[l], dim, config.rmsNormEps);
    		// qkv matmuls for this position
    		weights.wq[l].matmul(state.xb[0], state.q[0], dim, dim);
    		if (weights.q_bias != null && weights.q_bias[l] != null) {
    			//state.q[0].addInPlace(weights.q_bias[l]);
    			if(Llama3.DEBUG) {
    				System.out.println("state:"+state.q[0].size());
    				state.q[0].verify();
    				System.out.println("weights:"+weights.q_bias[l].size());
    				weights.q_bias[l].verify();
    			}
       			state.q[0].addInPlace(weights.q_bias[l]);
    		}
    		weights.wk[l].matmul(state.xb[0], state.k[0], kvDim, dim);
    		if (weights.k_bias != null && weights.k_bias[l] != null) {
    			state.k[0].addInPlace(weights.k_bias[l]);
    		}
    		weights.wv[l].matmul(state.xb[0], state.v[0], kvDim, dim);
    		if (weights.v_bias != null && weights.v_bias[l] != null) {
    			state.v[0].addInPlace(weights.v_bias[l]);
    		}
    		// RoPE relative positional encoding: complex-valued rotate q and k in each head
    		// GPT-NeoX style RoPE, real/imaginary components are stored with a headSize/2 offset per head, instead of consecutive.
    		for (int h = 0; h < config.numberOfHeads; ++h) {
    			int rotn = h < config.numberOfKeyValueHeads ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
    			int poffset = h * headSize;
    			for (int i0 = 0; i0 < headSize; i0 += 2) {
    				int ic = i0 / 2;
    				float fcr = weights.freq_cis_real.get(position * (headSize / 2) + ic);
    				float fci = weights.freq_cis_imag.get(position * (headSize / 2) + ic);
    				for (int v = 0; v < rotn; v++) {
    					FloatTensor vec = v == 0 ? state.q[0] : state.k[0]; // the vector to rotate (query or key)
    					float v0 = vec.getFloat(poffset + ic);
    					float v1 = vec.getFloat(poffset + ic + headSize/2);
    					vec.setFloat(poffset + ic, v0 * fcr - v1 * fci);
    					vec.setFloat(poffset + ic + headSize/2, v0 * fci + v1 * fcr);
    				}
    			}
    		}
    		// save key,value at this time step (position) to our kv cache
    		//int loff = l * config.seq_len * kvDim; // kv cache layer offset for convenience
    		state.k[0].copyTo(0, state.keyCache[l], position * kvDim, kvDim);
    		state.v[0].copyTo(0, state.valueCache[l], position * kvDim, kvDim);
    		int curLayer = l;
    		// multihead attention. iterate over all heads
    		Parallel.parallelFor(0, config.numberOfHeads, h -> {
    			// get the query vector for this head
    			// float* q = s.q + h * headSize;
    			int qOffset = h * headSize;
    			// attention scores for this head
    			// float* att = s.att + h * config.seq_len;
    			int attOffset = h * config.contextLength;
    			// iterate over all timesteps, including the current one
    			for (int t = 0; t <= position; t++) {
    				// get the key vector for this head and at this timestep
    				// float* k = s.key_cache + loff + t * dim + h * headSize;
    				int keyCacheOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
    				// calculate the attention score as the dot product of q and k
    				float score = state.q[0].dot(Llama3.cublasHandle[(int)h], qOffset, state.keyCache[curLayer], keyCacheOffset, headSize);
    				score /= sqrtHeadSize;
    				// save the score to the attention buffer
    				state.att[0].setFloat(attOffset + t, score);
    			}
    			// softmax the scores to get attention weights, from 0..position inclusively
    			state.att[0].softmaxInPlace(attOffset, position + 1);
    			// weighted sum of the values, store back into xb
    			// float* xb = s.xb + h * headSize;
    			int xbOffset = h * headSize;
    			// memset(xb, 0, headSize * sizeof(float));
    			state.xb[0].fillInPlace(xbOffset, headSize, 0f);
    			for (int t = 0; t <= position; t++) {
    				// get the value vector for this head and at this timestep
    				// float* v = s.value_cache + loff + t * dim + h * headSize;
    				int vOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
    				// get the attention weight for this timestep
    				float a = state.att[0].getFloat(attOffset + t);
    				// accumulate the weighted value into xb
    				state.xb[0].saxpyInPlace(xbOffset, state.valueCache[curLayer], vOffset, headSize, a);
    			}
    		});

    		// final matmul to get the output of the attention
    		weights.wo[l].matmul(state.xb[0], state.xb2[0], dim, dim);
    		// residual connection back into x
    		state.x[0].addInPlace(state.xb2[0]);
    		// ffn rmsnorm
    		rmsnorm(state.xb[0], state.x[0], weights.rms_ffn_weight[l], dim, config.rmsNormEps);
    		// Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    		// first calculate self.w1(x) and self.w3(x)
    		weights.w1[l].matmul(state.xb[0], state.hb[0], config.hiddenDim, dim);
    		weights.w3[l].matmul(state.xb[0], state.hb2[0], config.hiddenDim, dim);
    		// SwiGLU non-linearity
    		// silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
    		state.hb[0].mapInPlace(value -> value / (float) (1.0 + Math.exp(-value)));
    		// elementwise multiply with w3(x)
    		state.hb[0].multiplyInPlace(state.hb2[0]);
    		// final matmul to get the output of the ffn
    		weights.w2[l].matmul(state.hb[0], state.xb[0], dim, config.hiddenDim);
    		// residual connection
    		state.x[0].addInPlace(state.xb[0]);
    	}

    	// final rmsnorm
    	rmsnorm(state.x[0], state.x[0], weights.rms_final_weight, dim, config.rmsNormEps);
    	// classifier into logits
    	weights.wcls.matmul(state.x[0], state.logits, config.vocabularySize, dim);
    	return state.logits;
    }

    /**
     * LLM generation entry point, ingest prompt tokens and generates new tokens.
     *
     * <p>
     * All prompt tokens are ingested first, then inference starts, until a stop token is found.
     * The returned tokens only include generated/inferred tokens.
     *
     * @param model            model to run inference (including weights, configuration, tokenizer ...)
     * @param state            state of the model e.g. key/value caches ... this is mutated by this call
     * @param startPosition    start prompt ingestion + inference at this position in the context e.g. useful if state was kept across calls (chained generation). 0 implies run with no previous context.
     * @param promptTokens     prompt tokens to ingest, all the prompt tokens will be ingested, given there's enough capacity left in the context
     * @param stopTokens       set of tokens that abort generation during inference, stop tokens do not affect prompt ingestion
     * @param maxTokens        maximum number of tokens (can go up to {@link Configuration#contextLength context length}
     *                         if this value is negative or greater than {@link Configuration#contextLength context length}
     * @param sampler          {@link Sampler strategy} used to select tokens
     * @param echo             debugging flag, prints ALL, prompt and inferred tokens, to {@link System#err stderr}
     * @param onTokenGenerated callback, if non-null, it's called every time a token is inferred e.g. it's not called when ingesting prompt tokens
     * @return list of generated/inferred tokens, including the stop token, if any e.g. does not include any token from the prompt
     */
    public static List<Integer> generateTokens(Llama model, State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
                                               IntConsumer onTokenGenerated) {
        long startNanos = System.nanoTime();
        long startGen = 0;
        if (maxTokens < 0 || model.configuration().contextLength < maxTokens) {
            maxTokens = model.configuration().contextLength;
        }
        List<Integer> generatedTokens = new ArrayList<>(maxTokens);
        int token = state.latestToken; // BOS?
        int nextToken;
        int promptIndex = 0;
        for (int position = startPosition; position < maxTokens; ++position) {
            if (promptIndex < promptTokens.size()) {
                final int nTokens = Math.min(maxTokens - position, Math.min(promptTokens.size() - promptIndex, state.batchsize));
                final int[] tokens = new int[nTokens];
                for (int i = 0; i < nTokens; i++) {
                    tokens[i] = promptTokens.get(promptIndex + i);
                    if (echo) {
                        // log prompt token (different color?)
                        System.err.print(Tokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(tokens[i]))));
                    }
                }
                if (echo) {
                    System.out.format("position=%d, promptIdx=%d, promptSize=%d, tokens=%s%n", position, promptIndex, promptTokens.size(), Arrays.toString(tokens));
                }
                // Only compute logits on the very last batch.
                boolean computeLogits = promptIndex + nTokens >= promptTokens.size();
                forward(model, state, tokens, position, computeLogits);
                position += nTokens - 1; // -1 -> incremented later in the for loop
                promptIndex += nTokens;
                if (promptIndex < promptTokens.size()) {
                    continue;
                }
                startGen = System.nanoTime();
            } else {
                forward(model, state, new int[]{token}, position, true);
            }
            nextToken = sampler.sampleToken(state.logits);
            if (echo) {
                // log inferred token
                System.err.print(Tokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(nextToken))));
            }
            generatedTokens.add(nextToken);
            if (onTokenGenerated != null) {
                onTokenGenerated.accept(nextToken);
            }
            if (stopTokens.contains(nextToken)) {
                break;
            }
            state.latestToken = token = nextToken;
        }

        long elapsedNanos = System.nanoTime() - startNanos;
        long promptNanos = startGen - startNanos;
        long genNanos = elapsedNanos - startGen + startNanos;
        System.err.printf("%ncontext: %d/%d prompt: %.2f tokens/s (%d) generation: %.2f tokens/s (%d)%n",
                startPosition + promptIndex + generatedTokens.size(), model.configuration().contextLength,
                promptTokens.size() / (promptNanos / 1_000_000_000.0), promptTokens.size(),
                generatedTokens.size() / (genNanos / 1_000_000_000.0), generatedTokens.size());

        return generatedTokens;
    }

    /**
     * Qwen specific calls forwardQwen.
     * @param model
     * @param state
     * @param startPosition
     * @param promptTokens
     * @param stopTokens
     * @param maxTokens
     * @param sampler
     * @param echo
     * @param onTokenGenerated
     * @return
     */
    public static List<Integer> generateTokensQwen(Llama model, State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
    		IntConsumer onTokenGenerated) {
        long startNanos = System.nanoTime();
        if (maxTokens < 0 || model.configuration().contextLength < maxTokens) {
            maxTokens = model.configuration().contextLength;
        }
        List<Integer> generatedTokens = new ArrayList<>(maxTokens);
        int token = state.latestToken; // BOS?
        int nextToken;
        int promptIndex = 0;
        for (int position = startPosition; position < maxTokens; ++position) {
            forwardQwen(model, state, token, position);
            if (promptIndex < promptTokens.size()) {
                // Force-pick token from prompt.
                nextToken = promptTokens.get(promptIndex++);
                if (echo) {
                    // log prompt token (different color?)
                    System.err.print(Tokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(nextToken))));
                }
            } else {
                nextToken = sampler.sampleToken(state.logits);
                if (echo) {
                    // log inferred token
                    System.err.print(Tokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(nextToken))));
                }
                generatedTokens.add(nextToken);
                if (onTokenGenerated != null) {
                    onTokenGenerated.accept(nextToken);
                }
                if (stopTokens.contains(nextToken)) {
                    break;
                }
            }
            state.latestToken = token = nextToken;
        }

        long elapsedNanos = System.nanoTime() - startNanos;
        int totalTokens = promptIndex + generatedTokens.size();
        System.err.printf("%n%.2f tokens/s (%d)%n", totalTokens / (elapsedNanos / 1_000_000_000.0), totalTokens);

        return generatedTokens;
    }
    
}

interface TokenizerInterface {
	 public Map<String, Integer> getSpecialTokens();
	 public boolean isSpecialToken(int tokenIndex);
	 public String decode(List<Integer> tokens);
	 public List<Integer> encodeAsList(String text);
	 public int getTokenType(int tokenIndex);
	 public Collection<? extends Integer> encode(String text);
}
/**
 * Byte Pair Encoding tokenizer.
 * <p>
 * Based on <a href="https://github.com/karpathy/minbpe">minbpe</a>, algorithmically follows along the
 * <a href="https://github.com/openai/gpt-2/blob/master/src/encoder.py">GPT 2 tokenizer</a>
 */
class Tokenizer implements TokenizerInterface {
    private final Pattern compiledPattern;
    private final Vocabulary vocabulary;
    private final Map<Pair<Integer, Integer>, Integer> merges;
    private final Map<String, Integer> specialTokens;
    private int[] tokenTypes; // qwen2

    public String regexPattern() {
        if (compiledPattern == null) {
            return null;
        }
        return compiledPattern.pattern();
    }
    @Override
    public Map<String, Integer> getSpecialTokens() {
        return specialTokens;
    }
    @Override
    public boolean isSpecialToken(int tokenIndex) {
        return specialTokens.containsValue(tokenIndex);
    }
    @Override
    public int getTokenType(int tokenIndex) {
        return tokenTypes[tokenIndex];
    }
    
    public Tokenizer(Vocabulary vocabulary, List<Pair<Integer, Integer>> merges, String regexPattern, Map<String, Integer> specialTokens) {
        this.vocabulary = vocabulary;
        this.compiledPattern = regexPattern != null ? Pattern.compile(regexPattern) : null;
        this.specialTokens = new HashMap<>(specialTokens);
        this.merges = new HashMap<>();
        for (Pair<Integer, Integer> pair : merges) {
            int firstIndex = pair.first();
            int secondIndex = pair.second();
            int mergeIndex = vocabulary.getIndex(vocabulary.get(firstIndex) + vocabulary.get(secondIndex)).orElseThrow();
            this.merges.put(pair, mergeIndex);
        }
    }

    public Tokenizer(Vocabulary vocabulary, List<Pair<Integer, Integer>> merges, String regexPattern, Map<String, Integer> specialTokens, int[] tokenTypes) {
    	this(vocabulary, merges, regexPattern, specialTokens);
    	this.tokenTypes = tokenTypes;
    }
    
    private int[] encodeImpl(Collection<? extends Integer> intc) {
    	return intc.stream().mapToInt(i -> i).toArray();
    }

    /**
     * Unlike {@link #encodeOrdinary(String)}, this function handles special tokens.
     * allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
     * if none_raise, then an error is raised if any special token is encountered in text
     * this is the default tiktoken behavior right now as well
     * any other behavior is either annoying, or a major footgun.
     */
    List<Integer> encode(String text, Set<String> allowedSpecial) {
        // decode the user desire w.r.t. handling of special tokens
        Set<String> special = allowedSpecial;
        assert getSpecialTokens().keySet().containsAll(special);
        if (special.isEmpty()) {
            // shortcut: if no special tokens, just use the ordinary encoding
            return encodeOrdinary(text);
        }

        // otherwise, we have to be careful with potential special tokens in text
        // we handle special tokens by splitting the text
        // based on the occurrence of any exact match with any of the special tokens
        // we can use re.split for this. note that surrounding the pattern with ()
        // makes it into a capturing group, so the special tokens will be included
        String specialPattern = special
                .stream()
                .map(Pattern::quote)
                .collect(Collectors.joining("|", "(", ")"));

        String[] specialChunks = text.split(specialPattern);
        // now all the special characters are separated from the rest of the text
        // all chunks of text are encoded separately, then results are joined
        List<Integer> ids = new ArrayList<>();
        for (String part : specialChunks) {
            if (special.contains(part)) {
                // this is a special token, encode it separately as a special case
                ids.add(getSpecialTokens().get(part));
            } else {
                // this is an ordinary sequence, encode it normally
                ids.addAll(encodeOrdinary(part));
            }
        }
        return ids;
    }

    private static List<String> findAll(Pattern pattern, String text) {
        List<String> allMatches = new ArrayList<>();
        Matcher matcher = pattern.matcher(text);
        while (matcher.find()) {
            allMatches.add(matcher.group());
        }
        return allMatches;
    }

    /**
     * Encoding that ignores any special tokens.
     */
    public List<Integer> encodeOrdinary(String text) {
        // split text into chunks of text by categories defined in regex pattern
        List<String> textChunks = findAll(compiledPattern, text);
        // all chunks of text are encoded separately, then results are joined
        List<Integer> ids = new ArrayList<>();
        for (String chunk : textChunks) {
            List<Integer> chunkIds = encodeChunk(chunk);
            ids.addAll(chunkIds);
        }
        return ids;
    }

    private Map<Pair<Integer, Integer>, Integer> getStats(List<Integer> ids) {
        Map<Pair<Integer, Integer>, Integer> map = new HashMap<>();
        for (int i = 0; i + 1 < ids.size(); i++) {
            Pair<Integer, Integer> key = new Pair<>(ids.get(i), ids.get(i + 1));
            map.put(key, map.getOrDefault(key, 0) + 1);
        }
        return map;
    }

    private List<Integer> encodeChunk(String chunk) {
        // return the token ids
        // let's begin. first, convert all bytes to integers in range 0..255
        List<Integer> ids = new ArrayList<>();
        for (int b : chunk.toCharArray()) {
            int tokenIndex = this.vocabulary.getIndex(String.valueOf((char) b)).orElseThrow();
            ids.add(tokenIndex);
        }

        while (ids.size() >= 2) {
            // find the pair with the lowest merge index
            Map<Pair<Integer, Integer>, Integer> stats = getStats(ids);
            Pair<Integer, Integer> pair = stats.keySet().stream().min(Comparator.comparingInt(key -> this.merges.getOrDefault(key, Integer.MAX_VALUE))).orElseThrow();
            // subtle: if there are no more merges available, the key will
            // result in an inf for every single pair, and the min will be
            // just the first pair in the list, arbitrarily
            // we can detect this terminating case by a membership check
            if (!this.merges.containsKey(pair)) {
                break; // nothing else can be merged anymore
            }
            // otherwise let's merge the best pair (lowest merge index)
            int idx = this.merges.get(pair);
            ids = merge(ids, pair, idx);
        }
        return ids;
    }

    private static List<Integer> merge(List<Integer> ids, Pair<Integer, Integer> pair, int idx) {
        List<Integer> newids = new ArrayList<>();
        int i = 0;
        while (i < ids.size()) {
            // if not at the very last position AND the pair matches, replace it
            if (ids.get(i).equals(pair.first()) && i < ids.size() - 1 && ids.get(i + 1).equals(pair.second())) {
                newids.add(idx);
                i += 2;
            } else {
                newids.add(ids.get(i));
                i += 1;
            }
        }
        return newids;
    }

    public String decodeImpl(List<Integer> tokens) {
        StringBuilder sb = new StringBuilder();
        for (int token : tokens) {
            String tokenString = vocabulary.get(token);
            sb.append(tokenString);
        }
        return sb.toString();
    }

    /**
     * Returns list of utf-8 byte and a corresponding list of unicode strings.
     * The reversible bpe codes work on unicode strings.
     * This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
     * When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
     * This is a significant percentage of your normal, say, 32K bpe vocab.
     * To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
     * And avoids mapping to whitespace/control characters the bpe code barfs on.
     */
    private static Map<Integer, Integer> bytesToUnicode() {
        List<Integer> bs = new ArrayList<>();
        IntStream.rangeClosed('!', '~').forEach(bs::add);
        IntStream.rangeClosed('¡', '¬').forEach(bs::add);
        IntStream.rangeClosed('®', 'ÿ').forEach(bs::add);

        List<Integer> cs = new ArrayList<>(bs);
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (!bs.contains(b)) {
                bs.add(b);
                cs.add(256 + n);
                n += 1;
            }
        }

        // return dict(zip(bs, cs))
        return IntStream.range(0, bs.size())
                .boxed()
                .collect(Collectors.toMap(bs::get, cs::get));
    }

    static final Map<Integer, Integer> BYTE_ENCODER = bytesToUnicode();
    static final Map<Integer, Integer> BYTE_DECODER = BYTE_ENCODER.entrySet()
            .stream()
            .collect(Collectors.toMap(Map.Entry::getValue, Map.Entry::getKey));

    public Collection<? extends Integer> encode(String text) {
        StringBuilder sb = new StringBuilder();
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        for (byte b : bytes) {
        	sb.appendCodePoint(BYTE_ENCODER.get(Byte.toUnsignedInt(b)));
        }
        return encode(sb.toString(), Set.of());
    }

    public static String replaceControlCharacters(int[] codePoints) {
        // we don't want to print control characters
        // which distort the output (e.g. \n or much worse)
        // https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
        // http://www.unicode.org/reports/tr44/#GC_Values_Table\
        StringBuilder chars = new StringBuilder();
        for (int cp : codePoints) {
            if (Character.getType(cp) == Character.CONTROL && cp != '\n') {
                chars.append("\\u").append(HexFormat.of().toHexDigits(cp, 4)); // escape
            } else {
                chars.appendCodePoint(cp); // this character is ok
            }
        }
        return chars.toString();
    }

    public static String replaceControlCharacters(String str) {
        return replaceControlCharacters(str.codePoints().toArray());
    }
    @Override
    public List<Integer> encodeAsList(String text) {
        return Arrays.stream(encodeImpl(encode(text))).boxed().toList();
    }
    @Override
    public String decode(List<Integer> tokens) {
        String decoded = decodeImpl(tokens);
        int[] decodedBytesAsInts = decoded.codePoints().map(BYTE_DECODER::get).toArray();
        byte[] rawBytes = new byte[decodedBytesAsInts.length];
        for (int i = 0; i < decoded.length(); i++) {
            rawBytes[i] = (byte) decodedBytesAsInts[i];
        }
        return new String(rawBytes, StandardCharsets.UTF_8);
    }
}

/**
 * Wherein Llama models metadata.get("tokenizer.ggml.model") = gpt2
 * and Mistral uses metadata.get("tokenizer.ggml.model") = llama.
 */
class MistralTokenizer implements TokenizerInterface {
    private final Pattern compiledPattern;
    private final Vocabulary vocabulary;
    private final Map<String, Integer> specialTokens;
    private final int[] tokenType;
    private final int byte0;

    public String regexPattern() {
        if (compiledPattern == null) {
            return null;
        }
        return compiledPattern.pattern();
    }
    @Override
    public Map<String, Integer> getSpecialTokens() {
        return specialTokens;
    }
    @Override
    public boolean isSpecialToken(int tokenIndex) {
        return getTokenType(tokenIndex) != 1;
    }
    @Override
    public int getTokenType(int tokenIndex) {
        return tokenType[tokenIndex];
    }

    public MistralTokenizer(Vocabulary vocabulary, String regexPattern, Map<String, Integer> specialTokens, int[] tokenType) {
        this.vocabulary = vocabulary;
        this.compiledPattern = regexPattern != null ? Pattern.compile(regexPattern) : null;
        this.specialTokens = new HashMap<>(specialTokens);
        this.tokenType = tokenType;
        this.byte0 = vocabulary.getIndex("<0x00>").orElseThrow();
    }

    public List<Integer> encode(String text) {
        return encodeImpl(text.replace(' ', '▁'));
    }

    private List<Integer> encodeImpl(String text) {

        List<Integer> tokens = new ArrayList<>();

        // first encode every individual codepoint in the input string
        for (int i = 0, cpi; i < text.length(); i += Character.charCount(cpi)) {
            cpi = text.codePointAt(i);

            String singleCodepoint = Character.toString(cpi);
            int id = vocabulary.getIndex(singleCodepoint).orElse(-1);

            if (id != -1) {
                // we found this codepoint in vocab, add it as a token
                tokens.add(id);
            } else {
                // byte_fallback encoding: just encode each byte as a token
                // +byte0 here to skip all the control and special tokens e.g. <unk>, <s>, </s>
                // so the individual bytes only start at token <0x00>
                for (byte b : singleCodepoint.getBytes(StandardCharsets.UTF_8)) {
                    tokens.add(Byte.toUnsignedInt(b) + byte0);
                }
            }
        }


        // merge the best consecutive pair each iteration, according the scores in vocab_scores
        while (true) {
            float best_score = -1e10f;
            int best_id = -1;
            int best_idx = -1;

            for (int i = 0; i < tokens.size() - 1; ++i) {
                // check if we can merge the pair (tokens[i], tokens[i+1])
                String str_buffer = vocabulary.get(tokens.get(i)) + vocabulary.get(tokens.get(i + 1));
                int id = vocabulary.getIndex(str_buffer).orElse(-1);
                if (id != -1 && vocabulary.getScore(id) > best_score) {
                    // this merge pair exists in vocab! record its score and position
                    best_score = vocabulary.getScore(id);
                    best_id = id;
                    best_idx = i;
                }
            }

            if (best_idx == -1) {
                break; // we couldn't find any more pairs to merge, so we're done
            }

            // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
            tokens.set(best_idx, best_id);
            tokens.remove(best_idx + 1);
        }

        return tokens;
    }
    @Override
    public String decode(List<Integer> tokens) {
        StringBuilder sb = new StringBuilder();
        for (int token : tokens) {
            String tokenString = vocabulary.get(token);
            if (isSpecialToken(token)) {
                // some tokens designate raw bytes e.g. '<0x10>'
                String prefix = "<0x";
                String suffix = ">";
                if (tokenString.length() == 6 && tokenString.startsWith(prefix) && tokenString.endsWith(suffix)) {
                    String code = tokenString.substring(prefix.length(), tokenString.length() - suffix.length());
                    int cp = Integer.parseInt(code, 16);
                    tokenString = Character.toString(cp);
                }
            } else {
                tokenString = tokenString.replace('▁', ' ');

            }
            sb.append(tokenString);
        }
        return sb.toString();
    }

    public static String replaceControlCharacters(int[] codePoints) {
        // we don't want to print control characters
        // which distort the output (e.g. \n or much worse)
        // https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
        // http://www.unicode.org/reports/tr44/#GC_Values_Table\
        StringBuilder chars = new StringBuilder();
        for (int cp : codePoints) {
            if (Character.getType(cp) == Character.CONTROL && cp != '\n') {
                chars.append("\\u").append(HexFormat.of().toHexDigits(cp, 4)); // escape
            } else {
                chars.appendCodePoint(cp); // this character is ok
            }
        }
        return chars.toString();
    }

    public static String replaceControlCharacters(String str) {
        return replaceControlCharacters(str.codePoints().toArray());
    }

    public List<Integer> encodeAsList(String text) {
        return encode(text);
    }
}

final class Parallel {
    public static void parallelFor(int startInclusive, int endExclusive, IntConsumer action) {
        if (startInclusive == 0 && endExclusive == 1) {
            action.accept(0);
            return;
        }
        IntStream.range(startInclusive, endExclusive).parallel().forEach(action);
    }

    public static void parallelForLong(long startInclusive, long endExclusive, LongConsumer action) {
        if (startInclusive == 0 && endExclusive == 1) {
            action.accept(0);
            return;
        }
        LongStream.range(startInclusive, endExclusive).parallel().forEach(action);
    }
}

record Pair<First, Second>(First first, Second second) {
}

record GGMLTensorEntry(MemorySegment mappedFile, String name, GGMLType ggmlType, int[] shape,
                       MemorySegment memorySegment) {
}

final class RoPE {
	/**
	 * For GPT2 vocab
	 * @param contextLength
	 * @param headSize
	 * @param theta
	 * @param ropeScaling
	 * @param scaleFactor
	 * @param loFreqFactor
	 * @param hiFreqFactor
	 * @param oldContextLength
	 * @return
	 */
    public static Pair<float[], float[]> precomputeFreqsCis(int contextLength, int headSize, double theta,
                                                            boolean ropeScaling, float scaleFactor, float loFreqFactor, float hiFreqFactor, float oldContextLength) {
        assert headSize % 2 == 0;
        float[] cr = new float[contextLength * (headSize / 2)];
        float[] ci = new float[contextLength * (headSize / 2)];
        int n = 0;
        for (int pos = 0; pos < contextLength; ++pos) {
            for (int i = 0; i < headSize; i += 2) {
                float freq = (float) (1.0 / Math.pow(theta, i / (double) headSize));
                if (ropeScaling) {
                    // Llama 3.1 scaling
                    float loFreqWavelen = oldContextLength / loFreqFactor;
                    float hiFreqWavelen = oldContextLength / hiFreqFactor;
                    float wavelen = (float) (2.0 * Math.PI / freq);
                    if (wavelen < hiFreqWavelen) {
                        freq = freq;
                    } else if (wavelen > loFreqWavelen) {
                        freq = freq / scaleFactor;
                    } else {
                        float smooth = (oldContextLength / wavelen - loFreqFactor) / (hiFreqFactor - loFreqFactor);
                        freq = (1.0f - smooth) * freq / scaleFactor + smooth * freq;
                    }
                }
                float val = pos * freq;
                cr[n] = (float) Math.cos(val);
                ci[n] = (float) Math.sin(val);
                n++;
            }
        }
        assert contextLength * (headSize / 2) == n;
        return new Pair<>(cr, ci);
    }
    /**
     * for Llama vocab
     * @param contextLength
     * @param headSize
     * @param theta
     * @return
     */
    public static Pair<float[], float[]> precomputeFreqsCis(int contextLength, int headSize, double theta) {
        assert headSize % 2 == 0;
        float[] cr = new float[contextLength * (headSize / 2)];
        float[] ci = new float[contextLength * (headSize / 2)];
        int n = 0;
        for (int pos = 0; pos < contextLength; ++pos) {
            for (int i = 0; i < headSize; i += 2) {
                float freq = (float) (1.0 / Math.pow(theta, i / (double) headSize));
                float val = pos * freq;
                cr[n] = (float) Math.cos(val);
                ci[n] = (float) Math.sin(val);
                n++;
            }
        }
        assert contextLength * (headSize / 2) == n;
        return new Pair<>(cr, ci);
    }

}

record Vocabulary(String[] tokens, float[] scores, Map<String, Integer> tokenToIndex) {
    public Vocabulary(String[] vocabulary, float[] scores) {
        this(vocabulary, scores,
                IntStream.range(0, vocabulary.length)
                        .boxed()
                        .collect(Collectors.toMap(i -> vocabulary[i], i -> i))
        );
    }

    public String get(int tokenIndex) {
        return tokens[tokenIndex];
    }

    public OptionalInt getIndex(String token) {
        Integer value = tokenToIndex.get(token);
        return value != null ? OptionalInt.of(value) : OptionalInt.empty();
    }

    public int size() {
        return tokens.length;
    }
    /**
     * Added from Mistral Vocabulary - Groff
     * @param tokenIndex
     * @return
     */
    public float getScore(int tokenIndex) {
        return scores[tokenIndex];
    }
    
    public boolean scoresNull() {
    	return scores == null;
    }

}

@FunctionalInterface
interface Sampler {
    int sampleToken(FloatTensor logits);
    Sampler ARGMAX = FloatTensor::argmax;
}

record CategoricalSampler(RandomGenerator rng) implements Sampler {

    @Override
    public int sampleToken(FloatTensor logits) {
        // sample index from probabilities (they must sum to 1!)
        float random0to1 = rng.nextFloat(1f);
        float cdf = 0.0f;
        for (int i = 0; i < logits.size(); i++) {
            cdf += logits.getFloat(i);
            if (random0to1 < cdf) {
                return i;
            }
        }
        return logits.size() - 1; // in case of rounding errors
    }
}

final class ToppSampler implements Sampler {

    final int[] indices;
    final float topp;
    final RandomGenerator rng;

    public ToppSampler(int maxNumberOfElements, float topp, RandomGenerator rng) {
        this.indices = new int[maxNumberOfElements];
        this.topp = topp;
        this.rng = rng;
    }

    static void swap(int[] array, int from, int to) {
        int tmp = array[from];
        array[from] = array[to];
        array[to] = tmp;
    }

    static void siftDown(int[] array, int from, int n, Comparator<Integer> comparator) {
        int prev = from, next;
        while ((next = 2 * prev + 1) < n) {
            int r = 2 * prev + 2;
            if (r < n && comparator.compare(array[r], array[next]) < 0) {
                next = r;
            }
            if (comparator.compare(array[next], array[prev]) < 0) {
                swap(array, prev, next);
                prev = next;
            } else {
                break;
            }
        }
    }

    @Override
    public int sampleToken(FloatTensor logits) {
        // top-p sampling (or "nucleus sampling") samples from the smallest set of
        // tokens that exceed probability topp. This way we never sample tokens that
        // have very low probabilities and are less likely to go "off the rails".
        Comparator<Integer> comparator = Comparator.comparingDouble(logits::getFloat).reversed();

        int n = logits.size();
        int head = 0;
        int tail = n - 1;
        // values smaller than (1 - topp) / (n - 1) cannot be part of the result
        // so for efficiency we crop these out as candidates before sorting
        float cutoff = (1.0f - topp) / (n - 1);
        for (int i = 0; i < indices.length; i++) {
            if (logits.getFloat(i) >= cutoff) {
                indices[head++] = i;
            } else {
                indices[tail--] = i;
            }
        }

        int n0 = head;
        // build heap O(n0)
        for (int i = n0 / 2 - 1; i >= 0; --i) {
            siftDown(indices, i, n0, comparator);
        }

        // truncate the list where cumulative probability of the largest k elements exceeds topp
        // O(k lg n0)
        float cumulativeProb = 0.0f;
        int lastIndex = 0;
        for (int i = n0 - 1; i >= 0; i--) {
            swap(indices, 0, i);
            cumulativeProb += logits.getFloat(indices[i]);
            if (cumulativeProb > topp) {
                lastIndex = i;
                break; // we've exceeded topp by including lastIndex
            }
            siftDown(indices, 0, i - 1, comparator);
        }

        // sample from the truncated list
        float r = rng.nextFloat(1f) * cumulativeProb;
        float cdf = 0.0f;
        for (int i = n0 - 1; i >= lastIndex; i--) {
            cdf += logits.getFloat(indices[i]);
            if (r < cdf) {
                return indices[i];
            }
        }

        return indices[lastIndex]; // in case of rounding errors
    }
}

interface ChatFormatInterface {
	 public TokenizerInterface getTokenizer();
	 public Set<Integer> getStopTokens();
	 public List<Integer> encodeHeader(ChatFormat.Message message);
	 public List<Integer> encodeMessage(ChatFormat.Message message);
	 public List<Integer> encodeDialogPrompt(boolean appendAssistantTurn, List<ChatFormat.Message> dialog);
	 public int getBeginOfText();
}
/**
 * Utility tailored for Llama 3 instruct prompt format.
 */
class ChatFormat implements ChatFormatInterface {

    final Tokenizer tokenizer;
    final int beginOfText;
    final int endHeader;
    final int startHeader;
    final int endOfTurn;
    final int endOfText;
    final int endOfMessage;
    final Set<Integer> stopTokens;

    public ChatFormat(TokenizerInterface tokenizer) {
        this.tokenizer = (Tokenizer)tokenizer;
        Map<String, Integer> specialTokens = this.tokenizer.getSpecialTokens();
        this.beginOfText = specialTokens.get("<|begin_of_text|>");
        this.startHeader = specialTokens.get("<|start_header_id|>");
        this.endHeader = specialTokens.get("<|end_header_id|>");
        this.endOfTurn = specialTokens.get("<|eot_id|>");
        this.endOfText = specialTokens.get("<|end_of_text|>");
        this.endOfMessage = specialTokens.getOrDefault("<|eom_id|>", -1); // only in 3.1
        this.stopTokens = Set.of(endOfText, endOfTurn);
    }
    @Override
    public TokenizerInterface getTokenizer() {
        return tokenizer;
    }
    @Override
    public Set<Integer> getStopTokens() {
        return stopTokens;
    }
    @Override
    public int getBeginOfText() {
    	return beginOfText;
    }
    @Override
    public List<Integer> encodeHeader(ChatFormat.Message message) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(startHeader);
        tokens.addAll(this.tokenizer.encodeAsList(message.role().name()));
        tokens.add(endHeader);
        tokens.addAll(this.tokenizer.encodeAsList("\n"));
        return tokens;
    }
    @Override
    public List<Integer> encodeMessage(ChatFormat.Message message) {
        List<Integer> tokens = this.encodeHeader(message);
        tokens.addAll(this.tokenizer.encodeAsList(message.content().strip()));
        tokens.add(endOfTurn);
        return tokens;
    }
    @Override
    public List<Integer> encodeDialogPrompt(boolean appendAssistantTurn, List<ChatFormat.Message> dialog) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(beginOfText);
        for (ChatFormat.Message message : dialog) {
            tokens.addAll(this.encodeMessage(message));
        }
        if (appendAssistantTurn) {
            // Add the start of an assistant message for the model to complete.
            tokens.addAll(this.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
        }
        return tokens;
    }

    public record Message(ChatFormat.Role role, String content) {
    }

    public record Role(String name) {
        public static ChatFormat.Role SYSTEM = new ChatFormat.Role("system");
        public static ChatFormat.Role USER = new ChatFormat.Role("user");
        public static ChatFormat.Role ASSISTANT = new ChatFormat.Role("assistant");

        @Override
        public String toString() {
            return name;
        }
    }
}

/**
* Utility tailored for Mistral v0.3 instruct prompt format.
*/
final class MistralChatFormat implements ChatFormatInterface {

   protected final TokenizerInterface tokenizer;
   protected final int unknownToken;
   protected final int beginOfText;
   protected final int endOfText;
   protected final int beginOfInstruction;
   protected final int endOfInstruction;
   protected final int toolCalls;
   protected final int beginOfAvailableTools;
   protected final int endOfAvailableTools;
   protected final int beginOfToolResults;
   protected final int endOfToolResults;
   protected final int prefix;
   protected final int middle;
   protected final int suffix;

   public MistralChatFormat(TokenizerInterface tokenizer) {
       this.tokenizer = tokenizer;
       Map<String, Integer> specialTokens = this.tokenizer.getSpecialTokens();
       this.unknownToken = specialTokens.get("<unk>");
       this.beginOfText = specialTokens.get("<s>");
       this.endOfText = specialTokens.get("</s>");
       this.beginOfInstruction = specialTokens.get("[INST]");
       this.endOfInstruction = specialTokens.get("[/INST]");
       this.toolCalls = specialTokens.get("[TOOL_CALLS]");
       this.beginOfAvailableTools = specialTokens.get("[AVAILABLE_TOOLS]");
       this.endOfAvailableTools = specialTokens.get("[/AVAILABLE_TOOLS]");
       this.beginOfToolResults = specialTokens.get("[TOOL_RESULTS]");
       this.endOfToolResults = specialTokens.get("[/TOOL_RESULTS]");
       // Only Codestral supports FIM tokens.
       this.prefix = specialTokens.getOrDefault("[PREFIX]", unknownToken);
       this.suffix = specialTokens.getOrDefault("[SUFFIX]", unknownToken);
       this.middle = specialTokens.getOrDefault("[MIDDLE]", unknownToken);
   }
   @Override
   public TokenizerInterface getTokenizer() {
       return tokenizer;
   }
   @Override
   public Set<Integer> getStopTokens() {
       return Set.of(endOfText);
   }
   @Override
   public int getBeginOfText() {
   	return beginOfText;
   }
 
   public List<Integer> encodeMessage(String userMessage, boolean addHeader, boolean addFooter) {
       List<Integer> tokens = new ArrayList<>();
       if (addHeader) {
           tokens.add(this.beginOfInstruction);
       }
       if (userMessage != null) {
           tokens.addAll(this.tokenizer.encodeAsList(userMessage.strip()));
       }
       if (addFooter) {
           tokens.add(endOfInstruction);
       }
       return tokens;
   }

   public List<Integer> encodeFillInTheMiddle(String prefix, String suffix) {
       List<Integer> tokens = new ArrayList<>();
       tokens.add(this.suffix);
       tokens.addAll(tokenizer.encode(suffix));
       tokens.add(this.prefix);
       tokens.addAll(tokenizer.encode(prefix));
       return tokens;
   }
   @Override
   public List<Integer> encodeHeader(ChatFormat.Message message) {
       List<Integer> tokens = new ArrayList<>();
       tokens.add(this.beginOfInstruction);
       tokens.addAll(this.tokenizer.encodeAsList(message.role().name()));
       tokens.add(endOfInstruction);
       return tokens;
   }
   @Override
   public List<Integer> encodeMessage(ChatFormat.Message message) {
	   List<Integer> tokens = new ArrayList<>();
	   tokens.add(this.beginOfInstruction);
       tokens.addAll(this.tokenizer.encodeAsList(message.content().strip()));
       tokens.add(endOfInstruction);
       return tokens;
   }
   @Override
   public List<Integer> encodeDialogPrompt(boolean appendAssistantTurn, List<ChatFormat.Message> dialog) {
       List<Integer> tokens = new ArrayList<>();
       tokens.add(beginOfText);
       for (ChatFormat.Message message : dialog) {
           tokens.addAll(this.encodeMessage(message));
       }
       //if (appendAssistantTurn) {
       //    // Add the start of an assistant message for the model to complete.
       //    tokens.addAll(this.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
       //}
       tokens.add(endOfText);
       return tokens;
   }
}

/**
 * Utility tailored for the Chat Markup Language (ChatML) Qwen prompt format.
 */
class ChatMLFormat implements ChatFormatInterface {

    protected final TokenizerInterface tokenizer;
    protected final int imStart;
    protected final int endOfText;
    protected final int imEnd;

    public ChatMLFormat(TokenizerInterface tokenizer) {
        this.tokenizer = tokenizer;
        Map<String, Integer> specialTokens = this.tokenizer.getSpecialTokens();
        this.imStart = specialTokens.get("<|im_start|>");
        this.imEnd = specialTokens.get("<|im_end|>");
        this.endOfText = specialTokens.get("<|endoftext|>");
    }

    public TokenizerInterface getTokenizer() {
        return tokenizer;
    }

    public Set<Integer> getStopTokens() {
        return Set.of(imEnd, endOfText);
    }
    
    @Override
    public int getBeginOfText() {
    	return imStart;
    }
    
    public List<Integer> encodeHeader(ChatFormat.Message message) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(imStart);
        tokens.addAll(this.tokenizer.encodeAsList(message.role().name()));
        tokens.addAll(this.tokenizer.encodeAsList("\n"));
        return tokens;
    }

    public List<Integer> encodeMessage(ChatFormat.Message message) {
        List<Integer> tokens = this.encodeHeader(message);
        tokens.addAll(this.tokenizer.encodeAsList(message.content().strip()));
        tokens.add(imEnd);
        return tokens;
    }

    public List<Integer> encodeDialogPrompt(boolean appendAssistantTurn, List<ChatFormat.Message> dialog) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(imStart);
        for (ChatFormat.Message message : dialog) {
            tokens.addAll(this.encodeMessage(message));
        }
        if (appendAssistantTurn) {
            // Add the start of an assistant message for the model to complete.
            tokens.addAll(this.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
        }
        return tokens;
    }

}

/**
 * Support for AOT preloading of GGUF metadata with GraalVM's Native Image.
 *
 * <p>
 * To preload a model at build time, pass {@code -Dllama.PreloadGGUF=/path/to/model.gguf}
 * to the native-image builder command. At runtime, the preloaded model will be used
 * iff the specified and preloaded file names (base name) match.
 */
final class AOT {
    record PartialModel(String modelFileName, Llama model, long tensorDataOffset, Map<String, GGUF.GGUFTensorInfo> tensorInfos) {}

    private static final PartialModel PRELOADED_GGUF = preLoadGGUF(System.getProperty("llama.PreloadGGUF"));

    private static PartialModel preLoadGGUF(String modelPath) {
        if (modelPath == null || modelPath.isEmpty()) {
            return null;
        }
        try {
            Path path = Path.of(modelPath);
            if (!Files.exists(path) || !Files.isRegularFile(path)) {
                throw new IllegalArgumentException("Cannot pre-load model: " + path);
            }
            GGUF gguf = GGUF.loadModel(path);
            try (FileChannel fileChannel = FileChannel.open(path, StandardOpenOption.READ)) {
                return new PartialModel(
                        path.getFileName().toString(),
                        ModelLoader.loadModel(fileChannel, gguf, Llama3.Options.DEFAULT_MAX_TOKENS, false),
                        gguf.getTensorDataOffset(),
                        gguf.getTensorInfos()
                );
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Tries to reuse a compatible AOT preloaded model.
     * The file name (base name) must match with the preloaded file name.
     * No checksum/hash is checked for performance reasons.
     */
    public static Llama tryUsePreLoaded(Path modelPath, int contextLength) throws IOException {
        AOT.PartialModel preLoaded = AOT.PRELOADED_GGUF;
        if (preLoaded == null) {
            return null; // no pre-loaded model stored
        }
        String optionsModel = modelPath.getFileName().toString();
        String preLoadedModel = preLoaded.modelFileName();
        if (!Objects.equals(optionsModel, preLoadedModel)) {
            // Preloaded and specified model file names didn't match.
            return null;
        }
        Llama baseModel = preLoaded.model();
        try (var timer = Timer.log("Load tensors from pre-loaded model");
             var fileChannel = FileChannel.open(modelPath, StandardOpenOption.READ)) {
            // Load only the tensors (mmap slices).
            Map<String, GGMLTensorEntry> tensorEntries = GGUF.loadTensors(fileChannel, preLoaded.tensorDataOffset(), preLoaded.tensorInfos());
            Llama.Weights weights = ModelLoader.loadGPT2Weights(tensorEntries, baseModel.configuration());
            return new Llama(baseModel.configuration().withContextLength(contextLength), baseModel.tokenizer(), weights);
        }
    }
}

