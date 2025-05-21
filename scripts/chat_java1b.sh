java -server -XX:+UseParallelGC -Xmn96g -Xms96g -Xmx96g --enable-preview --add-modules jdk.incubator.vector -jar llama3.jar --model Llama-3.2-1B-Instruct-Q8_0.gguf --chat -n 128000 --seed 42
