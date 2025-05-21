java -server -XX:+UseParallelGC -Xmn96g -Xms96g -Xmx96g --enable-preview --add-modules jdk.incubator.vector -jar llama3.jar --model gemma-1.1-2b-it.gguf --chat -n 128000 --seed 42
