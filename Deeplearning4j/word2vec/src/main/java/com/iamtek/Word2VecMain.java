package com.iamtek;

import java.io.File;
import java.io.FileNotFoundException;
import java.text.NumberFormat;
import java.util.Arrays;
import java.util.Collection;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Word2VecMain {

    private static Logger log = LoggerFactory.getLogger(Word2VecMain.class);
    private static String FILE_PATH = "src/main/resources/";
    
    public static void main(String[] args) throws Exception {
        
    	Word2Vec vec = null;
    	
    	run(vec);
    	test(vec);
        
        
        /*log.info("Plot TSNE");
        BarnesHutTsne tsne = new BarnesHutTsne.Builder()
                .setMaxIter(1000)
                .stopLyingIteration(250)
                .learningRate(500)
                .useAdaGrad(false)
                .theta(0.5)
                .setMomentum(0.5)
                .normalize(true)
                .build();
        vec.lookupTable().plotVocab(tsne, 10, new File(FILE_PATH, "plot.txt"));*/
        
    }
	public static void run(Word2Vec vec) throws Exception {
        
		//get data
        log.info("Getting data...");
        String filePath = new ClassPathResource("raw_sentences.txt").getFile().getAbsolutePath();
        System.out.println(filePath);
        SentenceIterator iter =  new LineSentenceIterator(new File(filePath));
        
        //prepare data, tokenize data
        TokenizerFactory tf = new DefaultTokenizerFactory();
        tf.setTokenPreProcessor(new CommonPreprocessor());

        //build model
        log.info("Building model...");
        vec = new Word2Vec.Builder()
                .minWordFrequency(5)
                .iterations(1)
                .layerSize(100)
                .seed(100)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(tf)
                .build();
        
        //fit model
        log.info("Fitting model...");
        vec.fit();
        
        //saving
        WordVectorSerializer.writeFullModel(vec, FILE_PATH+"model.txt");
        WordVectorSerializer.writeWordVectors(vec, new File(FILE_PATH, "vectors.txt"));
        
	}
	
	public static void test(Word2Vec vec) throws Exception{
		
        //test model
        log.info("Testing model...");
        vec = WordVectorSerializer.loadFullModel(FILE_PATH+"model.txt");
        //vec = WordVectorSerializer.readWord2VecModel(new File(FILE_PATH, "GoogleNews-vectors-negative300.bin.gz"));
        String wd = "day";
        Collection<String> first = vec.wordsNearest(wd, 10);
        System.out.println("Closest to ["+wd+"]:");
    	NumberFormat nf = NumberFormat.getInstance();
    	nf.setMaximumFractionDigits(2);
        for (String w : first) {
			System.out.println(w + ":\t" + nf.format(((Word2Vec)vec).similarity(wd, w)));
		}
        Collection<String> list = vec.wordsNearest(Arrays.asList("man", "women"), Arrays.asList("him"), 10);
        System.out.println(list);

	}

}
