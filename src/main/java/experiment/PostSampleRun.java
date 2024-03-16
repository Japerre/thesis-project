package experiment;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import io.Config;
import io.InputReader;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.configuration2.Configuration;
import org.deidentifier.arx.*;
import sampling.Sampler;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.Callable;

public class PostSampleRun implements Callable<Long> {

    private Sampler sampler;
    private int k;
    private double sampleRate;
    private String kAnonBasePath;
    private String sampleBasePath;
    private String inputDataDefinitionPath;
    private String foldsDirPath;
    private int foldNumber;


    public PostSampleRun(Sampler sampler, String sampleBasePath, int k, double sampleRate, int foldNumber, Configuration cfg) {
        this.sampler = sampler;
        this.k = k;
        this.sampleRate = sampleRate;
        this.foldNumber = foldNumber;
        this.sampleBasePath = sampleBasePath;
        this.kAnonBasePath = cfg.getString("kAnonBasePath");
        this.sampleBasePath = sampleBasePath;
        this.foldsDirPath = cfg.getString("foldsPath");
        this.inputDataDefinitionPath = cfg.getString("inputDataDefinitionPath");
    }

    private void setGeneralizationLevels(Data data) throws FileNotFoundException {
        File statsFile = Path.of(kAnonBasePath).resolve("fold_"+foldNumber).resolve("k"+k).resolve("stats.json").toFile();
        FileReader fileReader = new FileReader(statsFile);
        Gson gson = new Gson();
        JsonObject jsonObject = gson.fromJson(fileReader, JsonObject.class);

        int[] genLevels = gson.fromJson(jsonObject.getAsJsonArray("node"), int[].class);
        String[] qid = gson.fromJson((jsonObject.getAsJsonArray("QID")), String[].class);

        for(int i=0;i<qid.length;i++){
            data.getDefinition().setMinimumGeneralization(qid[i],genLevels[i]);
            data.getDefinition().setMaximumGeneralization(qid[i],genLevels[i]);
        }

    }

    private void savePostSample(ARXResult result) throws IOException {
        Path kSubFolder = Path.of(sampleBasePath).resolve("fold_"+foldNumber).resolve("k"+k);
        Files.createDirectories(kSubFolder);
        Path postSampleFolder = kSubFolder.resolve("B("+sampleRate+")");
        Files.createDirectories(postSampleFolder);
        Path samplePath = postSampleFolder.resolve("B("+sampleRate+")_sample.csv");
        Path rowsPath = postSampleFolder.resolve("B("+sampleRate+")_rows");

        result.getOutput().getView().save(samplePath.toFile());
//        sampler.savePostSample(postSampleFolder); //? how to save the rows file
    }

    public void run() throws IOException {
        String trainSet = foldsDirPath+"/fold_"+foldNumber+"/train.csv";
        ImmutablePair<Data, String> immutablePair = InputReader.loadData(inputDataDefinitionPath, trainSet, false);
        Data data = immutablePair.getLeft();
        setGeneralizationLevels(data);
        ARXResult result = new ARXAnonymizer().anonymize(data, Config.getConfig(k));
        DataSubset postSubset = sampler.getPostSample(result.getOutput());
        result.createSubset(result.getOutput(), postSubset);
        savePostSample(result);
    }

    @Override
    public Long call() throws Exception {
        long start = System.currentTimeMillis();
        this.run();
        long end = System.currentTimeMillis();
        long duration = end - start;
        System.out.println(sampler.getPostSamplerName() + " k"+k+" B"+sampleRate+" completed in: " + duration/1000d + " seconds");
        return (end - start);
    }

}
