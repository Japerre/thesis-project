package experiment;

import io.Config;
import io.InputReader;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.deidentifier.arx.*;
import sampling.Sampler;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.concurrent.Callable;

public class PostSampleRun implements Callable<Long> {

    private Sampler sampler;
    private int k;
    private double sampleRate;
    private InputReader.SampleInput sampleInput;
    private String sampleBasePath;
    private int foldNumber;

    public PostSampleRun(Sampler sampler, int k, double sampleRate, InputReader.SampleInput sampleInput, String sampleBasePath, int foldNumber) {
        this.sampler = sampler;
        this.k = k;
        this.sampleRate = sampleRate;
        this.sampleInput = sampleInput;
        this.sampleBasePath = sampleBasePath;
        this.foldNumber = foldNumber;
    }

    private void setGeneralizationLevels(Data data) throws IOException {
        File statsFile = Path.of(sampleInput.kAnonFolderPath).resolve("fold_"+foldNumber).resolve("k"+k).resolve("stats.csv").toFile();
        Reader statsFileReader = new FileReader(statsFile);

        CSVFormat csvFormatStatsFile = CSVFormat.DEFAULT.builder()
                .setHeader("node", "QID", "suppressed in sample", "sample size", "input size", "equivalence classes", "average EQ size")
                .setSkipHeaderRecord(true)
                .setDelimiter(';')
                .build();

        Iterable<CSVRecord> statsRecords = csvFormatStatsFile.parse(statsFileReader);

        String QID = "";
        String node = "";

        for (CSVRecord record : statsRecords) {
            node = record.get("node");
            QID = record.get("QID");
        }

        String[] QIDArr = QID.substring(1, QID.length() - 1).split(", ");
        String[] nodeArr = node.substring(1, node.length() - 1).split(", ");
        int[] levels = Arrays.stream(nodeArr)
                .mapToInt(Integer::parseInt)
                .toArray();

        for(int i=0;i<QIDArr.length;i++){
            data.getDefinition().setMinimumGeneralization(QIDArr[i],levels[i]);
            data.getDefinition().setMaximumGeneralization(QIDArr[i],levels[i]);
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
//        sampler.savePostSample(sampleBasePath); //? how to save the rows file

    }

    public void run() throws IOException {
        String trainSet = sampleInput.foldsDirPath+"/fold_"+foldNumber+"/train.csv";
        ImmutablePair<Data, String> immutablePair = InputReader.loadData(sampleInput.inputDataDefenitionPath, trainSet, false);
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
