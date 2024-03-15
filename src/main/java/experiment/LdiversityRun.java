package experiment;

import io.Config;
import io.InputReader;
import org.apache.commons.configuration2.Configuration;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.deidentifier.arx.ARXAnonymizer;
import org.deidentifier.arx.ARXConfiguration;
import org.deidentifier.arx.ARXResult;
import org.deidentifier.arx.Data;
import org.deidentifier.arx.metric.Metric;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;

public class LdiversityRun implements Callable<Long> {
    private int k;
    private double l;
    private Configuration cfg;
    private String target;
    private String foldDir;
    private File kDir;
    private String trainFilePath;
    private File outputFile;

    public LdiversityRun(int k, double l, Configuration cfg, String foldDir, String trainFilePath){
        this.k = k;
        this.l = l;
        this.cfg = cfg;
        this.foldDir = foldDir;
        this.kDir = new File(foldDir, "k"+k);
        this.trainFilePath = trainFilePath;
        Path outputFilePath = Paths.get(foldDir,"k"+k, "l"+l, "sample.csv");
        outputFilePath.getParent().toFile().mkdirs();
        this.outputFile = outputFilePath.toFile();
    }

    private void printSettings(ARXResult result) {
        File settings = new File(outputFile.getParentFile(), "settings.csv");

        List<String> QID = new ArrayList<>(result.getDataDefinition().getQuasiIdentifyingAttributes());
        List<String> IS = new ArrayList<>(result.getDataDefinition().getInsensitiveAttributes());
        List<String> S = new ArrayList<>(result.getDataDefinition().getSensitiveAttributes());


        String[] headers = {"QID", "IS", "S", "target", "privacy criteria"};

        CSVFormat csvFormat = CSVFormat.DEFAULT.builder()
                .setHeader(headers)
                .setDelimiter(';')
                .build();

        try (
                FileWriter fileWriter = new FileWriter(settings);
                CSVPrinter printer = new CSVPrinter(fileWriter, csvFormat);
        ) {
            printer.printRecord(QID, IS, S, target, result.getConfiguration().getPrivacyModels());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void printStats(ARXResult result) {
        File stats = new File(outputFile.getParentFile(), "stats.csv");

        String[] headers = {"node", "QID", "suppressed in sample", "sample size", "input size", "equivalence classes", "average EQ size"};

        CSVFormat csvFormat = CSVFormat.DEFAULT.builder()
                .setHeader(headers)
                .setDelimiter(';')
                .build();

        String qid = Arrays.toString(result.getOutput().getTransformation().getQuasiIdentifyingAttributes());
        if(!cfg.containsKey("qid")){
            cfg.setProperty("qid", qid);
        }

        try (
                FileWriter fileWriter = new FileWriter(stats);
                CSVPrinter printer = new CSVPrinter(fileWriter, csvFormat);
        ) {
            printer.printRecord(
                    Arrays.toString(result.getOutput().getTransformation().getTransformation()),
                    Arrays.toString(result.getOutput().getTransformation().getQuasiIdentifyingAttributes()),
                    result.getOutput().getView().getStatistics().getEquivalenceClassStatistics().getNumberOfSuppressedRecords(),
                    result.getOutput().getView().getNumRows(),
                    result.getOutput().getNumRows(),
                    result.getOutput().getStatistics().getEquivalenceClassStatistics().getNumberOfEquivalenceClasses(),
                    result.getOutput().getStatistics().getEquivalenceClassStatistics().getAverageEquivalenceClassSize()
            );
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void saveResults(ARXResult result) throws IOException {
        printSettings(result);
        printStats(result);
        result.getOutput().save(outputFile);
    }

    public void run() throws IOException {
        ImmutablePair<Data, String> immutablePair = InputReader.loadData(cfg.getString("inputDataDefinitionPath"), trainFilePath, true);
        Data data = immutablePair.getLeft();
        target = immutablePair.getRight();
        ARXAnonymizer anonymizer = new ARXAnonymizer();
        ARXConfiguration arxConfiguration = cfg.getBoolean("accMetric") ?
                Config.getConfig(k, l, target, Metric.createClassificationMetric()) :
                Config.getConfig(k, l, target, Metric.createLossMetric(0.5, Metric.AggregateFunction.ARITHMETIC_MEAN));

        ARXResult result = anonymizer.anonymize(data, arxConfiguration);
        saveResults(result);
    }

    @Override
    public Long call() throws Exception {
        long start = System.currentTimeMillis();
        this.run();
        long end = System.currentTimeMillis();
        long duration = end - start;
        System.out.println("k"+k+" l"+l+": sucessfully ran in "+duration/1000d+" seconds");
        return (end - start);
    }

}
