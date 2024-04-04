package experiment;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
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

import static io.Utils.*;

public class LdiversityRun implements Callable<Long> {
    private int foldNumber;
    private int k;
    private double l;
    private Configuration cfg;
    private String target;
    private String foldDir;
    private File kDir;
    private String trainFilePath;
    private File outputFile;

    public LdiversityRun(int foldNumber, int k, double l, Configuration cfg, String foldDir, String trainFilePath){
        this.foldNumber = foldNumber;
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


    private void printSettings2(ARXResult result) throws IOException {
        File settingsFile = new File(outputFile.getParentFile(), "settings.json");

        List<String> QID = new ArrayList<>(result.getDataDefinition().getQuasiIdentifyingAttributes());
        List<String> IS = new ArrayList<>(result.getDataDefinition().getInsensitiveAttributes());
        List<String> S = new ArrayList<>(result.getDataDefinition().getSensitiveAttributes());

        JsonObject settings = new JsonObject();
        settings.add("QID", convertListToJsonArray(QID));
        settings.add("IS", convertListToJsonArray(IS));
        settings.add("S", convertListToJsonArray(S));
        settings.addProperty("target", target);
        settings.add("privacy criteria", convertIterableToJsonArray(result.getConfiguration().getPrivacyModels()));

        try (FileWriter writer = new FileWriter(settingsFile)) {
            Gson gson = new Gson();
            gson.toJson(settings, writer);
        }
    }

    private void printStats2(ARXResult result) throws IOException {
        File statsFile = new File(outputFile.getParentFile(), "stats.json");

        JsonObject stats = new JsonObject();
        stats.add("node", convertArrayToJsonArray(result.getOutput().getTransformation().getTransformation()));
        stats.add("QID", convertListToJsonArray(List.of(result.getOutput().getTransformation().getQuasiIdentifyingAttributes())));
        stats.addProperty("suppressed in sample", result.getOutput().getView().getStatistics().getEquivalenceClassStatistics().getNumberOfSuppressedRecords());
        stats.addProperty("sample size", result.getOutput().getNumRows() - result.getOutput().getView().getStatistics().getEquivalenceClassStatistics().getNumberOfSuppressedRecords());
        stats.addProperty("input size", result.getOutput().getNumRows());
        stats.addProperty("equivalence classes", result.getOutput().getStatistics().getEquivalenceClassStatistics().getNumberOfEquivalenceClasses());
        stats.addProperty("average EQ size", result.getOutput().getStatistics().getEquivalenceClassStatistics().getAverageEquivalenceClassSize());

        try (FileWriter writer = new FileWriter(statsFile)) {
            Gson gson = new Gson();
            gson.toJson(stats, writer);
        }
    }

    private void saveResults(ARXResult result) throws IOException {
        printSettings2(result);
        printStats2(result);
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
        System.out.println("fold " + foldNumber + " k" + k + " l" + l + ": sucessfully ran in " + duration / 1000d + " seconds");
        return (end - start);
    }

}
