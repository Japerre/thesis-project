package experiment;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import io.Config;
import io.InputReader;
import org.apache.commons.configuration2.Configuration;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.deidentifier.arx.*;
import org.deidentifier.arx.metric.Metric;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;

import static io.Utils.*;

public class KanonRun implements Callable {

    private int k;
    private Configuration config;
    private String target;
    private File kDir;
    private File kAnonFile;
    private String foldDir;
    private int foldNumber;
    private String trainFilePath;

    public KanonRun(int k, Configuration config, String foldDir, int foldNumber, String trainFilePath) {
        this.k = k;
        this.config = config;
        this.foldDir = foldDir;
        this.foldNumber = foldNumber;
        this.kDir = new File(foldDir, "k" + k);
        this.trainFilePath = trainFilePath;
        kDir.mkdirs();
        kAnonFile = new File(kDir, "output_sample.csv");
    }

    private void printSettings(ARXResult result) {
        File settings = new File(kDir, "settings.csv");

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
            printer.printRecord(QID, IS, S, target, k + "-anonimity");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void printSettings2(ARXResult result) throws IOException {
        File settingsFile = new File(kDir, "settings.json");

        List<String> QID = new ArrayList<>(result.getDataDefinition().getQuasiIdentifyingAttributes());
        List<String> IS = new ArrayList<>(result.getDataDefinition().getInsensitiveAttributes());
        List<String> S = new ArrayList<>(result.getDataDefinition().getSensitiveAttributes());

        JsonObject settings = new JsonObject();
        settings.add("QID", convertListToJsonArray(QID));
        settings.add("IS", convertListToJsonArray(IS));
        settings.add("S", convertListToJsonArray(S));
        settings.addProperty("target", target);
        settings.addProperty("privacy criteria", k + "-anonymity");

        try (FileWriter writer = new FileWriter(settingsFile)) {
            Gson gson = new Gson();
            gson.toJson(settings, writer);
        }
    }

    private void printStats(ARXResult result) {
        File stats = new File(kDir, "stats.csv");

        String[] headers = {"node", "QID", "suppressed in sample", "sample size", "input size", "equivalence classes", "average EQ size"};

        CSVFormat csvFormat = CSVFormat.DEFAULT.builder()
                .setHeader(headers)
                .setDelimiter(';')
                .build();

        String qid = Arrays.toString(result.getOutput().getTransformation().getQuasiIdentifyingAttributes());
        if (!config.containsKey("qid")) {
            config.setProperty("qid", qid);
        }

        try (
                FileWriter fileWriter = new FileWriter(stats);
                CSVPrinter printer = new CSVPrinter(fileWriter, csvFormat);
        ) {
            printer.printRecord(
                    Arrays.toString(result.getOutput().getTransformation().getTransformation()),
                    Arrays.toString(result.getOutput().getTransformation().getQuasiIdentifyingAttributes()),
                    result.getOutput().getView().getStatistics().getEquivalenceClassStatistics().getNumberOfSuppressedRecords(),
                    result.getOutput().getNumRows() - result.getOutput().getView().getStatistics().getEquivalenceClassStatistics().getNumberOfSuppressedRecords(),
                    result.getOutput().getNumRows(),
                    result.getOutput().getStatistics().getEquivalenceClassStatistics().getNumberOfEquivalenceClasses(),
                    result.getOutput().getStatistics().getEquivalenceClassStatistics().getAverageEquivalenceClassSize()
            );
        } catch (IOException e) {
            e.printStackTrace();
        }
    }



    private void printStats2(ARXResult result) throws IOException {
        File statsFile = new File(kDir, "stats.json");

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
        result.getOutput().save(kAnonFile);
    }

    public void run() throws IOException {
        ImmutablePair<Data, String> immutablePair = InputReader.loadData(config.getString("inputDataDefinitionPath"), trainFilePath, false);
        Data data = immutablePair.getLeft();
        target = immutablePair.getRight();
        ARXAnonymizer anonymizer = new ARXAnonymizer();
        ARXConfiguration arxConfiguration = config.getBoolean("accMetric") ?
                Config.getConfig(k, Metric.createClassificationMetric()) :
                Config.getConfig(k, Metric.createLossMetric(0.5, Metric.AggregateFunction.ARITHMETIC_MEAN));
        ARXResult result = anonymizer.anonymize(data, arxConfiguration);
        saveResults(result);
    }

    @Override
    public Long call() throws Exception {
        long start = System.currentTimeMillis();
        this.run();
        long end = System.currentTimeMillis();
        long duration = end - start;
        System.out.println("fold " + foldNumber + " k" + k + ": sucessfully ran in " + duration / 1000d + " seconds");
        return (end - start);
    }
}
