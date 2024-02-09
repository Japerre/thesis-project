package experiment;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
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
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.Callable;

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


    private void saveResults(ARXResult result) throws IOException {
        printSettings(result);
        printStats(result);
        result.getOutput().save(kAnonFile);
    }

    public void run() throws IOException {
        ImmutablePair<Data, String> immutablePair = InputReader.loadData(config.getString("inputDataDefenitionPath"), trainFilePath, false);
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