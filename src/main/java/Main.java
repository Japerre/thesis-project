import experiment.KanonRun;
import experiment.LdiversityRun;
import experiment.PostSampleRun;
import io.InputReader;
import io.OutputWriter;
import org.apache.commons.configuration2.Configuration;
import org.apache.commons.configuration2.builder.fluent.Configurations;
import org.apache.commons.configuration2.ex.ConfigurationException;
import sampling.Sampler;
import sampling.strategies.SamplerStrategy;
import sampling.Samplers;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.*;

import static io.Utils.*;

public class Main {

    static Configuration cfg;
    static int numberOfThreads;
    static int NUMBER_OF_FOLDS;
    static int[] kValues;
    static double[] sampleRates;
    static double[] lValues;
    static String[] qid;

    private static void readProgramConfig(String configFilePath) throws ConfigurationException, FileNotFoundException {
        Configurations configs = new Configurations();
        cfg = configs.properties(new File(configFilePath));

        kValues = Arrays.stream(cfg.getString("k_values").split(";"))
                .mapToInt(Integer::parseInt)
                .toArray();

        sampleRates = Arrays.stream(cfg.getString("b_values").split(";"))
                .mapToDouble(Double::parseDouble)
                .toArray();

        lValues = Arrays.stream(cfg.getString("l_values").split(";"))
                .mapToDouble(Double::parseDouble)
                .toArray();

        qid = getQID(cfg.getString("inputDataDefinitionPath"));

        NUMBER_OF_FOLDS = Integer.parseInt(cfg.getString("numberOfFolds"));
    }

    public static void runKAnon() throws IOException {

        String experimentBasePath = OutputWriter.makeExperimentDirectory(cfg);
        String kAnonBasePath = OutputWriter.makeKanonDirectory(cfg);
        String hierarchiesPath = OutputWriter.makeHierarchiesDirectory(cfg);

        ExecutorService executorService = Executors.newFixedThreadPool(numberOfThreads);
        List<Future<Long>> futures = new ArrayList<>();

        try {
            for (int foldNumber = 0; foldNumber < NUMBER_OF_FOLDS; foldNumber++) {
                String foldDir = OutputWriter.makeFoldDir(foldNumber, cfg.getString("kAnonBasePath"));
                String trainPath = cfg.getString("foldsPath") + "/fold_" + foldNumber + "/train.csv";

                for (int k : kValues) {
                    KanonRun kanonRun = new KanonRun(k, cfg, foldDir, foldNumber, trainPath);
                    futures.add(executorService.submit(kanonRun));
                }

            }
            for (Future<Long> future : futures) {
                future.get();
            }
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
            executorService.shutdown();
            System.exit(1);
        } finally {
            executorService.shutdown();
        }
        OutputWriter.writeExperimentStats2(kValues, sampleRates, lValues, qid, cfg);
    }

    public static void runEntropyLDiversity() throws IOException {
        String lDiversityPath = OutputWriter.makeLdiversityDirectory(cfg);
        ExecutorService executorService = Executors.newFixedThreadPool(numberOfThreads);
        List<Future<Long>> futures = new ArrayList<>();

        try {
            for (int foldNumber = 0; foldNumber < NUMBER_OF_FOLDS; foldNumber++) {
                String foldDir = OutputWriter.makeFoldDir(foldNumber, cfg.getString("lDiversityBasePath"));
                String trainPath = cfg.getString("foldsPath") + "/fold_" + foldNumber + "/train.csv";

                for (int k : kValues) {
                    for (double l : lValues) {
                        LdiversityRun ldiversityRun = new LdiversityRun(k, l, cfg, foldDir, trainPath);
                        futures.add(executorService.submit(ldiversityRun));
                    }
                }
            }
            for (Future<Long> future : futures) {
                future.get();
            }
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
            executorService.shutdown();
            System.exit(1);
        } finally {
            executorService.shutdown();
        }
    }

    public static void runPostSample(Samplers samplerType) throws IOException, InterruptedException, ExecutionException {
        String target = InputReader.getTarget(cfg.getString("inputDataDefinitionPath"));
        String sampleBasePath = OutputWriter.makeSampleDirectory(cfg.getString("experimentBasePath"), samplerType.getSamplerFolderName());
        ExecutorService executorService = Executors.newFixedThreadPool(numberOfThreads);
        List<Future<Long>> futures = new ArrayList<>();

        try {
            for (int foldNumber = 0; foldNumber < NUMBER_OF_FOLDS; foldNumber++) {
                for (int k : kValues) {
                    for (double sampleRate : sampleRates) {
                        SamplerStrategy samplerStrategy = samplerType.getSamplerStrategy(sampleRate, target);
                        PostSampleRun postSampleRun = new PostSampleRun(new Sampler(samplerStrategy), sampleBasePath, k, sampleRate, foldNumber, cfg);
                        futures.add(executorService.submit(postSampleRun));
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            executorService.shutdown();
            for (Future<Long> future : futures) {
                future.get();
            }
        }
    }

    public static void runPostSampleSingleThreaded(Samplers samplerType) throws IOException {
        String target = InputReader.getTarget(cfg.getString("inputDataDefinitionPath"));
        String sampleBasePath = OutputWriter.makeSampleDirectory(cfg.getString("experimentBasePath"), samplerType.getSamplerFolderName());

        for (int foldNumber = 0; foldNumber < NUMBER_OF_FOLDS; foldNumber++) {
            for (int k : kValues) {
                for (double sampleRate : sampleRates) {
                    SamplerStrategy samplerStrategy = samplerType.getSamplerStrategy(sampleRate, target);
                    PostSampleRun postSampleRun = new PostSampleRun(new Sampler(samplerStrategy), sampleBasePath, k, sampleRate, foldNumber, cfg);
                    postSampleRun.run();
                }
            }
        }
    }

    public static void makeFolds(String pythonEnvPath) throws InterruptedException, IOException {
        String inputDataFile = cfg.getString("inputDataFile");
        ProcessBuilder processBuilder = new ProcessBuilder(
                pythonEnvPath,
                "python/data_splitter.py",
                inputDataFile,
                cfg.getString("crossValidate"));

        processBuilder.redirectErrorStream(true);
        Process process = processBuilder.start();

        try (InputStream inputStream = process.getInputStream();
             InputStreamReader inputStreamReader = new InputStreamReader(inputStream);
             BufferedReader bufferedReader = new BufferedReader(inputStreamReader)) {
            String line;
            while ((line = bufferedReader.readLine()) != null) {
                System.out.println(line);
            }
        }
        int exitCode = process.waitFor();
        if (exitCode != 0) {
            throw new RuntimeException("Error executing Python script. Check the console for details.");
        }
    }

    public static void main(String[] args) throws Exception {

        String configFilePath = args[0];
        numberOfThreads = Integer.parseInt(args[1]);
        String pythonEnvPath = args[2]; //"C:\\Users\\tibol\\anaconda3\\envs\\folktables\\python"

//        String configFilePath = "src/config/nursery.properties";
//        String configFilePath = "src/config/ASCIncome_USA_2018_binned_imbalanced_16645.properties";
//        String configFilePath = "src/config/ASCIncome_USA_2018_binned_imbalanced_1664500.properties";
//        String configFilePath = "src/config/ACSIncome_USA_2018_binned_imbalanced_16645_acc_metric.properties";
//        String configFilePath = "src/config/cmc.properties";
        readProgramConfig(configFilePath);


        if (cfg.getBoolean("crossValidate")) {
            makeFolds(pythonEnvPath);
        }
        if (cfg.getBoolean("kAnon")) {
            runKAnon();
        }
        if (cfg.getBoolean("postSample")) {
//            runPostSample(Samplers.RSample);
            runPostSample(Samplers.SSAMPLE);
            runPostSample(Samplers.BSAMPLE);
        }
        if (cfg.getBoolean("lDiv")) {
            runEntropyLDiversity();
        }
    }

}
