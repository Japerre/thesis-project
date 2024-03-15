package io;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import org.apache.commons.configuration2.Configuration;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.*;
import java.util.Arrays;
import java.util.List;

import static io.Utils.*;


public class OutputWriter {

    public static void copyFolder(Path source, Path destination) throws IOException {
        FileUtils.deleteDirectory(destination.toFile());
        FileUtils.copyDirectory(source.toFile(), destination.toFile());
    }

    public static String writeExperimentStats2(int[] kArr, double[] bArr, double[] lValues, String[] qid, Configuration cfg) throws IOException {
        File directory = new File(cfg.getString("outputBasePath"));
        File experimentStatsFile = new File(cfg.getString("experimentStatsFile"));
        String inputDatasetFolder = cfg.getString("inputDatasetFolder");
        String experimentBasePath = cfg.getString("experimentBasePath");
        String inputDataDefinitionPath = cfg.getString("inputDataDefinitionPath");
        String inputDataDefinitionAbsolutePath = Paths.get(System.getProperty("user.dir")).resolve(inputDataDefinitionPath).toString();
        String kAnonBasePath = cfg.getString("kAnonBasePath");
        Path inputDatasetPath = Paths.get(cfg.getString("inputDataFileFolder") + "train.csv");

        // Create a JsonObject to hold the data
        JsonObject jsonObject = new JsonObject();
        Gson gson = new Gson();

        jsonObject.add("k_values", convertArrayToJsonArray(kArr));
        jsonObject.add("b_values", convertArrayToJsonArray(bArr));
        jsonObject.add("l_values", convertArrayToJsonArray(lValues));
        jsonObject.add("qid", convertListToJsonArray(List.of(qid)));

        jsonObject.addProperty("experimentBasePath", experimentBasePath);
        jsonObject.addProperty("kAnonFolderPath", kAnonBasePath);
        jsonObject.addProperty("inputDatasetPath", inputDatasetPath.toString());
        jsonObject.addProperty("inputDataDefinitionPath", inputDataDefinitionPath);
        jsonObject.addProperty("inputDataDefinitionAbsolutePath", inputDataDefinitionAbsolutePath);

        // Write the JSON object to the file
        try (FileWriter writer = new FileWriter(experimentStatsFile)) {
            gson.toJson(jsonObject, writer);
        } catch (IOException e) {
            e.printStackTrace();
        }

        return experimentStatsFile.getPath();
    }

//    public static String writeExperimentStats(int[] kArr, double[] bArr, Configuration config) throws IOException {
//        File directory = new File(config.getString("outputBasePath"));
//        File experimentStatsFile = new File(config.getString("experimentStatsFile"));
//        String inputDatasetFolder = config.getString("inputDatasetFolder");
//        String experimentBasePath = config.getString("experimentBasePath");
//        String inputDataDefenitionPath = config.getString("inputDataDefenitionPath");
//        String inputDataDefenitionAbsolutePath = Paths.get(System.getProperty("user.dir")).resolve(inputDataDefenitionPath).toString();
//
//        String kAnonBasePath = config.getString("kAnonBasePath");
//
//        Path inputDatasetPath = Path.of(config.getString("inputDataFileFolder")+"train.csv");
//
//        String[] headers = {"k", "b", "experimentBasePath", "kAnonFolderPath", "inputDatasetPath", "inputDataDefenitionPath", "inputDataDefenitionAbsolutePath", "QID"};
//
//        CSVFormat csvFormat = CSVFormat.DEFAULT.builder()
//                .setHeader(headers)
//                .setDelimiter(';')
//                .build();
//
//        try (
//                FileWriter fileWriter = new FileWriter(experimentStatsFile);
//                CSVPrinter printer = new CSVPrinter(fileWriter, csvFormat);
//        ) {
//            printer.printRecord(
//                    Arrays.toString(kArr),
//                    Arrays.toString(bArr),
//                    experimentBasePath,
//                    kAnonBasePath,
//                    inputDatasetPath.toString(),
//                    inputDataDefenitionPath,
//                    inputDataDefenitionAbsolutePath,
//                    config.getString("qid")
//            );
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//
//        return experimentStatsFile.getPath();
//    }

    public static String makeKanonDirectory(Configuration config) throws IOException {
        String kAnonBasePath = config.getString("kAnonBasePath");
        Files.createDirectories(Paths.get(kAnonBasePath));
        return kAnonBasePath;
    }

    public static String makeLdiversityDirectory(Configuration cfg) throws IOException {
        String lDiversityBasePath = cfg.getString("lDiversityBasePath");
        Files.createDirectories(Paths.get(lDiversityBasePath));
        return lDiversityBasePath;
    }

    private static void deleteExistingExperiment(Path experimentBasePath) throws IOException {
        if (!Files.exists(experimentBasePath)) {
            return;
        }
        Files.walk(experimentBasePath)
                .map(Path::toFile)
                .forEach(file -> {
                    file.delete();
                });
    }

    public static String makeExperimentDirectory(Configuration config) throws IOException {
        String experimentBasePath = config.getString("experimentBasePath");

        deleteExistingExperiment(Path.of(experimentBasePath));
        Files.createDirectories(Paths.get(experimentBasePath));

        Path inputDataFileFolder = Path.of(config.getString("inputDataFileFolder"));

        Path inputDatasetFolder = Path.of(config.getString("inputDatasetFolder"));

        Files.createDirectories(inputDatasetFolder);

        copyDirectory(inputDataFileFolder, inputDatasetFolder);

        return experimentBasePath;
    }

    public static String makeSampleDirectory(String experimentBasePath, String name) throws IOException {
        String sampleBasePath = experimentBasePath + "/" + name;
        Files.createDirectories(Path.of(sampleBasePath));
        return sampleBasePath;
    }

    public static String makeHierarchiesDirectory(Configuration config) throws IOException {
        String hierarchiesPath = config.getString("experimentBasePath") + "/hierarchies";
        Files.createDirectories(Path.of(hierarchiesPath));
        String hierarchiesDirectoryPath = config.getString("hierarchiesDirectoryPath");

        // Copy all files from the source directory to the destination directory
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(Paths.get(hierarchiesDirectoryPath))) {
            for (Path entry : stream) {
                Files.copy(entry, Paths.get(hierarchiesPath, entry.getFileName().toString()), StandardCopyOption.REPLACE_EXISTING);
            }
        }
        return hierarchiesPath;
    }

    public static String makeFoldDir(int foldNumber, String privCritBasePath) throws IOException {
        String foldDir = privCritBasePath+"/fold_"+foldNumber;
        Files.createDirectories(Path.of(foldDir));
        return foldDir;
    }


}
