package io;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import org.deidentifier.arx.AttributeType;
import sampling.Grouper;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

public class Utils {

    public static void copyDirectory(Path source, Path target) throws IOException {
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(source)) {
            for (Path entry : stream) {
                Path targetEntry = target.resolve(entry.getFileName());
                if (Files.isDirectory(entry)) {
                    Files.createDirectories(targetEntry);
                    copyDirectory(entry, targetEntry); // Recursive call for subdirectories
                } else {
                    Files.copy(entry, targetEntry, StandardCopyOption.REPLACE_EXISTING);
                }
            }
        }
    }

    public static String getFileNameWithoutExtension(String pathString) {
        Path path = Paths.get(pathString);
        String fileName = path.getFileName().toString();
        int lastDotIndex = fileName.lastIndexOf('.');
        return fileName.substring(0, lastDotIndex);
    }

    public static List<Integer> spreadEqually(int amount, int groups){
        List<Integer> result = new ArrayList<>(groups);
        for(int i = 0; i<groups;i++){
            result.add(amount/groups);
        }

        int leftover = amount%groups;
        for (int i =0;i<leftover;i++){
            int tmp = result.get(i)+1;
            result.set(i,tmp);
        }
        Collections.shuffle(result);
        return result;
    }

    @Deprecated
    public static void printEqDistribution(Grouper groups, Set<Integer> sample, int size){
        Grouper.EquivalenceClass[] reverseGroups = groups.inverseGroups();
        // calculate samplesize for each group
        Map<String,Integer> sizes = new HashMap<>();
        Map<String,Integer> sampleSizes = new HashMap<>();
        groups.equivalenceClasses.forEach((qid,eq) ->{
            sizes.put(qid,eq.size());
        });
        sample.forEach(row -> {
            Grouper.EquivalenceClass eq = reverseGroups[row];
            int value = sampleSizes.computeIfAbsent(eq.qid,(k) -> 0);
            sampleSizes.put(eq.qid,value+1);
        });

        // print sample distribution + original (qid,count)
        try(PrintWriter writer1 = new PrintWriter("groups.csv");){
            sizes.forEach((qid,eqsize) -> {
                writer1.println(qid+";"+eqsize+";"+sampleSizes.computeIfAbsent(qid,(k)->0));
            });
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }


    public static <T> JsonArray convertListToJsonArray(List<T> list){
        JsonArray jsonArray = new JsonArray();
        for (T item : list) {
            jsonArray.add(item.toString());
        }
        return jsonArray;
    }

    public static JsonArray convertArrayToJsonArray(int[] array) {
        JsonArray jsonArray = new JsonArray();
        for (int item : array) {
            jsonArray.add(item);
        }
        return jsonArray;
    }

    public static JsonArray convertArrayToJsonArray(double[] array) {
        JsonArray jsonArray = new JsonArray();
        for (double item : array) {
            jsonArray.add(item);
        }
        return jsonArray;
    }


    public static String[] getQID(String inputDataDefinitionPath) throws FileNotFoundException {
        FileReader fileReader = new FileReader(inputDataDefinitionPath);
        Gson gson = new Gson();
        JsonObject jsonObject = gson.fromJson(fileReader, JsonObject.class);

        JsonArray qidArray = jsonObject.getAsJsonArray("QID");
        String[] qid = new String[qidArray.size()];
        for (int i = 0; i < qidArray.size(); i++) {
            JsonObject qidObject = qidArray.get(i).getAsJsonObject();
            qid[i] = qidObject.get("colName").getAsString();
        }
        return qid;
    }

}
