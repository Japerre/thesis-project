package sampling;

import org.apache.mahout.math.Arrays;
import org.deidentifier.arx.DataHandle;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Grouper {

    public Map<String, EquivalenceClass> equivalenceClasses;
    public String target;
    public int rows;
    public Map<String, Integer> targetCounts = new HashMap<>();
    public Map<String, List<Integer>> targetGroups = new HashMap<>();

    public static class EquivalenceClass {
        public String qid;
        public List<Integer> rows = new ArrayList<>();
        public Map<String,List<Integer>> rowsByTarget = new HashMap<>();
        public Map<String, Integer> targetCounts = new HashMap<>();

        public void addRow(int row, String targetValue){
            rows.add(row);
            rowsByTarget.computeIfAbsent(targetValue,s -> new ArrayList<>()).add(row);
            targetCounts.put(targetValue,targetCounts.computeIfAbsent(targetValue,s -> 0)+1);
        }

        public EquivalenceClass(String qid){
            this.qid = qid;
        }

        public int size(){
            return rows.size();
        }
    }

    public Grouper(DataHandle output){
        this(output,"");
    }

    public Grouper(DataHandle output, String target){
        this(output,new ArrayList<>(output.getDefinition().getQuasiIdentifyingAttributes()),target);
    }

    public Grouper(DataHandle output, List<String> attributes, String target){
        this.target = target;
        this.rows = output.getNumRows();
        this.equivalenceClasses = new HashMap<>();
        groupyfy(output,attributes);
    }

    public EquivalenceClass[] inverseGroups(){
        EquivalenceClass[] reverse = new EquivalenceClass[this.rows];
        this.equivalenceClasses.forEach((qid,eq) -> {
            eq.rows.forEach(row -> reverse[row] = eq);
        });
        return reverse;
    }

    private List<Map<String,Integer>> valueIndexMap;

    private String getQID(DataHandle output, int row, int[] indices) {
        int[] qidValues = new int[indices.length];
        for(int i=0; i < indices.length; i++){
            String qidValue = output.getValue(row, indices[i]);
            Map<String,Integer> indexMap = valueIndexMap.get(i);
            qidValues[i] = indexMap.computeIfAbsent(qidValue,s -> indexMap.size());
        }
        return Arrays.toString(qidValues);
    }

    private void groupyfy(DataHandle output, List<String> attributes){
        valueIndexMap = new ArrayList<>(attributes.size());
        for(int i=0;i<attributes.size();i++){
            valueIndexMap.add(new HashMap<>());
        }
        int[] indices = new int[attributes.size()];
        int index = 0;
        for (String attribute : attributes) {
//            System.out.println("attribute = " + attribute);
//            System.out.println(output.getColumnIndexOf(attribute));
            indices[index++] = output.getColumnIndexOf(attribute);
        }
        // group rows in EQ
        int targetCol = output.getColumnIndexOf(this.target);
        for (int row = 0; row < output.getNumRows(); row++) {
            String targetValue = targetCol==-1?"":output.getValue(row,targetCol);
            this.add(getQID(output,row,indices), targetValue, row);
        }
    }

    public EquivalenceClass get(String qid){
        return equivalenceClasses.get(qid);
    }

    private void add(String qid,String targetValue, int row){
        this.targetCounts.put(targetValue,targetCounts.computeIfAbsent(targetValue,s -> 0)+1);
        this.targetGroups.computeIfAbsent(targetValue,s -> new ArrayList<>()).add(row);
        this.equivalenceClasses.computeIfAbsent(qid, EquivalenceClass::new).addRow(row,targetValue);
    }

}
