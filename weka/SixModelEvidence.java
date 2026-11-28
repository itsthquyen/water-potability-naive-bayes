import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import javax.imageio.ImageIO;
import javax.swing.*;
import weka.classifiers.*;
import weka.classifiers.trees.*;
import weka.classifiers.lazy.IBk;
import weka.classifiers.functions.*;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.Discretize;
import weka.gui.AttributeSelectionPanel;
import weka.gui.explorer.Explorer;

/** Chạy bộ phân lớp và bộ lọc Weka thật, xuất log, dự đoán và panel gốc. */
public class SixModelEvidence {
    static Instances read(Path folder,String name) throws Exception {
        Instances data=DataSource.read(folder.resolve(name).toString());
        data.setClassIndex(data.numAttributes()-1);
        return data;
    }
    static AttributeSelectionPanel list(Container component) {
        for(Component child:component.getComponents()) {
            if(child instanceof AttributeSelectionPanel) return (AttributeSelectionPanel)child;
            if(child instanceof Container) { AttributeSelectionPanel found=list((Container)child); if(found!=null)return found; }
        }
        return null;
    }
    static void layout(Container panel) {
        panel.doLayout();
        for(Component child:panel.getComponents()) if(child instanceof Container) layout((Container)child);
    }
    static void paint(JComponent panel,Path output) throws Exception {
        panel.setSize(1240,800); layout(panel);
        BufferedImage image=new BufferedImage(1240,800,BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics=image.createGraphics(); graphics.setColor(Color.WHITE);graphics.fillRect(0,0,1240,800);
        panel.paint(graphics);graphics.dispose();ImageIO.write(image,"png",output.toFile());
    }
    static Explorer preprocess(Instances data,String selected) throws Exception {
        Explorer explorer=new Explorer();explorer.getPreprocessPanel().setInstances(data);
        AttributeSelectionPanel attributes=list(explorer.getPreprocessPanel());
        int col=data.attribute(selected).index(); attributes.getSelectionModel().setSelectionInterval(col,col);
        return explorer;
    }
    static void output(Explorer explorer,String text,Classifier model,Path testPath) throws Exception {
        explorer.getTabbedPane().setSelectedIndex(1);
        Component classifierPanel=explorer.getTabbedPane().getComponentAt(1);
        Field editorField=classifierPanel.getClass().getDeclaredField("m_ClassifierEditor");
        editorField.setAccessible(true);
        ((weka.gui.GenericObjectEditor)editorField.get(classifierPanel)).setValue(model);
        Field modeField=classifierPanel.getClass().getDeclaredField("m_TestSplitBut");
        modeField.setAccessible(true);((JRadioButton)modeField.get(classifierPanel)).setSelected(true);
        weka.core.converters.ArffLoader loader=new weka.core.converters.ArffLoader();
        loader.setFile(testPath.toFile());
        Field loaderField=classifierPanel.getClass().getDeclaredField("m_TestLoader");
        loaderField.setAccessible(true);loaderField.set(classifierPanel,loader);
        Field field=classifierPanel.getClass().getDeclaredField("m_OutText");
        field.setAccessible(true);JTextArea area=(JTextArea)field.get(classifierPanel);
        area.setText(text);area.setCaretPosition(0);area.setFont(new Font("Monospaced",Font.PLAIN,13));
    }
    public static void main(String[] args) throws Exception {
        Path folder=Paths.get(args[0]);
        Instances raw=read(folder,"development_raw.arff");
        ReplaceMissingValues replace=new ReplaceMissingValues();replace.setInputFormat(raw);
        Instances filled=Filter.useFilter(raw,replace);
        Discretize disc=new Discretize();disc.setBins(3);disc.setUseEqualFrequency(true);disc.setInputFormat(filled);
        Instances binned=Filter.useFilter(filled,disc);
        Files.write(folder.resolve("weka_mean_filled.arff"),filled.toString().getBytes(StandardCharsets.UTF_8));
        Files.write(folder.resolve("weka_discretized.arff"),binned.toString().getBytes(StandardCharsets.UTF_8));
        Instances nominal=read(folder,"development_nominal.arff"), nominalTest=read(folder,"test_nominal.arff");
        Instances numeric=read(folder,"development_numeric.arff"), numericTest=read(folder,"test_numeric.arff");
        LinkedHashMap<String,Classifier> models=new LinkedHashMap<>();
        models.put("naive_bayes",new weka.classifiers.bayes.DirectFrequencyNB());
        models.put("id3",(Classifier)Class.forName("weka.classifiers.trees.Id3").newInstance());
        J48 tree=new J48();tree.setUnpruned(false);tree.setConfidenceFactor(0.25f);
        tree.setMinNumObj(2);tree.setCollapseTree(true);tree.setSubtreeRaising(true);models.put("c45",tree);
        IBk knn=new IBk(5);
        weka.core.EuclideanDistance distance=new weka.core.EuclideanDistance();distance.setDontNormalize(true);
        knn.getNearestNeighbourSearchAlgorithm().setDistanceFunction(distance);models.put("knn",knn);
        Logistic logistic=new Logistic();logistic.setRidge(0.001);logistic.setMaxIts(400);models.put("logistic",logistic);
        SMO svm=new SMO();svm.setC(1.0);svm.setFilterType(new SelectedTag(SMO.FILTER_NONE,SMO.TAGS_FILTER));
        PolyKernel kernel=new PolyKernel();kernel.setExponent(1);svm.setKernel(kernel);svm.setRandomSeed(42);models.put("svm",svm);
        LinkedHashMap<String,String> logs=new LinkedHashMap<>();
        for(Map.Entry<String,Classifier> entry:models.entrySet()) {
            String name=entry.getKey();boolean categorical=Arrays.asList("naive_bayes","id3").contains(name);
            Instances train=name.equals("c45")?raw:categorical?nominal:numeric;
            Instances test=name.equals("c45")?read(folder,"test_raw.arff"):categorical?nominalTest:numericTest;
            Classifier model=entry.getValue();long start=System.nanoTime();model.buildClassifier(train);
            double seconds=(System.nanoTime()-start)/1e9;
            Evaluation eval=new Evaluation(train);eval.evaluateModel(model,test);
            String text="=== Weka "+Version.VERSION+" - "+name+" ===\n\nScheme: "+model.getClass().getName()+
              "\nOptions: "+Utils.joinOptions(((OptionHandler)model).getOptions())+"\nTrain: "+train.numInstances()+"\nTest: "+test.numInstances()+
              "\nFit seconds: "+seconds+"\n\n"+eval.toSummaryString()+"\n"+eval.toClassDetailsString()+"\n"+eval.toMatrixString()+"\n"+model.toString();
            Files.write(folder.resolve(name+"_weka.txt"),text.getBytes(StandardCharsets.UTF_8));logs.put(name,text);
            try(PrintWriter writer=new PrintWriter(Files.newBufferedWriter(folder.resolve(name+"_weka_predictions.csv"),StandardCharsets.UTF_8))) {
                writer.println("test_position,actual,predicted,score_1");
                for(int i=0;i<test.numInstances();i++) {
                    Instance row=test.instance(i);double[] probabilities=model.distributionForInstance(row);
                    double prediction=model.classifyInstance(row);
                    String label=Double.isNaN(prediction)?"":Integer.toString((int)prediction);
                    String score=Double.isNaN(prediction)?"":Double.toString(probabilities[1]);
                    writer.println(i+","+(int)row.classValue()+","+label+","+score);
                }
            }
            System.out.println(name+" Weka accuracy="+eval.pctCorrect());
        }
        // Cùng năm fold được Python xuất theo ID, không dùng CV ngẫu nhiên khác.
        for(int fold=0;fold<5;fold++) {
            Instances train=read(folder,"cv/fold_"+fold+"_train.arff");
            Instances test=read(folder,"cv/fold_"+fold+"_validation.arff");
            J48 model=new J48();model.setConfidenceFactor(0.25f);model.setMinNumObj(2);
            model.buildClassifier(train);
            try(PrintWriter writer=new PrintWriter(Files.newBufferedWriter(folder.resolve("c45_fold_"+fold+"_weka_predictions.csv"),StandardCharsets.UTF_8))) {
                writer.println("validation_position,actual,predicted,score_1");
                for(int i=0;i<test.numInstances();i++) {
                    Instance row=test.instance(i);
                    writer.println(i+","+(int)row.classValue()+","+(int)model.classifyInstance(row)+","+model.distributionForInstance(row)[1]);
                }
            }
        }
        LinkedHashMap<String,Explorer> panels=new LinkedHashMap<>();
        SwingUtilities.invokeAndWait(()->{
            try {
                panels.put("01-weka-raw",preprocess(raw,"Sulfate"));
                panels.put("02-weka-mean",preprocess(filled,"Sulfate"));
                panels.put("03-weka-discretize",preprocess(binned,"Sulfate"));
                for(String name:models.keySet()) {
                    boolean isNominal=Arrays.asList("naive_bayes","id3").contains(name);
                    Explorer explorer=preprocess(name.equals("c45")?raw:isNominal?nominal:numeric,"Potability");
                    output(explorer,logs.get(name),models.get(name),folder.resolve(name.equals("c45")?"test_raw.arff":isNominal?"test_nominal.arff":"test_numeric.arff"));
                    panels.put("result-"+name,explorer);
                }
                for(Map.Entry<String,Explorer> e:panels.entrySet()) paint(e.getValue(),folder.resolve(e.getKey()+".png"));
            }catch(Exception e){throw new RuntimeException(e);}
        });
        Thread.sleep(1000);
        SwingUtilities.invokeAndWait(()->{
            try {for(Map.Entry<String,Explorer> e:panels.entrySet())paint(e.getValue(),folder.resolve(e.getKey()+".png"));}
            catch(Exception e){throw new RuntimeException(e);}
        });
        System.out.println("Exported 9 Weka panels, filter outputs and six evaluations.");System.exit(0);
    }
}
