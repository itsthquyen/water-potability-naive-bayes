package weka.classifiers.bayes;

import java.math.BigInteger;
import java.math.BigDecimal;
import java.math.MathContext;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Capabilities;

/** Naive Bayes danh nghĩa, ước lượng tần suất trực tiếp để đối chiếu Python. */
public class DirectFrequencyNB extends AbstractClassifier {
    int[] totals;
    int[][][] counts;
    int dimensions;
    public Capabilities getCapabilities() {
        Capabilities result=super.getCapabilities();result.disableAll();
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        return result;
    }
    public void buildClassifier(Instances data) {
        dimensions=data.numAttributes()-1;totals=new int[2];counts=new int[2][dimensions][];
        for(int c=0;c<2;c++)for(int j=0;j<dimensions;j++)counts[c][j]=new int[data.attribute(j).numValues()];
        for(Instance row:data) {
            int c=(int)row.classValue();totals[c]++;
            for(int j=0;j<dimensions;j++)counts[c][j][(int)row.value(j)]++;
        }
        if(totals[0]==0 || totals[1]==0)throw new IllegalArgumentException("Both classes required");
    }
    public double[] distributionForInstance(Instance row) {
        BigInteger[] a=new BigInteger[2],b=new BigInteger[2];
        for(int c=0;c<2;c++) {
            a[c]=BigInteger.valueOf(totals[c]);
            b[c]=BigInteger.valueOf(totals[0]+totals[1]).multiply(BigInteger.valueOf(totals[c]).pow(dimensions));
            for(int j=0;j<dimensions;j++)a[c]=a[c].multiply(BigInteger.valueOf(counts[c][j][(int)row.value(j)]));
        }
        BigInteger x=a[0].multiply(b[1]),y=a[1].multiply(b[0]),sum=x.add(y);
        if(sum.signum()==0)throw new IllegalArgumentException("Zero evidence in both classes");
        return new double[]{new BigDecimal(x).divide(new BigDecimal(sum),MathContext.DECIMAL128).doubleValue(),
                            new BigDecimal(y).divide(new BigDecimal(sum),MathContext.DECIMAL128).doubleValue()};
    }
    public String toString() { return "Direct frequency Naive Bayes; no pseudo-count; nominal attributes; classes 0,1."; }
}
