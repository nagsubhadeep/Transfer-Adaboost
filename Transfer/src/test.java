import java.io.FileReader;
import weka.core.Instances;
import weka.filters.*;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Standardize;
import weka.filters.unsupervised.attribute.StringToWordVector;



public class test
{

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception
	{
		// TODO Auto-generated method stub
		TransferAdaBoostExp tr = new TransferAdaBoostExp();
		//tr.setSourceDataFilenameList("train.arff");
		System.out.println(tr.getTechnicalInformation());
		tr.setSourceDataFilenameList("SentimentTrain2.arff");
		
//		System.out.println(tr.getSourceDataFilenameList());
		
//		Instances[] source = tr.getSourceData();
		Instances target = new Instances(new FileReader("DVDTrain2.arff"));
		Instances source = new Instances(new FileReader("SentimentTrain2.arff"));
		
//		Filter filter = new StringToWordVector();
//		filter.setInputFormat(target);
//		Instances f1 =Filter.useFilter(target, filter);
//		
		target.setClassIndex(target.numAttributes()-1);
		
		source.setClassIndex(target.numAttributes()-1);
		
		source.deleteWithMissingClass();
		target.deleteWithMissingClass();
	
		
//		Filter filter1 = new NumericToNominal();
//		filter1.setInputFormat(target);
//		Instances f2 = Filter.useFilter(target, filter1);
//		
//		
//		Filter filter = new StringToWordVector();
//		filter.setInputFormat(f2);
//		Instances f1 =Filter.useFilter(f2, filter);
//		
//		Standardize filter3 = new Standardize();
//		filter3.setInputFormat(f1);
//		Instances f0 = Filter.useFilter(f1, filter3);
		
		System.out.println();
		tr.buildClassifier(target);
		tr.toString();
		
		
		
	}


}
