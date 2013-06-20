import java.io.FileReader;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.StringToWordVector;


public class testMain
{

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception
	{
		Instances target = new Instances(new FileReader("DVDTrain4.arff"));
		target.setClassIndex(target.numAttributes()-1);

		
		Filter filter1 = new NumericToNominal();
		filter1.setInputFormat(target);
		Instances f2 = Filter.useFilter(target, filter1);
		
		
		Filter filter = new StringToWordVector();
		filter.setInputFormat(f2);
		Instances newFiltrate =Filter.useFilter(f2, filter);

		transformToBinary.convertMulticlassToBinary(newFiltrate, "1", "classes");
		
//		System.out.println("Target"+newFiltrate.classAttribute().isNominal());
		
		  
		TransferAdaBoostExp tr = new TransferAdaBoostExp();
		System.out.println(tr.getTechnicalInformation());
		System.out.println();
		tr.setNumIterations(10);
		
		tr.setSourceDataFilenameList("SentimentTrain4.arff");
		tr.buildClassifier(newFiltrate);
		System.out.println(tr.toString());
	}

}
