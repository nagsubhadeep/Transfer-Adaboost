/**
 *    TransferBoostExp.java
 *
 *    
 *    Based on AdaBoostM1.java by Eibe Frank and Len Trigg
 *
 */



import weka.classifiers.functions.SMO;
import weka.classifiers.meta.*;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;
import weka.core.TechnicalInformation;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.FileReader;
import java.io.IOException;
//import java.text.DecimalFormat;
import java.util.*;
import edu.umbc.cs.maple.utils.JavaUtils;
import edu.umbc.cs.maple.utils.WekaUtils;

/**
 * 
 * TransferAdaBoostExp.java
 * @author Subhadeep
 * 
 * 
 * Based on AdaBoostM1.java by Eibe Frank and Len Trigg
 *
 */
public class TransferAdaBoostExp extends AdaBoostM1
{

	
	/** for serialization */
	static final long serialVersionUID = 1L;

	/** Max num iterations tried to find classifier with non-zero error. */
	private static int MAX_NUM_RESAMPLING_ITERATIONS = 8;

	/** Array for storing the weights sum assigned to each source task. Inherited from AdaboostM1 library */
	protected double[][] m_sumSourceWeights;
	
	/** Array for storing the weights sum assigned to the target task. Inherited from AdaboostM1 library */
	protected double[] m_sumTargetWeights;

	/** The list of source task data files Inherited from AdaboostM1 library */
	protected String sourceFile = "";
	
	/** Matrix to store the alphas for each boosting iteration Inherited from AdaboostM1 library */
	protected double[][] m_alphas = null;

	/** Flag to enable/disable transfer Inherited from AdaboostM1 library */
	protected boolean m_UseTransfer = true;
	
	/** Flag whether to use early termination of the boosting iterations. Inherited from AdaboostM1 library */
	protected boolean m_EarlyTermination = true;


	/**
	 * Constructor to initialize the classifier.
	 */
	public TransferAdaBoostExp()
	{
		m_Classifier = new weka.classifiers.trees.DecisionStump();
	}
	
	




	/**
	 * Returns an instance of a TechnicalInformation object, containing 
	 * detailed information about the technical background of this classifier.
	 * @return the technical information about this classifier
	 */
	public TechnicalInformation getTechnicalInformation()
	{
		TechnicalInformation result;
		result = new TechnicalInformation(Type.TECHREPORT);
		result.setValue(Field.AUTHOR, "Deep, Advisor: Dr. Y.S. Kim");
		result.setValue(Field.TITLE, "Transfer based Learning");
		result.setValue(Field.YEAR, "2013");
		result.setValue(Field.ORGANIZATION, "Utah State University");
		result.setValue(Field.ADDRESS, "Logan, UT");
		return result;
	}
	



	//Accessor and Mutator for the Source Data File
	
	/**
	 * Sets the source data file
	 * @param sourceDataFilename the filename for the data to transfer
	 */
	public void setSourceDataFilenameList(String sourceDataFilenameList)
	{
		sourceFile = sourceDataFilenameList;
	}


	/**
	 * Get the source data filenames
	 * @return the source data filenames
	 */
	public String getSourceDataFilenameList()
	{
		return sourceFile;
	}


	/**
	 * loads the source data for transfer
	 * @throws Exception 
	 */
	public Instances[] getSourceData(Instances targetData) throws Exception
	{
		//declaring a dynamic array to hold source data
		ArrayList<Instances> sourceData = new ArrayList<Instances>();
		for (String sourceDataFilename : sourceFile.split(","))
		{
			if (sourceDataFilename.length() > 0)
			{
				try
				{
					Instances tempData = new Instances(new FileReader(sourceDataFilename));
					tempData.setClassIndex(tempData.numAttributes()-1);
//					tempData.deleteWithMissingClass();
					

					Filter filter1 = new NumericToNominal();
					filter1.setInputFormat(tempData);
					Instances f2 = Filter.useFilter(tempData, filter1);
					
					
					Filter filter = new StringToWordVector();
					filter.setInputFormat(f2);
					Instances sourceFiltrate =Filter.useFilter(f2, filter);
					
					transformToBinary.convertMulticlassToBinary(sourceFiltrate, "1", "sentiment");
		
					for(int i = 0; i<sourceFiltrate.numAttributes();i++)
					{
						for(int j = 0; j < targetData.numAttributes();j++)
						{
							if(sourceFiltrate.attribute(i).name().equals(targetData.attribute(j).name()))
							{
								sourceFiltrate.deleteAttributeAt(i);
							}
						}
					}
					sourceFiltrate.deleteWithMissingClass();
					sourceData.add(sourceFiltrate);
					
				}
				catch (IOException e)
				{
					throw new IllegalArgumentException("File \"" + sourceDataFilename + "\" is not a valid WEKA-compatible data file.");
				}
			}
		}
		
		return (Instances[]) sourceData.toArray(new Instances[0]);
	}



	/**
	 * Sets whether to use transfer
	 * @param b true for using transfer
	 */
	public void setEarlyTermination(boolean flag)
	{
		m_EarlyTermination = flag;
	}


	/**
	 * Gets whether to use transfer
	 * @returns whether transfer is being used
	 */
	public boolean getEarlyTermination()
	{
		return m_EarlyTermination;
	}
	
	
	/**
	 * Sets whether to use transfer
	 * @param b true for using transfer
	 */
	public void setUseTransfer(boolean flag)
	{
		m_UseTransfer = flag;
	}


	/**
	 * Gets whether to use transfer
	 * @returns a true or false statement basing on if the transfer is being used
	 */
	public boolean getUseTransfer()
	{
		return m_UseTransfer;
	}
	


	/**
	 * Boosting method.
	 *
	 * @param targetData the training data from the target task to be used for generating the boosted classifier.
	 * @throws Exception if the classifier could not be built successfully
	 */

	public void buildClassifier(Instances targetData) throws Exception
	{	
		
		System.out.println("Processing the Classifier...");
		
		//SVMClassify();
		
		super.buildClassifier(targetData);

		// can classifier handle the data?
		getCapabilities().testWithFail(targetData);

		// remove instances with missing class
		targetData = new Instances(targetData);
		targetData.deleteWithMissingClass();
		
		
		
		// get the source data
		System.out.println("Getting the Source Data...");
		Instances[] sourceData = getSourceData(targetData);
		
		// if transfer is disabled, then merge all the data together
		if (!getUseTransfer())
		{
			for (Instances data : sourceData)
			{
				targetData = Instances.mergeInstances(targetData,data);
			}
			sourceData = new Instances[0];  // no source instances
		}
		

		m_NumClasses = targetData.numClasses();
		if ((!m_UseResampling) && (m_Classifier instanceof WeightedInstancesHandler))
		{
			buildClassifierWithWeights(targetData, sourceData);
		}
		else
		{
			buildClassifierUsingResampling(targetData, sourceData);
		}
	}
	


	/**
	 * Boosting method. Boosts using resampling
	 *
	 * @param data the merged source transfer data and training data to be used for generating the boosted classifier.
	 * @param splitPoint the instance index at which the source transfer data stops and the training data begins.
	 * @throws Exception if the classifier could not be built successfully
	 */
	protected void buildClassifierUsingResampling(Instances targetData, Instances[] sourceData) throws Exception
	{

		Evaluation evaluation;
		
		Random randGenerator = new Random(m_Seed);
		int resamplingIterations = 0;

		// Initialize data
		m_Betas = new double[m_Classifiers.length];
		m_alphas = new double[m_Classifiers.length][sourceData.length+1]; // one more for target data
		m_sumSourceWeights = new double[m_Classifiers.length][sourceData.length];
		m_sumTargetWeights = new double[m_Classifiers.length];
		m_NumIterationsPerformed = 0;

		// normalize the weights
		double sumProbs = targetData.sumOfWeights();
		for (int k = 0; k < sourceData.length; k++)
		{
			sumProbs += sourceData[k].sumOfWeights();
		}
		int numInstancesTargetData = targetData.numInstances();
		for (int i=0; i < numInstancesTargetData; i++)
		{
			targetData.instance(i).setWeight(targetData.instance(i).weight() / sumProbs);
		}
		for (int k=0; k < sourceData.length; k++)
		{
			int numInstancesSourceDataK = sourceData[k].numInstances();
			for (int i=0; i < numInstancesSourceDataK; i++)
			{
				sourceData[k].instance(i).setWeight(sourceData[k].instance(i).weight() / sumProbs);
			}
		}
		
		Instances allData = null;
			
		// Do boostrap iterations
		for (m_NumIterationsPerformed = 0; m_NumIterationsPerformed < m_Classifiers.length; m_NumIterationsPerformed++)
		{
			if (m_Debug)
			{
				System.err.println("Training classifier " + (m_NumIterationsPerformed + 1));
			}
			
			// record the sum of the source and target weights
			m_sumTargetWeights[m_NumIterationsPerformed] = targetData.sumOfWeights();
			for (int k=0; k < sourceData.length; k++)
			{
				m_sumSourceWeights[m_NumIterationsPerformed][k] += sourceData[k].sumOfWeights();
			}
			
			// create the combined dataset
			allData = new Instances(targetData);
			for (int k=0; k < sourceData.length; k++)
			{
				allData = Instances.mergeInstances(allData, sourceData[k]);
			}	

			// Select instances to train the classifier on
			//   trainData corresponds to the S in the paper analysis with 
			//   the first elements trainData[0] as the targetData and 
			//   the other elements trainData[1...length] as source data
			Instances[] trainData = new Instances[sourceData.length+1];
			if (m_WeightThreshold < 100)
			{
				trainData[0] = selectWeightQuantile(targetData, (double) m_WeightThreshold / 100);
				for (int k=0; k < sourceData.length; k++)
				{
					trainData[k+1] = selectWeightQuantile(sourceData[k], (double) m_WeightThreshold / 100);
				}
			}
			else
			{
				trainData[0] = new Instances(targetData);
				for (int k=0; k < sourceData.length; k++)
				{
					trainData[k+1] = new Instances(sourceData[k]);
				}
			}
			
			// create the combined dataset for the training instances
			Instances trainAllData = new Instances(trainData[0]);
			for (int k=1; k < trainData.length; k++)
			{
				trainAllData = Instances.mergeInstances(trainAllData, trainData[k]);
			}	

			trainAllData.setClassIndex(trainAllData.numAttributes()-1);
			
			
//			SMO svm = new SMO();
//			svm.setNumFolds(6);
//			svm.buildClassifier(trainAllData);
//			System.out.println(svm.toString());
			
			// train the classifier using resampling
			resamplingIterations = 0;
			double[] weights = WekaUtils.getWeights(trainAllData);
			double epsilon;
			
			do
			{
				Instances sample = trainAllData.resampleWithWeights(randGenerator, weights);
				sample.setClassIndex(sample.numAttributes()-1);

				// Build and evaluate classifier
				m_Classifiers[m_NumIterationsPerformed].buildClassifier(sample);
				Instances evalData = allData;
				
				evalData.setClassIndex(evalData.numAttributes()-1);
				
				evaluation = new Evaluation(evalData);
				evaluation.evaluateModel(m_Classifiers[m_NumIterationsPerformed], evalData);
				epsilon = evaluation.errorRate();
				resamplingIterations++;
			}
			while (Utils.eq(epsilon, 0) && (resamplingIterations < MAX_NUM_RESAMPLING_ITERATIONS));
			
			

			if (m_EarlyTermination)
			{
				// Stop if error too big or 0
				if (Utils.grOrEq(epsilon, 0.5) || Utils.eq(epsilon, 0))
				{
					if (m_NumIterationsPerformed == 0)
					{
						m_NumIterationsPerformed = 1; // If we're the first we have to to use it
					}
					break;
				}
			}
			else
			{
				// ensure that we're not dividing by zero later
				if (Utils.eq(epsilon, 0))
				{
					epsilon = 0.000001;
					System.err.println("Warning:  epsilon ("+epsilon+") ~= 0");
				}
				// ensure that epsilon is <= 0.5
				if (epsilon > 0.5)
				{
					epsilon = 0.5;
					System.err.println("Warning:  epsilon ("+epsilon+") > 0.5");
				}
			}
			
			SMO svm = new SMO();
			svm.setNumFolds(6);
			
			// train a classifier solely on the target data
			Classifier targetClassifier = Classifier.makeCopy(m_Classifier);
			double[] targetWeights = WekaUtils.getWeights(trainData[0]);
			targetClassifier.buildClassifier(trainData[0].resampleWithWeights(randGenerator, targetWeights));
			
			// compute the baseline error without transfer
			Evaluation alphaEvaluation = new Evaluation(targetData);
			alphaEvaluation.evaluateModel(targetClassifier, targetData);
			double targetEpsilon = alphaEvaluation.errorRate();
			
			Instances evalData1 = targetData;
			evalData1.setClassIndex(evalData1.numAttributes()-1);
			
			Filter filter1 = new NumericToNominal();
			filter1.setInputFormat(evalData1);
			Instances FilterEval = Filter.useFilter(evalData1, filter1);
			FilterEval.setClassIndex(FilterEval.numAttributes()-1);
			
			// choose the alphas
			m_alphas[m_NumIterationsPerformed][0] = 0; // targetData alpha is 0
			for (int k=1; k<trainData.length; k++)
			{
				// train a classifier on both the target and source task
				Classifier sourceTargetClassifier = Classifier.makeCopy(m_Classifier);
				Instances sourceTargetData = Instances.mergeInstances(trainData[0], trainData[k]);
				sourceTargetData.setClassIndex(sourceTargetData.numAttributes()-1);
				
				//Changing the Numeric Class to a nominal class
				Filter filter2 = new NumericToNominal();
				filter2.setInputFormat(sourceTargetData);
				Instances FilterST = Filter.useFilter(sourceTargetData, filter2);
				FilterST.setClassIndex(FilterST.numAttributes()-1);
				
				double[] sourceTargetWeights = WekaUtils.getWeights(FilterST);
				
				svm.buildClassifier(FilterST);
				targetClassifier.buildClassifier(FilterST.resampleWithWeights(randGenerator, sourceTargetWeights));
				
				// compute the error with transfer
				alphaEvaluation = new Evaluation(FilterEval);
				alphaEvaluation.evaluateModel(sourceTargetClassifier, FilterEval);
				double sourceTargetEpsilon = alphaEvaluation.errorRate();
				
				// set alpha
				m_alphas[m_NumIterationsPerformed][k] = targetEpsilon - sourceTargetEpsilon;
			}
			
			
			// compute beta
			m_Betas[m_NumIterationsPerformed] = Math.log((1 - epsilon) / epsilon);
			if (m_Debug)
			{
				System.err.println("\terror rate = " + epsilon + "  beta = " + m_Betas[m_NumIterationsPerformed] + "  alphas = "+JavaUtils.arrayToString(m_alphas[m_NumIterationsPerformed]));
			}
			
			// Update instance weights
			setWeights(targetData, sourceData, m_Betas[m_NumIterationsPerformed], m_alphas[m_NumIterationsPerformed]);
			System.out.println("\nResults of the Boosted Classifier in Iteration# " +(m_NumIterationsPerformed+1)+ " basing on Source and Target Data using Resampling\n======\n");
			System.out.println(alphaEvaluation.toSummaryString());

			if (m_NumIterationsPerformed == m_Classifiers.length - 1)
			{
				
				System.out.println(alphaEvaluation.toClassDetailsString("\nResults of the Boosted Classifier basing on Source and Target Data using Resampling\n======\n"));
			}
		}
		
		if (m_Debug)
		{
			System.out.println(" final instance weights = ["+JavaUtils.arrayToString(WekaUtils.getWeights(allData), JavaUtils.format4DecimalPlaces)+"]");
		}
	}


	/**
	 * Sets the weights for the next iteration.
	 * 
	 * @param training the merged source and target training instances
	 * @param the instance index where the source data stops and the target begins
	 * @param reweight the reweighting factor
	 * @throws Exception if something goes wrong
	 */
	protected void setWeights(Instances targetData, Instances[] sourceData, double beta, double[] alpha) throws Exception
	{

		double oldSumOfWeights = targetData.sumOfWeights();
		@SuppressWarnings("unused")
		int numInstances = targetData.numInstances();
		for (int k=0; k<sourceData.length; k++)
		{
			oldSumOfWeights += sourceData[k].sumOfWeights();
			numInstances += sourceData[k].numInstances();
		}
		
		// reweight the source data
		for (int k=0; k<sourceData.length; k++)
		{
			
			double reweightAlpha = Math.exp(alpha[k+1]);
			double reweightAlphaBeta = Math.exp(beta + alpha[k+1]);
			
			int numInstancesSourceDataK = sourceData[k].numInstances();
			for (int i = 0; i < numInstancesSourceDataK; i++)
			{
				Instance instance = sourceData[k].instance(i);
				if (!Utils.eq(m_Classifiers[m_NumIterationsPerformed].classifyInstance(instance), instance.classValue()))
					instance.setWeight(instance.weight() * reweightAlphaBeta);
				else
					instance.setWeight(instance.weight() * reweightAlpha);
			}
		}
		
		// reweight the target data
		double reweightBeta = Math.exp(beta);
		int numInstancesTargetData = targetData.numInstances();
		for (int i = 0; i < numInstancesTargetData; i++)
		{
			Instance instance = targetData.instance(i);
			if (!Utils.eq(m_Classifiers[m_NumIterationsPerformed].classifyInstance(instance), instance.classValue()))
				instance.setWeight(instance.weight() * reweightBeta);
		}
		
		
		// Renormalize weights
		double newSumOfWeights = targetData.sumOfWeights();
		for (int k=0; k<sourceData.length; k++)
		{
			newSumOfWeights += sourceData[k].sumOfWeights();
		}
		
		double normalizationConstant = (oldSumOfWeights / newSumOfWeights);
		
		// reweight the data's weights
		for (int k=0; k<sourceData.length; k++)
		{
			int numInstancesSourceDataK = sourceData[k].numInstances();
			for (int i = 0; i < numInstancesSourceDataK; i++)
			{
				Instance instance = sourceData[k].instance(i);
				instance.setWeight(instance.weight() * normalizationConstant);
			}
		}
		for (int i = 0; i < numInstancesTargetData; i++)
		{
			Instance instance = targetData.instance(i);
			instance.setWeight(instance.weight() * normalizationConstant);
		}
	}


	/**
	 * Boosting method. Boosts any classifier that can handle weighted instances.
	 *
	 * @param data the merged source transfer data and training data to be used for generating the boosted classifier.
	 * @param splitPoint the instance index at which the source transfer data stops and the training data begins.
	 * @throws Exception if the classifier could not be built successfully
	 */
	protected void buildClassifierWithWeights(Instances targetData, Instances[] sourceData) throws Exception
	{

		Evaluation evaluation;
		
		Random randGenerator = new Random(m_Seed);

		// Initialize data
		m_Betas = new double[m_Classifiers.length];
		m_alphas = new double[m_Classifiers.length][sourceData.length+1]; // one more for target data
		m_sumSourceWeights = new double[m_Classifiers.length][sourceData.length];
		m_sumTargetWeights = new double[m_Classifiers.length];
		m_NumIterationsPerformed = 0;

		// normalize the weights
		double sumProbs = targetData.sumOfWeights();
		for (int k = 0; k < sourceData.length; k++)
		{
			sumProbs += sourceData[k].sumOfWeights();
		}
		int numInstancesTargetData = targetData.numInstances();
		for (int i=0; i < numInstancesTargetData; i++)
		{
			targetData.instance(i).setWeight(targetData.instance(i).weight() / sumProbs);
		}
		for (int k=0; k < sourceData.length; k++)
		{
			int numInstancesSourceDataK = sourceData[k].numInstances();
			for (int i=0; i < numInstancesSourceDataK; i++)
			{
				sourceData[k].instance(i).setWeight(sourceData[k].instance(i).weight() / sumProbs);
			}
		}
		
		Instances allData = null;
			
		// Do boostrap iterations
		for (m_NumIterationsPerformed = 0; m_NumIterationsPerformed < m_Classifiers.length; m_NumIterationsPerformed++)
		{
			if (m_Debug)
			{
				System.err.println("Training classifier " + (m_NumIterationsPerformed + 1));
			}
			
			// record the sum of the source and target weights
			m_sumTargetWeights[m_NumIterationsPerformed] = targetData.sumOfWeights();
			for (int k=0; k < sourceData.length; k++)
			{
				m_sumSourceWeights[m_NumIterationsPerformed][k] += sourceData[k].sumOfWeights();
			}
		
			allData = new Instances(targetData);
			for (int k=0; k < sourceData.length; k++)
			{
				allData = Instances.mergeInstances(allData, sourceData[k]);
			}	
			
			//for(int i = 0; i< sourceData.length;i++)

			// Select instances to train the classifier on
			//   trainData corresponds to the S in the paper analysis with 
			//   the first elements trainData[0] as the targetData and 
			//   the other elements trainData[1...length] as source data
			Instances[] trainData = new Instances[sourceData.length+1];
			if (m_WeightThreshold < 100)
			{
				trainData[0] = selectWeightQuantile(targetData, (double) m_WeightThreshold / 100);
				for (int k=0; k < sourceData.length; k++)
				{
					trainData[k+1] = selectWeightQuantile(sourceData[k], (double) m_WeightThreshold / 100);
				}
			}
			else
			{
				trainData[0] = new Instances(targetData);
				for (int k=0; k < sourceData.length; k++)
				{
					trainData[k+1] = new Instances(sourceData[k]);
				}
			}
			
			// create the combined dataset for the training instances
			Instances trainAllData = new Instances(trainData[0]);
			for (int k=1; k < trainData.length; k++)
			{
				trainAllData = Instances.mergeInstances(trainAllData, trainData[k]);
			}	
			
			trainAllData.setClassIndex(trainAllData.numAttributes()-1);
			
			// Build the classifier
			if (m_Classifiers[m_NumIterationsPerformed] instanceof Randomizable)
			{
				((Randomizable) m_Classifiers[m_NumIterationsPerformed]).setSeed(randGenerator.nextInt());
			}
//
			SMO svm = new SMO();
			svm.setNumFolds(6);
//			svm.buildClassifier(trainAllData);
//			System.out.println(svm.toString());
//			System.out.println(trainAllData.classAttribute().isNominal());
			m_Classifiers[m_NumIterationsPerformed].buildClassifier(trainAllData);
			
			

			// Evaluate the classifier on the target task data
			Instances evalData = allData; // targetData;
			evalData.setClassIndex(evalData.numAttributes()-1);
			evaluation = new Evaluation(evalData);
			evaluation.evaluateModel(m_Classifiers[m_NumIterationsPerformed], evalData);
			double epsilon = evaluation.errorRate();
			//System.out.println("\nClassifier results based on the current Target Data\n");
//			System.out.println(evaluation.toSummaryString("\nEvaluating the Classifier solely based on Target Data\n======\n", true));

			if (m_EarlyTermination)
			{
				// Stop if error too big or 0
				if (Utils.grOrEq(epsilon, 0.5) || Utils.eq(epsilon, 0))
				{
					if (m_NumIterationsPerformed == 0)
					{
						m_NumIterationsPerformed = 1; // If we're the first we have to to use it
					}
					break;
				}
			}
			else
			{
				// ensure that we're not dividing by zero later
				if (Utils.eq(epsilon, 0))
				{
					epsilon = 0.000001;
					System.err.println("Warning:  epsilon ("+epsilon+") ~= 0");
				}
				// ensure that epsilon is <= 0.5
				if (epsilon > 0.5)
				{
					epsilon = 0.5;
					System.err.println("Warning:  epsilon ("+epsilon+") > 0.5");
				}
			}
			
			
			
			// train a classifier solely on the target data
			Classifier targetClassifier = Classifier.makeCopy(m_Classifier);
			targetClassifier.buildClassifier(trainData[0]);
			
			// compute the baseline error without transfer
			Evaluation alphaEvaluation = new Evaluation(targetData);
			alphaEvaluation.evaluateModel(targetClassifier, targetData);
			double targetEpsilon = alphaEvaluation.errorRate();
			
			Instances evalData1 = targetData;
			evalData1.setClassIndex(evalData1.numAttributes()-1);
			
			Filter filter1 = new NumericToNominal();
			filter1.setInputFormat(evalData1);
			Instances FilterEval = Filter.useFilter(evalData1, filter1);
			FilterEval.setClassIndex(FilterEval.numAttributes()-1);
			// choose the alphas
			m_alphas[m_NumIterationsPerformed][0] = 0; // targetData alpha is 0
			for (int k=1; k<trainData.length; k++)
			{
				// train a classifier on both the target and source task
				Classifier sourceTargetClassifier = Classifier.makeCopy(m_Classifier);
				Instances sourceTargetData = Instances.mergeInstances(trainData[0], trainData[k]);
				sourceTargetData.setClassIndex(sourceTargetData.numAttributes()-1);
				
//				Changing the Numeric Class to a nominal class
				Filter filter2 = new NumericToNominal();
				filter2.setInputFormat(sourceTargetData);
				Instances FilterST = Filter.useFilter(sourceTargetData, filter2);
				FilterST.setClassIndex(FilterST.numAttributes()-1);

				svm.buildClassifier(FilterST);
//				System.out.println(svm.toString());
				sourceTargetClassifier.buildClassifier(FilterST);
				
				// compute the error with transfer
				alphaEvaluation = new Evaluation(FilterEval);
				alphaEvaluation.evaluateModel(sourceTargetClassifier, FilterEval);
				double sourceTargetEpsilon = alphaEvaluation.errorRate();
				
				// set alpha
				m_alphas[m_NumIterationsPerformed][k] = targetEpsilon - sourceTargetEpsilon;
				//System.out.println(alphaEvaluation.toClassDetailsString("\nResults of the Boosted Classifier basing on Source and Target Data\n======\n"));
			}
			
			
			
			// compute beta
			m_Betas[m_NumIterationsPerformed] = Math.log((1 - epsilon) / epsilon);
			if (m_Debug)
			{
				System.err.println("\terror rate = " + epsilon + "  beta = " + m_Betas[m_NumIterationsPerformed] + "  alphas = "+JavaUtils.arrayToString(m_alphas[m_NumIterationsPerformed]));
			}
			
			// Update instance weights
			setWeights(targetData, sourceData, m_Betas[m_NumIterationsPerformed], m_alphas[m_NumIterationsPerformed]);
			System.out.println("\nResults of the Boosted Classifier in Iteration# " +(m_NumIterationsPerformed+1)+ " basing on Source and Target Data\n======\n");
			System.out.println(alphaEvaluation.toSummaryString());

			if (m_NumIterationsPerformed == m_Classifiers.length - 1)
			{
				
				System.out.println(alphaEvaluation.toClassDetailsString("\nResults of the Boosted Classifier basing on Source and Target Data\n======\n"));
			}
		}
		
		if (m_Debug)
		{
			System.out.println(" final instance weights = ["+JavaUtils.arrayToString(WekaUtils.getWeights(allData), JavaUtils.format4DecimalPlaces)+"]");
		}	
	}



	/**
	 * Returns description of the boosted classifier.
	 *
	 * @return description of the boosted classifier as a string
	 */
	public String toString()
	{
		StringBuffer text = new StringBuffer();

		if (m_NumIterationsPerformed == 0)
		{
			text.append("TransferBoost: No model built yet.\n");
		}
		else if (m_NumIterationsPerformed == 1)
		{
			text.append("TransferBoost: No boosting possible, one classifier used!\n");
		}
		else
		{
			text.append("TransferBoost complete");
			text.append("Number of performed Iterations: " + m_NumIterationsPerformed + "\n");
		}

		return text.toString();
	}

}