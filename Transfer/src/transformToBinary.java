import weka.core.Instance;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instances;

public class transformToBinary
{
public static void convertMulticlassToBinary(Instances data, String positiveClassValue, String value)
{
		
		// ensure that data is nominal
		if (!data.classAttribute().isNominal()) 
			throw new IllegalArgumentException("Instances must have a nominal class.");

		
		// create the new class attribute
		FastVector newClasses = new FastVector(2);
		newClasses.addElement("Y");
		newClasses.addElement("N");
		Attribute newClassAttribute = new Attribute(value, newClasses);
		
		// alter the class attribute to be binary
		int newClassAttIdx = data.classIndex();
		data.insertAttributeAt(newClassAttribute, newClassAttIdx);
		int classAttIdx = data.classIndex();
		
		// set the instances classes to be binary, with the labels [Y,N] (indices 0 and 1 respectively)
		int numInstances = data.numInstances();
		for (int instIdx = 0; instIdx < numInstances; instIdx++)
		{
			Instance inst = data.instance(instIdx);
			if (inst.stringValue(classAttIdx).equals(positiveClassValue))
			{
				inst.setValue(newClassAttIdx, 0);  // set it to the first class, which will be Y
			} else {
				inst.setValue(newClassAttIdx, 1);  // set it to the second class, which will be 0
			}
		}

		// switch the class index to the new class and delete the old class
		data.setClassIndex(newClassAttIdx);
		data.deleteAttributeAt(classAttIdx);
		
		// alter the dataset name
		data.setRelationName(data.relationName()+"-"+positiveClassValue);
	}
}
