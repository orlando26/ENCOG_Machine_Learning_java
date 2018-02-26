package classification;

import java.io.File;

import org.encog.Encog;
import org.encog.app.analyst.AnalystFileFormat;
import org.encog.app.analyst.EncogAnalyst;
import org.encog.app.analyst.csv.normalize.AnalystNormalizeCSV;
import org.encog.app.analyst.csv.segregate.SegregateCSV;
import org.encog.app.analyst.csv.segregate.SegregateTargetPercent;
import org.encog.app.analyst.csv.shuffle.ShuffleCSV;
import org.encog.app.analyst.script.normalize.AnalystNormalize;
import org.encog.app.analyst.wizard.AnalystWizard;
import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.mathutil.Equilateral;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.persist.EncogDirectoryPersistence;
import org.encog.util.arrayutil.ClassItem;
import org.encog.util.csv.CSVFormat;
import org.encog.util.simple.EncogUtility;

public class IrisClassification {
	public static void main(String[] args) {
//		System.out.println("shuffle CSV Data file...");
//		shuffle(Config.BASE_FILE);
//		
//		System.out.println("Segregate csv into traianing and evaluation...");
//		segregate(Config.SHUFFLED_BASE_FILE);
//		
//		System.out.println("Normalize data...");
//		normalize();
//		
//		System.out.println("Create Network...");
//		createNetwork(Config.TRAINED_NETWORK_FILE);
//		
//		System.out.println("Train network...");
//		trainNetwork();
		
		System.out.println("Evaluate network...");
		evaluate();
	}
	
	public static void shuffle(String source){
		// Shuffle the CSV.
		ShuffleCSV shuffle = new ShuffleCSV();
		File csv = new File(source);
		shuffle.analyze(csv, true, CSVFormat.ENGLISH);
		shuffle.setProduceOutputHeaders(true);
		shuffle.process(new File(Config.SHUFFLED_BASE_FILE));
	}
	
	public static void segregate(String source){
		// Segregate file into training and evaluation files
		SegregateCSV seg = new SegregateCSV();
		
		seg.getTargets().add(new SegregateTargetPercent(new File(Config.TRAINING_FILE), 75));
		seg.getTargets().add(new SegregateTargetPercent(new File(Config.EVALUATION_FILE), 25));
		seg.setProduceOutputHeaders(true);
		seg.analyze(new File(source), true, CSVFormat.ENGLISH);
		seg.process();
	}
	
	public static void normalize(){
		// Analyst
		EncogAnalyst analyst = new EncogAnalyst();
		
		// Wizard
		AnalystWizard wizard = new AnalystWizard(analyst);
		wizard.wizard(new File(Config.BASE_FILE), true, AnalystFileFormat.DECPNT_COMMA);
		
		
		// Norm for Training
		AnalystNormalizeCSV norm = new AnalystNormalizeCSV();
		norm.analyze(new File(Config.TRAINING_FILE), true, CSVFormat.ENGLISH, analyst);
		norm.setProduceOutputHeaders(true);
		norm.normalize(new File(Config.NORMALIZED_TRAINING_FILE));
		
		// Norm for evaluation
		norm.analyze(new File(Config.EVALUATION_FILE), true, CSVFormat.ENGLISH, analyst);
		norm.normalize(new File(Config.NORMALIZED_EVAL_FILE));
		
		// Save the Analyst file
		analyst.save(new File(Config.ANALYST_FILE));	
	}
	
	public static void createNetwork(String networkFile){
		BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(new ActivationLinear(), true, 4));
		network.addLayer(new BasicLayer(new ActivationTANH(), true, 6));
		network.addLayer(new BasicLayer(new ActivationTANH(), true, 2));
		
		network.getStructure().finalizeStructure();
		network.reset();
		EncogDirectoryPersistence.saveObject(new File(networkFile), network);
	}
	
	public static void trainNetwork(){
		BasicNetwork network = (BasicNetwork)EncogDirectoryPersistence.loadObject(new File(Config.TRAINED_NETWORK_FILE));
		MLDataSet trainingSet = EncogUtility.loadCSV2Memory(Config.NORMALIZED_TRAINING_FILE, network.getInputCount(),
				network.getOutputCount(), true, CSVFormat.ENGLISH, false);
		
		ResilientPropagation train = new ResilientPropagation(network, trainingSet);
		int epoch = 1;
		do{
			train.iteration();
			System.out.println("Epoch: " + epoch + ", error: " + train.getError());
			epoch++;
		}while(train.getError() > 0.01);
		
		EncogDirectoryPersistence.saveObject(new File(Config.TRAINED_NETWORK_FILE), network);
	}
	
	public static void evaluate(){
		BasicNetwork network = (BasicNetwork)EncogDirectoryPersistence.loadObject(new File(Config.TRAINED_NETWORK_FILE));
		EncogAnalyst analyst = new EncogAnalyst();
		
		analyst.load(new File(Config.ANALYST_FILE));
		MLDataSet evaluationSet = EncogUtility.loadCSV2Memory(Config.NORMALIZED_EVAL_FILE, network.getInputCount(),
				network.getOutputCount(), true, CSVFormat.ENGLISH, false);
		
		int count = 0;
		int CorrectCount = 0;
		
		for (MLDataPair item : evaluationSet) {
			count++;
			MLData output = network.compute(item.getInput());
			
			double sepal_l = analyst.getScript().getNormalize().getNormalizedFields().get(0).deNormalize(item.getInputArray()[0]);
			double sepal_w = analyst.getScript().getNormalize().getNormalizedFields().get(1).deNormalize(item.getInputArray()[1]);
			double petal_l = analyst.getScript().getNormalize().getNormalizedFields().get(2).deNormalize(item.getInputArray()[2]);
			double petal_w = analyst.getScript().getNormalize().getNormalizedFields().get(3).deNormalize(item.getInputArray()[3]);
			
			int classCount = analyst.getScript().getNormalize().getNormalizedFields().get(4).getClasses().size();
			double normalizationHigh = analyst.getScript().getNormalize().getNormalizedFields().get(4).getNormalizedHigh();
			double normalizationLow = analyst.getScript().getNormalize().getNormalizedFields().get(4).getNormalizedLow();
			
			Equilateral eq = new Equilateral(classCount, normalizationHigh, normalizationLow);
			int predictedClassInt = eq.decode(output.getData());
			String predictedClass = analyst.getScript().getNormalize().getNormalizedFields().get(4).getClasses().get(predictedClassInt).getName();
			
			int idealClassInt = eq.decode(item.getIdealArray());
			String idealClass = analyst.getScript().getNormalize().getNormalizedFields().get(4).getClasses().get(idealClassInt).getName();
			
			if (predictedClassInt == idealClassInt)CorrectCount++;
			
			System.out.println("Count: " + count + "Properties: [" + sepal_l + ", " + sepal_w + ", " + petal_l + ", " + petal_w + "], Ideal: "
					+ idealClass + " Predicted: " + predictedClass);
		}
		System.out.println("Total test count: " + count);
		System.out.println("Total correct prediction count: " + CorrectCount);
		System.out.println("% Success: " + (CorrectCount * 100)/count);
	}
	
}
