package income;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;

import org.encog.app.analyst.AnalystFileFormat;
import org.encog.app.analyst.EncogAnalyst;
import org.encog.app.analyst.csv.normalize.AnalystNormalizeCSV;
import org.encog.app.analyst.csv.segregate.SegregateCSV;
import org.encog.app.analyst.csv.segregate.SegregateTargetPercent;
import org.encog.app.analyst.csv.shuffle.ShuffleCSV;
import org.encog.app.analyst.wizard.AnalystWizard;
import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.mathutil.Equilateral;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.persist.EncogDirectoryPersistence;
import org.encog.util.arrayutil.NormalizationAction;
import org.encog.util.csv.CSVFormat;
import org.encog.util.simple.EncogUtility;


public class IncomePrediction {

	public static void main(String[] args) {
		System.out.println("Shuffle file...");
		shuffle(FilesPath.BASE_FILE);

		System.out.println("Segregate file into training and evaluation...");
		segregate(FilesPath.SHUFFLED_BASE_FILE);

		System.out.println("Normalize data...");
		normalize();

		System.out.println("Create network...");
		createNetwork(FilesPath.TRAINED_NETWORK_FILE);

		System.out.println("Training network...");
		trainNetwork();

		System.out.println("Evaluate network...");
		evaluate();
	}

	public static void shuffle(String source){
		// Shuffle the CSV.
		ShuffleCSV shuffle = new ShuffleCSV();
		File csv = new File(source);
		shuffle.analyze(csv, true, CSVFormat.ENGLISH);
		shuffle.setProduceOutputHeaders(true);
		shuffle.process(new File(FilesPath.SHUFFLED_BASE_FILE));
	}

	public static void segregate(String source){
		// Segregate file into training and evaluation files
		SegregateCSV seg = new SegregateCSV();

		seg.getTargets().add(new SegregateTargetPercent(new File(FilesPath.TRAINING_FILE), 75));
		seg.getTargets().add(new SegregateTargetPercent(new File(FilesPath.EVALUATION_FILE), 25));
		seg.setProduceOutputHeaders(true);
		seg.analyze(new File(source), true, CSVFormat.ENGLISH);
		seg.process();
	}

	public static void normalize(){
		// Analyst
		EncogAnalyst analyst = new EncogAnalyst();

		// Wizard
		AnalystWizard wizard = new AnalystWizard(analyst);
		wizard.wizard(new File(FilesPath.BASE_FILE), true, AnalystFileFormat.DECPNT_COMMA);

		analyst.getScript().getNormalize().getNormalizedFields().get(0).setAction(NormalizationAction.Normalize);
		analyst.getScript().getNormalize().getNormalizedFields().get(1).setAction(NormalizationAction.Equilateral);
		analyst.getScript().getNormalize().getNormalizedFields().get(2).setAction(NormalizationAction.Normalize);
		analyst.getScript().getNormalize().getNormalizedFields().get(3).setAction(NormalizationAction.Equilateral);
		analyst.getScript().getNormalize().getNormalizedFields().get(4).setAction(NormalizationAction.Normalize);
		analyst.getScript().getNormalize().getNormalizedFields().get(5).setAction(NormalizationAction.Equilateral);
		analyst.getScript().getNormalize().getNormalizedFields().get(6).setAction(NormalizationAction.Equilateral);
		analyst.getScript().getNormalize().getNormalizedFields().get(7).setAction(NormalizationAction.Equilateral);
		analyst.getScript().getNormalize().getNormalizedFields().get(8).setAction(NormalizationAction.Equilateral);
		analyst.getScript().getNormalize().getNormalizedFields().get(9).setAction(NormalizationAction.Equilateral);
		analyst.getScript().getNormalize().getNormalizedFields().get(10).setAction(NormalizationAction.Normalize);
		analyst.getScript().getNormalize().getNormalizedFields().get(11).setAction(NormalizationAction.Normalize);
		analyst.getScript().getNormalize().getNormalizedFields().get(12).setAction(NormalizationAction.Normalize);
		analyst.getScript().getNormalize().getNormalizedFields().get(13).setAction(NormalizationAction.Equilateral);

		analyst.getScript().getNormalize().getNormalizedFields().get(14).setAction(NormalizationAction.Equilateral);



		// Norm for Training
		AnalystNormalizeCSV norm = new AnalystNormalizeCSV();
		norm.analyze(new File(FilesPath.TRAINING_FILE), true, CSVFormat.ENGLISH, analyst);
		norm.setProduceOutputHeaders(true);
		norm.normalize(new File(FilesPath.NORMALIZED_TRAINING_FILE));

		// Norm for evaluation
		norm.analyze(new File(FilesPath.EVALUATION_FILE), true, CSVFormat.ENGLISH, analyst);
		norm.normalize(new File(FilesPath.NORMALIZED_EVAL_FILE));

		// Save the Analyst file
		analyst.save(new File(FilesPath.ANALYST_FILE));	
	}

	public static void createNetwork(String networkFile){
		BasicNetwork network = new BasicNetwork();

		network.addLayer(new BasicLayer(new ActivationLinear(), true, 14));
		network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 10));
		network.addLayer(new BasicLayer(new ActivationTANH(), false, 2));
		network.getStructure().finalizeStructure();
		network.reset();
		EncogDirectoryPersistence.saveObject(new File(networkFile), network);
	}

	public static void trainNetwork(){
		BasicNetwork network = (BasicNetwork)EncogDirectoryPersistence.loadObject(new File(FilesPath.TRAINED_NETWORK_FILE));
		MLDataSet trainingSet = EncogUtility.loadCSV2Memory(FilesPath.NORMALIZED_TRAINING_FILE, network.getInputCount(),
				network.getOutputCount(), true, CSVFormat.ENGLISH, false);

		ResilientPropagation train = new ResilientPropagation(network, trainingSet);
		int epoch = 1;
		do{
			train.iteration();
			System.out.println("Epoch: " + epoch + ", error: " + train.getError());
			epoch++;
		}while(epoch <= 500);

		EncogDirectoryPersistence.saveObject(new File(FilesPath.TRAINED_NETWORK_FILE), network);
	}

	public static void evaluate(){
		BasicNetwork network = (BasicNetwork)EncogDirectoryPersistence.loadObject(new File(FilesPath.TRAINED_NETWORK_FILE));
		EncogAnalyst analyst = new EncogAnalyst();
		analyst.load(FilesPath.ANALYST_FILE);
		MLDataSet evaluationSet = EncogUtility.loadCSV2Memory(FilesPath.NORMALIZED_EVAL_FILE, network.getInputCount(),
				network.getOutputCount(), true, CSVFormat.ENGLISH, false);

		int count = 0;
		int CorrectCount = 0;

		for (MLDataPair item : evaluationSet) {
			count++;
			BasicMLData output = (BasicMLData) network.compute(item.getInput());
			int classCount = analyst.getScript().getNormalize().getNormalizedFields().get(14).getClasses().size();
			double normalizationHigh = analyst.getScript().getNormalize().getNormalizedFields().get(14).getNormalizedHigh();
			double normalizationLow = analyst.getScript().getNormalize().getNormalizedFields().get(14).getNormalizedLow();

			Equilateral eq = new Equilateral(classCount, normalizationHigh, normalizationLow);
			double[] idealArray = item.getIdealArray();

			int idealClassInt = eq.decode(idealArray);
			String idealClass = analyst.getScript().getNormalize().getNormalizedFields().get(14).getClasses().get(idealClassInt).getName();


			int predictedClassInt = eq.decode(output.getData());
			String predictedClass = analyst.getScript().getNormalize().getNormalizedFields().get(14).getClasses().get(predictedClassInt).getName();


			if (predictedClassInt == idealClassInt)CorrectCount++;

			System.out.println("Count: " + count + " - Ideal: "
					+ idealClass + " Predicted: " + predictedClass);
		}
		System.out.println("Total test count: " + count);
		System.out.println("Total correct prediction count: " + CorrectCount);
		System.out.println("% Success: " + (CorrectCount * 100)/count);
	}

	public void evaluateOne(){
		
		EncogAnalyst analyst = new EncogAnalyst();
		// Wizard
		AnalystWizard wizard = new AnalystWizard(analyst);
		wizard.wizard(new File(FilesPath.BASE_FILE), true, AnalystFileFormat.DECPNT_COMMA);
		String input = "34,Federal-gov,67083,Bachelors,13,Never-married,Exec-manageria,Unmarried,Asian-Pac-Islander,Male,1471,0,40,Cambodia";
		printToCSV(input);

		AnalystNormalizeCSV norm = new AnalystNormalizeCSV();
		norm.setProduceOutputHeaders(true);

		// Norm for evaluation
		norm.analyze(new File(FilesPath.SINGLE_EVAL), true, CSVFormat.ENGLISH, analyst);
		norm.normalize(new File(FilesPath.SINGLE_EVAL));

		// Save the Analyst file
		analyst.save(new File(FilesPath.ANALYST_FILE));	
		BasicNetwork network = (BasicNetwork)EncogDirectoryPersistence.loadObject(new File(FilesPath.TRAINED_NETWORK_FILE));
		
		//		AnalystNormalize norm = new AnalystNormalize(analyst.getScript());
		//		norm.
		//		
		//		BasicMLData data = new BasicMLData();

	}

	public void printToCSV(String line){
		try (PrintWriter out = new PrintWriter(FilesPath.SINGLE_EVAL)) {
			out.println(line);
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
