package main;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

public class XOR_Demo {
	public static void main(String[] args) {
		double[][] XOR_Input = {
				new double[]{0.0, 0.0},
				new double[]{1.0, 0.0},
				new double[]{0.0, 1.0},
				new double[]{1.0, 1.0}
		};
		
		double[][] XOR_Ideal = {
				new double[]{0.0},
				new double[]{1.0},
				new double[]{1.0},
				new double[]{0.0}
		};
		
		BasicMLDataSet trainingSet = new BasicMLDataSet(XOR_Input, XOR_Ideal);
		
		BasicNetwork network = createNetwork();
		
		ResilientPropagation train = new ResilientPropagation(network, trainingSet);
		
		int epoch = 1;
		do{
			train.iteration();
			System.out.println("Epoch No " + epoch + ", Error: " + train.getError());
			epoch++;
		}while(train.getError() > 0.001);
		
		for (MLDataPair mlDataPair : trainingSet) {
			MLData data = network.compute(mlDataPair.getInput());
			System.out.println("Input: " + mlDataPair.getInput() + "Output: " + data.getData()[0]);
		}
		
	}
	
	private static BasicNetwork createNetwork(){
		BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(null, true, 2));
		network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 2));
		network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
		network.getStructure().finalizeStructure();
		network.reset();
		return network;
	}
}
