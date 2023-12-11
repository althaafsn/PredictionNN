package cpen221.mp3.Model;


public class Template {

    public NeuralNetwork neuralNetwork;
    public double learnRate;
    public int iteration;

    public Template(int[] layerSizes, double learnRate, int iteration) {
        this.neuralNetwork = new NeuralNetwork(layerSizes);
        this.learnRate = learnRate;
        this.iteration = iteration;
    }
}
