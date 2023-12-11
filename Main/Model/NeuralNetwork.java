package cpen221.mp3.Model;

import javax.management.modelmbean.ModelMBean;
import java.util.Arrays;
import cpen221.mp3.Model.Template;

public class NeuralNetwork {
    private Layer[] layers;

    /*
     ABSTRACTION FUNCTION:
     - A neural network is a collection of layers that are connected to each other.
     The neural network has a number of layers.
     - This neural network is constructed to predict the next steps of a given data input.

     REPRESENTATION INVARIANT:
     - layers != null
     */

    /**
     * Create a new neural network with the given layer sizes.
     * @param layerSizes an array of integers representing the number of nodes in each layer
     */
    public NeuralNetwork(int[] layerSizes) {
        layers = new Layer[layerSizes.length - 1];

        for (int i = 0; i < layerSizes.length - 1; i++) {
            layers[i] = new Layer(layerSizes[i], layerSizes[i + 1]);
        }
    }

    /**
     * Predict the next steps of the given data input.
     * @param dataInput an array of doubles representing the data input
     * @param steps the number of steps to predict
     * @param template the template of the neural network
     * @param isBoolean whether the neural network is predicting a boolean
     * @return an array of doubles representing the predictions
     */
    public double[] predict(double[] dataInput, int steps, Template template, boolean isBoolean) {
        double[] predictions = new double[steps];

        double[][] dataInputProcessed = processDataInput(dataInput, template);
        double[][] dataOutputProcessed = processDataOutput(dataInput, template);
        fit(dataInputProcessed, dataOutputProcessed, template.learnRate, template.iteration, isBoolean);
        double[] currentInput = dataInputProcessed[dataInputProcessed.length - 1];

        for (int i = 0; i < steps; i++) {
            double prediction;
            if (isBoolean) {
                prediction = getOutput(currentInput, isBoolean)[0] > 0.5 ? 1 : 0;
            } else {
                prediction = getOutput(currentInput, isBoolean)[0];
            }
            predictions[i] = prediction;
            for (int j = 0; j < currentInput.length - 1; j++) {
                currentInput[j] = currentInput[j + 1];
            }
            currentInput[currentInput.length - 1] = prediction;
        }
        return predictions;
    }

    /**
     * Predict the next steps of the given data input.
     * @param dataInput an array of doubles representing the data input
     * @param steps the number of steps to predict
     * @param template the template of the neural network
     * @param isBoolean whether the neural network is predicting a boolean
     * @return an array of doubles representing the predictions
     */
    public double[] predictTimeStamp(double[] dataInput, int steps, Template template, boolean isBoolean) {
        double[] predictions = new double[steps];
        double[] actualPredictions = new double[steps];

        double[][] dataInputProcessed = processDataInputTimeStamp(dataInput, template);
        double[][] dataOutputProcessed = processDataOutputTimeStamp(dataInput, template);
        fit(dataInputProcessed, dataOutputProcessed, template.learnRate, template.iteration, isBoolean);
        double[] currentInput = dataInputProcessed[dataInputProcessed.length - 1];

        for (int i = 0; i < steps; i++) {
            double prediction;
            if (isBoolean) {
                prediction = getOutput(currentInput, isBoolean)[0] > 0.5 ? 1 : 0;
            } else {
                prediction = getOutput(currentInput, isBoolean)[0];
            }
            predictions[i] = prediction;
            for (int j = 0; j < currentInput.length - 1; j++) {
                currentInput[j] = currentInput[j + 1];
            }
            currentInput[currentInput.length - 1] = prediction;

            if (i == 0) {
                actualPredictions[i] = dataInput[dataInput.length - 1] + predictions[i];
            } else {
                actualPredictions[i] = actualPredictions[i - 1] + predictions[i];
            }
        }

        return actualPredictions;
    }

    /**
     * Get the output of the neural network.
     * @param inputs the inputs of the neural network
     * @return the output of the neural network
     */
    private double[] getOutput(double[] inputs, boolean isBoolean) {
        double[] outputs = new double[inputs.length];
        System.arraycopy(inputs, 0, outputs, 0, inputs.length);
        for (Layer layer : layers) {
            outputs = layer.getOutput(outputs, isBoolean);
        }
        return outputs;
    }

    /**
     * Get the cost of the neural network.
     * @param input the inputs of the neural network
     * @param expectedOut the expected output of the neural network
     * @return the cost of the neural network
     */
    private double costOneData(double[] input, double[] expectedOut, boolean isBoolean) {
        double cost = 0;
        Layer outputLayer = layers[layers.length - 1];

        for (int node = 0; node < outputLayer.getNextNodeSize(); node++) {
            cost += outputLayer.nodeCost(getOutput(input, isBoolean)[node], expectedOut[node]);
        }

        return cost;
    }

    /**
     * Get the final cost of the neural network.
     * @param inputs the inputs of the neural network
     * @param expectedOutputs the expected outputs of the neural network
     * @return the final cost of the neural network
     */
    private double finalCost(double[][] inputs, double[][] expectedOutputs, boolean isBoolean) {
        double cost = 0;

        for (int i = 0; i < inputs.length; i++) {
            cost += costOneData(inputs[i], expectedOutputs[i], isBoolean);
        }

        return cost / inputs.length;
    }

    /**
     * Make the neural network learn.
     * @param inputs the inputs of the neural network
     * @param expectedOut the expected outputs of the neural network
     * @param learnRate the learning rate of the neural network
     */
    private void learn(double[][] inputs, double[][] expectedOut, double learnRate, boolean isBoolean) {
        double originalCost = finalCost(inputs, expectedOut, isBoolean);
        for (Layer layer : layers) {
            double h_VALUE = 0.0001;
            for (int node = 0; node < layer.getNodeSize(); node++) {
                for (int nextNode = 0; nextNode < layer.getNextNodeSize(); nextNode++) {
                    double[][] weights = layer.getWeights();
                    weights[node][nextNode] += h_VALUE;
                    layer.setWeights(weights);
                    double newCost = finalCost(inputs, expectedOut, isBoolean);
                    weights[node][nextNode] -= h_VALUE;
                    double gradient = (newCost - originalCost) / h_VALUE;
                    layer.setWeightGradients(node, nextNode, gradient);
                }
            }

            for (int node = 0; node < layer.getNextNodeSize(); node++) {
                double[] biases = layer.getBiases();
                biases[node] += h_VALUE;
                layer.setBiases(biases);
                double newCost = finalCost(inputs, expectedOut, isBoolean);
                biases[node] -= h_VALUE;
                double gradient = (newCost - originalCost) / h_VALUE;
                layer.setBiasGradients(node, gradient);
            }
        }

        for (Layer layer : layers) {
            layer.updateGradients(learnRate);
        }
    }

    /**
     * Fit the neural network.
     * @param inputs the inputs of the neural network
     * @param expectedOut the expected outputs of the neural network
     * @param learnRate the learning rate of the neural network
     * @param iterations the number of iterations of the neural network
     */
    private void fit(double[][] inputs, double[][] expectedOut, double learnRate, int iterations, boolean isBoolean) {
        for (int i = 0; i < iterations; i++) {
            learn(inputs, expectedOut, learnRate, isBoolean);
        }
    }

    /**
     * Process the data input so that it is usable by the model.
     * @param dataInput the data input
     * @param template the template of the neural network
     * @return the processed data input
     */
    private double[][] processDataInput(double[] dataInput, Template template) {
        int firstLayerSize = layers[0].getNodeSize();
        double[][] dataInputProcessed = new double[dataInput.length - firstLayerSize + 1][firstLayerSize];
        for (int i = 0; i < dataInput.length - firstLayerSize + 1; i++) {
            System.arraycopy(dataInput, i, dataInputProcessed[i], 0, firstLayerSize);
        }

        return dataInputProcessed;
    }

    /**
     * Process the data output so that it is usable by the model.
     * @param dataInput the data input
     * @param template the template of the neural network
     * @return the processed data output
     */
    private double[][] processDataOutput(double[] dataInput, Template template) {
        int lastLayerSize = layers[layers.length - 1].getNextNodeSize();
        double[][] dataOutputProcessed = new double[dataInput.length - lastLayerSize + 1][lastLayerSize];
        for (int i = 0; i < dataInput.length - lastLayerSize + 1; i++) {
            try {
                System.arraycopy(dataInput, i + 1, dataOutputProcessed[i], 0, lastLayerSize);
            } catch (Exception e) {
                System.arraycopy(dataInput, 0, dataOutputProcessed[i], 0, lastLayerSize);
            }
        }
        return dataOutputProcessed;
    }

    private double[][] processDataInputTimeStamp(double[] dataInput, Template template) {
        double[] newDataInput = new double[dataInput.length - 1];
        for (int i = 0; i < dataInput.length - 1; i++) {
            newDataInput[i] = dataInput[i + 1] - dataInput[i];
        }

        int firstLayerSize = layers[0].getNodeSize();
        double[][] dataInputProcessed = new double[newDataInput.length - firstLayerSize + 1][firstLayerSize];
        for (int i = 0; i < newDataInput.length - firstLayerSize + 1; i++) {
            System.arraycopy(newDataInput, i, dataInputProcessed[i], 0, firstLayerSize);
        }

        return dataInputProcessed;
    }

    private double[][] processDataOutputTimeStamp(double[] dataInput, Template template) {
        double[] newDataInput = new double[dataInput.length - 1];
        for (int i = 0; i < dataInput.length - 1; i++) {
            newDataInput[i] = dataInput[i + 1] - dataInput[i];
        }

        int lastLayerSize = layers[layers.length - 1].getNextNodeSize();
        double[][] dataOutputProcessed = new double[newDataInput.length - lastLayerSize + 1][lastLayerSize];
        for (int i = 0; i < newDataInput.length - lastLayerSize + 1; i++) {
            try {
                System.arraycopy(newDataInput, i + 1, dataOutputProcessed[i], 0, lastLayerSize);
            } catch (Exception e) {
                System.arraycopy(newDataInput, 0, dataOutputProcessed[i], 0, lastLayerSize);
            }
        }
        return dataOutputProcessed;
    }
}
