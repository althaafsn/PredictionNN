package cpen221.mp3.Model;

public class Layer {
    private int nodeSize;
    private int nextNodeSize;
    private double[][] weights;
    private double[][] weightGradients;
    private double[] biases;
    private double[] biasGradients;

    /*
     ABSTRACTION FUNCTION:
     - A layer is a collection of nodes that are connected to the next layer.
     The layer has a number of nodes and next nodes, and each node has a weight
     and a bias. The layer also has a weight gradient and a bias gradient.

     REPRESENTATION INVARIANT:
     - nodeSize > 0
     - nextNodeSize > 0
     - weights != null
     - weightGradients != null
     - biases != null
     - biasGradients != null
     */

    /**
     * Create a new layer with the given number of nodes and next nodes.
     * @param nodes the number of nodes
     * @param nextNodes the number of next nodes
     * Pre-condition: nodes > 0, nextNodes > 0
     * Post-condition: Creates a new layer with the given number of nodes and next nodes.
     */
    public Layer(int nodes, int nextNodes) {
        this.nodeSize = nodes;
        this.nextNodeSize = nextNodes;
        this.weights = new double[nodeSize][nextNodeSize];
        this.biases = new double[nextNodeSize];
        this.weightGradients = new double[nodeSize][nextNodeSize];
        this.biasGradients = new double[nextNodeSize];

        for (int i = 0; i < nodeSize; i++) {
            for (int j = 0; j < nextNodeSize; j++) {
                this.weights[i][j] = 0.5;
            }
        }
        for (int i = 0; i < nextNodeSize; i++) {
            this.biases[i] = 0.5;
        }
    }

    /**
     * Get the output of the layer.
     * @param inputs the inputs to the layer
     * @return the output of the layer
     */
    public double[] getOutput(double[] inputs, boolean isBoolean) {
        double[] values = new double[this.nextNodeSize];

        for (int i = 0; i < this.nextNodeSize; i++) {
            double sum = biases[i];
            for (int j = 0; j < this.nodeSize; j++) {
                sum += inputs[j] * weights[j][i];
            }
            if (isBoolean) {
                values[i] = activationFunctionBoolean(sum);
            } else {
                values[i] = activationFunctionDouble(sum);
            }
        }

        return values;
    }

    /**
     * The activation function for the layer.
     * @param x the input
     * @return the output of the activation function
     */
    private double activationFunctionBoolean(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    /**
     * The activation function for the layer.
     * @param x the input
     * @return the output of the activation function
     */
    private double activationFunctionDouble(double x) {
        if (x < 0) {
            return 0;
        } else {
            return x;
        }
    }

    /**
     * Get the cost of the node
     * @param output the output of the node
     * @param expected the expected output of the node
     * @return the cost of the node
     */
    public double nodeCost(double output, double expected) {
        return Math.abs(output - expected);
    }

    /**
     * Update the gradients of the layer.
     * @param learnRate the learning rate
     */
    public void updateGradients(double learnRate) {
        for (int i = 0; i < this.nodeSize; i++) {
            for (int j = 0; j < this.nextNodeSize; j++) {
                this.weights[i][j] -= learnRate * this.weightGradients[i][j];
            }
        }
        for (int i = 0; i < this.nextNodeSize; i++) {
            this.biases[i] -= learnRate * this.biasGradients[i];
        }
    }

    /**
     * Get the number of nodes in the layer.
     * @return the number of nodes in the layer
     */
    public int getNodeSize() {
        return nodeSize;
    }

    /**
     * Get the number of next nodes in the layer.
     * @return the number of next nodes in the layer
     */
    public int getNextNodeSize() {
        return nextNodeSize;
    }

    /**
     * Get the weights of the layer.
     * @return the weights of the layer
     */
    public double[][] getWeights() {
        return weights;
    }

    /**
     * Set the weight gradient of the layer.
     * @param weights the weight gradient of the layer
     */
    public void setWeights(double[][] weights) {
        this.weights = weights;
    }

    /**
     * Set the weight gradient of the layer.
     * @param node the node of the weight gradient
     * @param nextNode the next node of the weight gradient
     * @param gradient the gradient of the weight
     */
    public void setWeightGradients(int node, int nextNode, double gradient) {
        this.weightGradients[node][nextNode] = gradient;
    }

    /**
     * Get the biases of the layer.
     * @return the biases of the layer
     */
    public double[] getBiases() {
        return biases;
    }

    /**
     * Set the biases of the layer.
     * @param biases the biases of the layer
     */
    public void setBiases(double[] biases) {
        this.biases = biases;
    }

    /**
     * Set the bias gradient of the layer.
     * @param node the node of the bias gradient
     * @param gradient the gradient of the bias
     */
    public void setBiasGradients(int node, double gradient) {
        this.biasGradients[node] = gradient;
    }
}
