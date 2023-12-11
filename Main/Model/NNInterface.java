package cpen221.mp3.Model;

import java.util.ArrayList;

public class NNInterface {

    public ModelType model;

    /*
     ABSTRACTION FUNCTION:
     This class provides a static method to predict the next steps of the given data input.

     REPRESENTATION INVARIANT:
     - dataInput != null
     - predictions != null
     - steps > 0
     - model != null
     */

    /**
     * Predict the next steps of the given data input.
     * @param dataInput an array of doubles representing the data input
     * @param steps the number of steps to predict
     * @param model the model of the neural network
     * @return an array of doubles representing the predictions
     * pre-condition: dataInput.length > 4, steps > 0, model != null
     */
    public static double[] predict(double[] dataInput, int steps, ModelType model) {
        double[] predictions;
        boolean isBoolean = model == ModelType.BOOLEAN;
        int numRepeat = 0;

        if (isBoolean) {
            numRepeat = getNumRepeat(dataInput);
        }

        Template template;
        switch (model) {
            case BOOLEAN:
                template = new Template(new int[]{numRepeat + 1, 1}, 0.5, 5000);
                predictions = template.neuralNetwork.predict(dataInput, steps, template, isBoolean);
                break;
            case DOUBLE:
                template = new Template(new int[]{4, 3, 1}, 0.000135, 65000);

                predictions = template.neuralNetwork.predict(dataInput, steps, template, isBoolean);
                if (checkUniqueValues(dataInput)) {
                    for (int i = 0; i < predictions.length; i++) {
                        if (i >= 2) {
                            predictions[i] = predictions[i - 2];
                        } else {
                            if (Math.abs(predictions[i] - dataInput[i])
                                    > Math.abs(predictions[i] - dataInput[i + 1])) {
                                predictions[i] = dataInput[i + 1];
                            } else {
                                predictions[i] = dataInput[i];
                            }
                        }
                    }
                }
                break;
            case DOUBLE_TIMESTAMP:
                template = new Template(new int[]{3, 6, 1}, 0.02, 20000);
                predictions = template.neuralNetwork.predictTimeStamp(dataInput, steps, template, isBoolean);
                break;
            default:
                template = new Template(new int[]{1, 1}, 0.05, 1000);
                predictions = template.neuralNetwork.predict(dataInput, steps, template, isBoolean);
                break;
        }

        for (int i = 0; i < predictions.length; i++) {
            predictions[i] = Math.round(predictions[i] * 1000.0) / 1000.0;
        }

        return predictions;
    }

    /**
     * Get the number of repeats in the given data input.
     * @param dataInput an array of doubles representing the data input
     * @return the number of repeats in the given data input
     */
    private static int getNumRepeat(double[] dataInput) {
        int numRepeatFirst = 1;
        int numRepeatSecond = 1;
        int numRepeat;
        int numRepeatChange = 0;

        for (int i = 0; i < dataInput.length - 1; i++) {
            if (numRepeatChange == 0) {
                if (dataInput[i] == dataInput[i + 1]) {
                    numRepeatFirst++;
                } else {
                    numRepeatChange++;
                }
            } else if (numRepeatChange == 1) {
                if (dataInput[i] == dataInput[i + 1]) {
                    numRepeatSecond++;
                } else {
                    numRepeatChange++;
                }
            }
            if (numRepeatChange == 2) {
                break;
            }
        }

        numRepeat = numRepeatFirst + numRepeatSecond;

        if (numRepeat > dataInput.length / 2) {
            numRepeat = dataInput.length / 2;
        }
        return numRepeat;
    }

    /**
     * Check if the given data input has unique values.
     * @param dataInput an array of doubles representing the data input
     * @return true if the given data input has unique values and false otherwise
     */
    private static boolean checkUniqueValues(double[] dataInput) {
        ArrayList<Double> uniqueValues = new ArrayList<>();
        int uniqueValue = 1;

        for (int i = 0; i < dataInput.length - 1; i++) {
            uniqueValues.add(dataInput[i]);
            if (dataInput[i] != dataInput[i + 1]
                    && !uniqueValues.contains(dataInput[i + 1])) {
                uniqueValue++;
            }
            if (uniqueValue > 2) {
                return false;
            }
        }
        return true;
    }
}
