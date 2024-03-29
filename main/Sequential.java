public class Sequential {
    Layer headLayer;
    Tensor2D result = null;
    Sequential(Layer[] layers) {
        this.headLayer = layers[0];
        for (int i = 0; i < layers.length; i++) {
            if (i < layers.length - 1) layers[i].nextLayer = layers[i + 1];
            if (i > 0) layers[i].previousLayer = layers[i - 1];
            layers[i].init();
        }
    }

    public void fit(Tensor2D input, Tensor2D target, int epochs, double learningRate) {
        System.out.println("Fitting...");
        // for (int i = 0; i < epochs; i++) {
        //     headLayer.input = input.transpose();
        //     // headLayer.target = target.transpose();
        //     // headLayer.forward();
        //     // headLayer.backward();
        //     // headLayer.updateWeights(learningRate);
        // }
    }

    public Tensor2D predict(Tensor2D input) {
        System.out.println("Predicting...");
        headLayer.input = input.transpose();
        Tensor2D result = headLayer.forward().transpose();
        this.result = result;
        return result;
    }

    public void print_result() {
        // Iterate through each sample:
        for (int i = 0; i < this.result.nrows; i++) {
            System.out.print("Sample " + (i + 1) + ": ");
            // Iterate through each feature:
            for (int j = 0; j < this.result.ncols; j++) {
                System.out.print(this.result.data[i][j] + " ");
            }
            System.out.println();
        }
    }

    public void summary() {
        Layer currentLayer = headLayer;
        while (currentLayer != null) {
            System.out.println(currentLayer);
            currentLayer = currentLayer.nextLayer;
        }
    }
}
