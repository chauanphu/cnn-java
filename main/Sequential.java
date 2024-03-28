public class Sequential {
    Layer[] layers;
    Sequential(Layer[] layers) {
        this.layers = layers;
    }

    public Tensor2D predict(Tensor2D input) {
        Tensor2D previousInput = input.transpose();
        Tensor2D output = null;
        int m = input.nrows;
        layers[0].input = previousInput;
        layers[0].initiate_weights(layers[0].nNodes, input.ncols);
        layers[0].initiate_intercept(layers[0].nNodes, m);
        for (int i = 1; i < layers.length; i++) {
            // a = W * x + b
            // Input layer's dimensions = [n_prev, m]
            layers[i].input = previousInput;
            // W's dimensions =  [n_curr, n_prev]
            layers[i].initiate_weights(layers[i].nNodes, layers[i - 1].nNodes);
            // b's dimensions = [n_curr, m]
            layers[i].initiate_intercept(layers[i].nNodes, m);
            output = layers[i].forward();
            previousInput = output;
        }
        return output;
    }

    public void summary() {
        for (Layer layer : layers) {
            System.out.println(layer.toString());
        }
    }
}
