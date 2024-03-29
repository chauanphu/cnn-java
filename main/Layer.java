public abstract class Layer {
    // Each node represents a neuron in the layer
    int nNodes;
    Tensor2D output;
    Tensor2D input;
    Layer previousLayer = null;
    Layer nextLayer = null;
    static String name;
    Layer(int nNodes) {
        this.nNodes = nNodes;
    }

    public abstract Tensor2D forward();

    public abstract void init();

    public String getName() {
        return name;
    }

    @Override
    public String toString() {
        return "Number of nodes: " + nNodes;
    }
}

class InputLayer extends Layer {
    static String name = "Input Layer";
    InputLayer(int nNodes) {
        super(nNodes);
    }

    @Override
    public Tensor2D forward() {
        output = input;
        return nextLayer.forward();
    }

    @Override
    public void init() {}
}

class DenseLayer extends Layer {

    Tensor2D weights;
    Tensor2D intercept;
    static String name = "Dense Layer";
    Activation activation;
    DenseLayer(int nNodes, Activation.ActivationType activationType) {
        super(nNodes);
        this.activation = new Activation(activationType);
    }
    
    @Override
    public Tensor2D forward() {
        System.out.println("Layer: " + name + " is forward");
        // z = W * x + b; z's dimensions = [n_curr, m], W's dimensions =  [n_curr, n_prev], b's dimensions = [n_curr, m]
        Tensor2D input = previousLayer.output;
        Tensor2D z = this.weights.dot(input).add_vector(intercept);
        Tensor2D a = activation.cal(z);
        this.output = a;
        if (nextLayer != null) {
            nextLayer.input = this.output;
            return nextLayer.forward();
        } else {
            return this.output;
        }
    }

    
    public void initiate_weights (int row, int col) {
        this.weights = Tensor2D.randoms(row, col);
    }

    public void initiate_intercept (int row, int col) {
        this.intercept = Tensor2D.randoms(row, col);
    }

    @Override
    public void init() {
        initiate_weights(nNodes, previousLayer.nNodes);
        initiate_intercept(nNodes, 1);
    }
}