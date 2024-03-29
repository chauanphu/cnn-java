public abstract class Layer {
    // Each node represents a neuron in the layer
    int nNodes;
    Tensor2D output;
    Tensor2D input;
    Layer previousLayer = null;
    Layer nextLayer = null;

    Layer(int nNodes) {
        this.nNodes = nNodes;
    }

    public abstract Tensor2D forward();

    public abstract void init();

    @Override
    public String toString() {
        return "Number of nodes: " + nNodes;
    }
}

class InputLayer extends Layer {
    InputLayer(int nNodes, Tensor2D input) {
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

    DenseLayer(int nNodes) {
        super(nNodes);
    }

    @Override
    public Tensor2D forward() {
        // z = W * x + b; z's dimensions = [n_curr, m], W's dimensions =  [n_curr, n_prev], b's dimensions = [n_curr, m]
        Tensor2D input = previousLayer.output;
        Tensor2D z = this.weights.dot(input).add_vector(intercept);
        this.output = z;
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